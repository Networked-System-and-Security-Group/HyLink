// NCCL hook for LD_PRELOAD
// This file intercepts NCCL calls and routes them through AMP-CCL

#include "core/virtual_collective.h"
#include "core/domain_manager.h"
#include "core/comm_init.h"
#include "core/node_scope.h"
#include "core/stream_sync.h"
#include "backend/original_collective_api.h"
#include "common/op_key.h"
#include "common/config.h"
#include "common/log.h"
#include "common/intercept_policy.h"
#include "backend/dtype/native_dtype.h"
#include <dlfcn.h>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>

// NCCL types (forward declarations if headers not available)
#ifndef NCCL_H
#define NCCL_UNIQUE_ID_BYTES 128
typedef enum { ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat16, ncclFloat32, ncclFloat64 } ncclDataType_t;
typedef enum { ncclSum, ncclProd, ncclMax, ncclMin } ncclRedOp_t;
typedef void* ncclComm_t;
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;
#else
#define NCCL_UNIQUE_ID_BYTES sizeof(ncclUniqueId)
#endif

// CUDA stream / error types:
#if defined(__CUDA_RUNTIME_H__)
using ampccl_cudaStream_t = cudaStream_t;
using ampccl_cudaError_t  = cudaError_t;
#else
typedef void* cudaStream_t;
typedef void* ampccl_cudaStream_t;
typedef int   ampccl_cudaError_t;
#endif

// Forward declarations for original NCCL functions
typedef int (*ncclGetUniqueId_t)(ncclUniqueId* uniqueId);
typedef int (*ncclCommInitRank_t)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank);
typedef int (*ncclCommInitRankConfig_t)(
    ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, void* config);
typedef int (*ncclCommInitAll_t)(ncclComm_t* comms, int ndev, int* devlist);
typedef int (*ncclCommDestroy_t)(ncclComm_t comm);
typedef int (*ncclCommSplit_t)(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, void* config);

typedef int (*ncclAllReduce_t)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclAllGather_t)(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclReduceScatter_t)(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclReduce_t)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclBroadcast_t)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclAllToAll_t)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclSend_t)(
    const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
typedef int (*ncclRecv_t)(
    void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
typedef int (*ncclGroupStart_t)();
typedef int (*ncclGroupEnd_t)();
typedef int (*ncclCommCount_t)(ncclComm_t comm, int* count);
typedef int (*ncclCommUserRank_t)(ncclComm_t comm, int* rank);

// CUDA runtime: dlsym cudaStreamSynchronize for SyncDeviceStreamRaw only (not exported from this .so).
typedef ampccl_cudaError_t (*cudaStreamSynchronize_t)(ampccl_cudaStream_t stream);

// Function pointers to original NCCL functions
static ncclGetUniqueId_t orig_ncclGetUniqueId = nullptr;
static ncclCommInitRank_t orig_ncclCommInitRank = nullptr;
static ncclCommInitRankConfig_t orig_ncclCommInitRankConfig = nullptr;
static ncclCommInitAll_t orig_ncclCommInitAll = nullptr;
static ncclCommDestroy_t orig_ncclCommDestroy = nullptr;
static ncclCommSplit_t orig_ncclCommSplit = nullptr;
static ncclAllReduce_t orig_ncclAllReduce = nullptr;
static ncclAllGather_t orig_ncclAllGather = nullptr;
static ncclReduceScatter_t orig_ncclReduceScatter = nullptr;
static ncclReduce_t orig_ncclReduce = nullptr;
static ncclBroadcast_t orig_ncclBroadcast = nullptr;
static ncclAllToAll_t orig_ncclAllToAll = nullptr;
static ncclSend_t orig_ncclSend = nullptr;
static ncclRecv_t orig_ncclRecv = nullptr;
static ncclGroupStart_t orig_ncclGroupStart = nullptr;
static ncclGroupEnd_t orig_ncclGroupEnd = nullptr;
static ncclCommCount_t orig_ncclCommCount = nullptr;
static ncclCommUserRank_t orig_ncclCommUserRank = nullptr;
static cudaStreamSynchronize_t orig_cudaStreamSynchronize = nullptr;

// CUDA runtime: dlsym cudaEventSynchronize for SyncPCIeEndEventBeforeCallbackIfEnabled.
typedef ampccl_cudaError_t (*cudaEventSynchronize_t)(void* event);
static cudaEventSynchronize_t orig_cudaEventSynchronize = nullptr;
static std::atomic<uint64_t> g_nccl_init_all_seq{0};
static std::mutex g_nccl_comm_split_mutex;
static std::unordered_map<void*, uint64_t> g_nccl_comm_split_ordinals;

static uint64_t NextNcclCommSplitOrdinal(ncclComm_t parent) {
    std::lock_guard<std::mutex> lock(g_nccl_comm_split_mutex);
    uint64_t& ordinal = g_nccl_comm_split_ordinals[parent];
    return ordinal++;
}

static void EraseNcclCommSplitOrdinal(ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(g_nccl_comm_split_mutex);
    g_nccl_comm_split_ordinals.erase(comm);
}

// Load original NCCL functions
static void LoadOriginalFunctions() {
    static bool loaded = false;
    if (loaded) return;

    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("libnccl.so.2", RTLD_LAZY);
    }

    if (handle) {
        AMPCCL_LOG(INFO, "find libnccl.so, load nccl comm api success!");
        orig_ncclGetUniqueId = (ncclGetUniqueId_t)dlsym(handle, "ncclGetUniqueId");
        orig_ncclCommInitRank = (ncclCommInitRank_t)dlsym(handle, "ncclCommInitRank");
        orig_ncclCommInitRankConfig =
            (ncclCommInitRankConfig_t)dlsym(handle, "ncclCommInitRankConfig");
        orig_ncclCommInitAll = (ncclCommInitAll_t)dlsym(handle, "ncclCommInitAll");
        orig_ncclCommDestroy = (ncclCommDestroy_t)dlsym(handle, "ncclCommDestroy");
        orig_ncclCommSplit = (ncclCommSplit_t)dlsym(handle, "ncclCommSplit");
        orig_ncclAllReduce = (ncclAllReduce_t)dlsym(handle, "ncclAllReduce");
        orig_ncclAllGather = (ncclAllGather_t)dlsym(handle, "ncclAllGather");
        orig_ncclReduceScatter = (ncclReduceScatter_t)dlsym(handle, "ncclReduceScatter");
        orig_ncclReduce = (ncclReduce_t)dlsym(handle, "ncclReduce");
        orig_ncclBroadcast = (ncclBroadcast_t)dlsym(handle, "ncclBroadcast");
        orig_ncclAllToAll = (ncclAllToAll_t)dlsym(handle, "ncclAlltoAll");
        if (!orig_ncclAllToAll) {
            orig_ncclAllToAll = (ncclAllToAll_t)dlsym(handle, "ncclAllToAll");
        }
        orig_ncclSend = (ncclSend_t)dlsym(handle, "ncclSend");
        orig_ncclRecv = (ncclRecv_t)dlsym(handle, "ncclRecv");
        orig_ncclGroupStart = (ncclGroupStart_t)dlsym(handle, "ncclGroupStart");
        orig_ncclGroupEnd = (ncclGroupEnd_t)dlsym(handle, "ncclGroupEnd");
        orig_ncclCommCount = (ncclCommCount_t)dlsym(handle, "ncclCommCount");
        orig_ncclCommUserRank = (ncclCommUserRank_t)dlsym(handle, "ncclCommUserRank");
    }

    // CUDA runtime for internal SyncDeviceStreamRaw
    if (!orig_cudaStreamSynchronize) {
        void* cuda_handle = dlopen("libcudart.so", RTLD_LAZY);
        if (cuda_handle) {
            orig_cudaStreamSynchronize = (cudaStreamSynchronize_t)dlsym(cuda_handle, "cudaStreamSynchronize");
            orig_cudaEventSynchronize = (cudaEventSynchronize_t)dlsym(cuda_handle, "cudaEventSynchronize");
            AMPCCL_LOG(INFO, "find libcudart.so, load cudaStreamSynchronize success!");
        }
    }

    if (orig_ncclAllReduce || orig_ncclAllGather) {
        ampccl::OriginalCollectiveAPI::RegisterNccl(
            reinterpret_cast<void*>(orig_ncclAllReduce),
            reinterpret_cast<void*>(orig_ncclAllGather),
            reinterpret_cast<void*>(orig_ncclReduceScatter),
            reinterpret_cast<void*>(orig_ncclBroadcast),
            reinterpret_cast<void*>(orig_ncclAllToAll));
        ampccl::OriginalCollectiveAPI::RegisterNcclP2P(
            reinterpret_cast<void*>(orig_ncclSend),
            reinterpret_cast<void*>(orig_ncclRecv),
            reinterpret_cast<void*>(orig_ncclGroupStart),
            reinterpret_cast<void*>(orig_ncclGroupEnd),
            reinterpret_cast<void*>(orig_ncclCommCount),
            reinterpret_cast<void*>(orig_ncclCommUserRank),
            reinterpret_cast<void*>(orig_ncclReduce));
        // Native variable-count AllGather (NCCL >= 2.28 exports ncclAllGatherv).
        void* native_agv = dlsym(handle, "ncclAllGatherv");
        if (!native_agv) {
            native_agv = dlsym(handle, "ncclAllGatherV");
        }
        if (native_agv) {
            ampccl::OriginalCollectiveAPI::RegisterNcclNativeAllGatherV(native_agv);
            AMPCCL_LOG(INFO, "native ncclAllGatherv found: AllGatherV uses NCCL directly");
        } else {
            AMPCCL_LOG(INFO, "no native ncclAllGatherv: AllGatherV falls back to send/recv emulation");
        }
    }

    // Register synchronize API (intercepted but not re-exported)
    if (orig_cudaStreamSynchronize) {
        ampccl::OriginalCollectiveAPI::RegisterSynchronize(
            reinterpret_cast<void*>(orig_cudaStreamSynchronize));
    }
    if (orig_cudaEventSynchronize) {
        ampccl::OriginalCollectiveAPI::RegisterSynchronizeEvent(
            reinterpret_cast<void*>(orig_cudaEventSynchronize));
    }

    AMPCCL_LOG(INFO, "register nccl comm api success!\n"
        "allgather, allreduce, broadcast, reducescatter.\n"
        "adaptive enabled: %d\n"
        "enable pcie: %d\n",
        ampccl::Config::IsAdaptiveEnabled(), ampccl::Config::IsPCIeEnabled());

    loaded = true;
}

// Helper to convert NCCL result to BackendResult
static ampccl::BackendResult ConvertNCCLResult(int nccl_result) {
    if (nccl_result == 0) {
        return ampccl::BackendResult::Success;
    }
    return ampccl::BackendResult(ampccl::BackendResult::UnhandledError, nccl_result);
}

// Helper to convert BackendResult back to NCCL return code (for App-facing return)
static int BackendResultToNcclCode(const ampccl::BackendResult& r) {
    if (r.IsOk()) return 0;
    return (r.original != 0) ? r.original : -1;
}

// Look up domain by raw NCCL communicator (registered at CommInit).
static ampccl::CommDomain* GetDomainByRawComm(ncclComm_t comm) {
    return ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
}

static bool IsAmpcclRuntimeEnabled() {
    return ampccl::Config::IsAdaptiveEnabled() && !ampccl::Config::IsForceDisable();
}

static bool ApplyNodeScopeDecision(ampccl::CommDomain* domain, const ampccl::LogicalCommKey& logical_key,
                                   int rank, int nranks, bool rank_already_registered) {
    if (domain == nullptr) {
        return false;
    }
    bool allow = true;
    bool intra_node = true;
    int local_seen = nranks;
    if (ampccl::Config::IsIntraNodeOnlyEnabled() && !ampccl::Config::IsForceEnableInterNode()) {
        ampccl::NodeScopeResult scope;
        if (rank_already_registered) {
            scope = ampccl::ResolveNodeScope(logical_key, nranks, ampccl::Config::GetNodeScopeWaitMs());
        } else {
            scope = ampccl::DetectNodeScope(logical_key, rank, nranks, ampccl::Config::GetNodeScopeWaitMs());
        }
        allow = scope.allow_ampccl;
        intra_node = scope.intra_node;
        local_seen = scope.local_seen;
    }
    domain->set_node_scope(intra_node, allow);
    AMPCCL_LOG(INFO, "[Rank %d] AMP-CCL node scope: logical_hash=%llu nranks=%d local_seen=%d intra=%d allow=%d",
               rank, static_cast<unsigned long long>(logical_key.comm_hash), nranks, local_seen,
               intra_node ? 1 : 0, allow ? 1 : 0);
    return allow;
}

static void RegisterNcclCommAfterInit(ncclComm_t comm, int nranks, const ncclUniqueId& commId, int myrank) {
    if (comm == nullptr || !IsAmpcclRuntimeEnabled()) {
        return;
    }
    AMPCCL_LOG(INFO, "[Rank %d] register nccl comm: comm=%p nranks=%d", myrank, comm, nranks);
    ampccl::CommDomainKey key = ampccl::BuildKeyFromNcclInit(
        nranks, &commId, NCCL_UNIQUE_ID_BYTES, myrank);
    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromNcclInit(
        nranks, &commId, NCCL_UNIQUE_ID_BYTES);
    ampccl::DomainManager::GetInstance().RegisterRawComm(comm, key, logical_key);
    ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
    if (ApplyNodeScopeDecision(domain, logical_key, myrank, nranks, false)) {
        ampccl::InitPCIeForDomain(domain, myrank, nranks, &commId, sizeof(ncclUniqueId));
    }
}

void ampccl::SyncDeviceStreamRaw(void* stream) {
    if (orig_cudaStreamSynchronize && stream) {
        (void)orig_cudaStreamSynchronize(static_cast<ampccl_cudaStream_t>(stream));
    }
}

// Hooked NCCL functions
extern "C" {

int ncclGetUniqueId(ncclUniqueId* uniqueId) {
    LoadOriginalFunctions();
    if (orig_ncclGetUniqueId) {
        return orig_ncclGetUniqueId(uniqueId);
    }
    return -1;
}

int ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank) {
    LoadOriginalFunctions();
    if (!orig_ncclCommInitRank) {
        return -1;
    }
    int ret = orig_ncclCommInitRank(comm, nranks, commId, myrank);
    if (ret != 0 || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    RegisterNcclCommAfterInit(*comm, nranks, commId, myrank);
    return ret;
}

int ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, void* config) {
    LoadOriginalFunctions();
    if (!orig_ncclCommInitRankConfig) {
        return -1;
    }
    int ret = orig_ncclCommInitRankConfig(comm, nranks, commId, myrank, config);
    if (ret != 0 || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    RegisterNcclCommAfterInit(*comm, nranks, commId, myrank);
    return ret;
}

int ncclCommInitAll(ncclComm_t* comms, int ndev, int* devlist) {
    LoadOriginalFunctions();
    if (!orig_ncclCommInitAll) {
        return -1;
    }
    int ret = orig_ncclCommInitAll(comms, ndev, devlist);
    if (ret != 0 || comms == nullptr || ndev <= 0) {
        return ret;
    }
    if (!IsAmpcclRuntimeEnabled()) {
        return ret;
    }
    uint64_t init_all_nonce = g_nccl_init_all_seq.fetch_add(1, std::memory_order_relaxed);
    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromInitAll(ndev, init_all_nonce);
    for (int r = 0; r < ndev; ++r) {
        ncclComm_t comm = comms[r];
        if (comm == nullptr) {
            continue;
        }
        ampccl::CommDomainKey key = ampccl::BuildKeyFromNcclInitAll(ndev, r);
        key.topology_hash = logical_key.comm_hash;
        ampccl::DomainManager::GetInstance().RegisterRawComm(comm, key, logical_key);
        (void)ampccl::RegisterNodeScopeRank(logical_key, r, ndev);
    }
    for (int r = 0; r < ndev; ++r) {
        ncclComm_t comm = comms[r];
        if (comm == nullptr) {
            continue;
        }
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
        if (ApplyNodeScopeDecision(domain, logical_key, r, ndev, true)) {
            ampccl::InitPCIeForDomain(domain, r, ndev, nullptr, 0);
        }
    }
    return ret;
}

int ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, void* config) {
    LoadOriginalFunctions();
    if (!orig_ncclCommSplit) {
        return -1;
    }

    int ret = orig_ncclCommSplit(comm, color, key, newcomm, config);
    if (ret != 0) {
        return ret;
    }
    if (!IsAmpcclRuntimeEnabled()) {
        return ret;
    }

    ampccl::CommDomain* parent_domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
    if (parent_domain == nullptr) {
        AMPCCL_LOG(WARN, "ncclCommSplit child is not registered: parent raw comm %p has no AMP-CCL domain",
                   comm);
        return ret;
    }

    uint64_t split_ordinal = NextNcclCommSplitOrdinal(comm);
    if (newcomm == nullptr || *newcomm == nullptr) {
        AMPCCL_LOG(INFO, "ncclCommSplit produced no child comm: parent=%p color=%d ordinal=%llu",
                   comm, color, static_cast<unsigned long long>(split_ordinal));
        return ret;
    }
    if (!orig_ncclCommCount || !orig_ncclCommUserRank) {
        AMPCCL_LOG(WARN, "ncclCommSplit child is not registered: missing ncclCommCount/UserRank symbols");
        return ret;
    }

    int child_nranks = 0;
    int child_rank = -1;
    int count_ret = orig_ncclCommCount(*newcomm, &child_nranks);
    int rank_ret = orig_ncclCommUserRank(*newcomm, &child_rank);
    if (count_ret != 0 || rank_ret != 0 || child_nranks <= 0 ||
        child_rank < 0 || child_rank >= child_nranks) {
        AMPCCL_LOG(WARN,
                   "ncclCommSplit child is not registered: count_ret=%d rank_ret=%d child_rank=%d child_nranks=%d",
                   count_ret, rank_ret, child_rank, child_nranks);
        return ret;
    }

    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromCommSplit(
        parent_domain->logical_key, color, split_ordinal, child_nranks);
    ampccl::CommDomainKey domain_key = ampccl::BuildKeyFromNcclCommSplit(
        parent_domain->logical_key, color, split_ordinal, child_nranks, child_rank);
    ampccl::DomainManager::GetInstance().RegisterRawComm(*newcomm, domain_key, logical_key);
    ampccl::CommDomain* child_domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(*newcomm);
    AMPCCL_LOG(INFO,
               "[Rank %d] ncclCommSplit registered child: parent=%p child=%p color=%d ordinal=%llu child_nranks=%d",
               child_rank, comm, *newcomm, color,
               static_cast<unsigned long long>(split_ordinal), child_nranks);

    // Diagnostic: child (CommSplit) domains use pure NCCL.
    // Set AMPCCL_PCIE_CHILD_COMMS=1 to re-enable PCIe for children.
    {
        const char* pcie_child = std::getenv("AMPCCL_PCIE_CHILD_COMMS");
        bool child_pcie = (pcie_child && std::strcmp(pcie_child, "1") == 0);
        if (child_pcie && ApplyNodeScopeDecision(child_domain, logical_key, child_rank, child_nranks, false)) {
            uint64_t child_pccl_bootstrap[6] = {
                0x414d5043434c5350ULL,
                parent_domain->logical_key.comm_hash,
                static_cast<uint64_t>(parent_domain->logical_key.world_size),
                static_cast<uint64_t>(child_nranks),
                static_cast<uint64_t>(static_cast<int64_t>(color)),
                split_ordinal,
            };
            ampccl::InitPCIeForDomain(child_domain, child_rank, child_nranks,
                                      child_pccl_bootstrap, sizeof(child_pccl_bootstrap));
            AMPCCL_LOG(INFO, "[Rank %d] ncclCommSplit child: PCIe enabled", child_rank);
        } else {
            (void)ApplyNodeScopeDecision(child_domain, logical_key, child_rank, child_nranks, false);
            child_domain->set_node_scope(true, false);
            AMPCCL_LOG(INFO, "[Rank %d] ncclCommSplit child: pure NCCL", child_rank);
        }
    }
    return ret;
}

int ncclCommDestroy(ncclComm_t comm) {
    LoadOriginalFunctions();
    if (IsAmpcclRuntimeEnabled()) {
        EraseNcclCommSplitOrdinal(comm);
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
        if (domain) {
            domain->PrintAvgBw();
            ampccl::DestroyPCIeForDomain(domain);
        }
        ampccl::DomainManager::GetInstance().UnregisterRawComm(comm);
    }
    if (orig_ncclCommDestroy) {
        return orig_ncclCommDestroy(comm);
    }
    return -1;
}

int ncclAllReduce(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_ncclAllReduce) {
        return orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        int ret = -1;
        if (orig_ncclAllReduce) {
            ret = orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }
        AMPCCL_LOG(INFO, "ncclAllReduce comm=%p domain=%p allow=%d bytes=%zu ret=%d",
                   comm, (void*)domain, domain ? (int)domain->allow_ampccl() : -1,
                   count * 4ULL, ret);
        return ret;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::AllReduce, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)), 0, 0, count)) {
        return orig_ncclAllReduce
            ? orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
            : -1;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllReduce(
        domain, sendbuff, recvbuff, count,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return BackendResultToNcclCode(result);
}

int ncclAllGather(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_ncclAllGather) {
        return orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        if (orig_ncclAllGather) {
            return orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        }
        return -1;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::AllGather, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)), sendcount, 0, 0)) {
        return orig_ncclAllGather
            ? orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
            : -1;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllGather(
        domain, sendbuff, recvbuff, sendcount,
        static_cast<int>(datatype), comm, stream);

    return BackendResultToNcclCode(result);
}

int ncclReduceScatter(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_ncclReduceScatter) {
        return orig_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        if (orig_ncclReduceScatter) {
            return orig_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
        }
        return -1;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::ReduceScatter, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)), 0, recvcount, 0)) {
        return orig_ncclReduceScatter
            ? orig_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
            : -1;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::ReduceScatter(
        domain, sendbuff, recvbuff, recvcount,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return BackendResultToNcclCode(result);
}

int ncclAllToAll(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_ncclAllToAll) {
        return orig_ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        if (orig_ncclAllToAll) {
            return orig_ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
        }
        return -1;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::AllToAll, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)), 0, 0, count)) {
        return orig_ncclAllToAll
            ? orig_ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream)
            : -1;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllToAll(
        domain, sendbuff, recvbuff, count, static_cast<int>(datatype), comm, stream);

    return BackendResultToNcclCode(result);
}

int ncclAlltoAll(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

int ncclAllGatherV(
    const void* sendbuff, size_t sendcount, void* recvbuff,
    const size_t* recvcounts, const size_t* rdispls,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    ampccl::BackendResult result = ampccl::OriginalCollectiveAPI::NcclAllGatherV(
        sendbuff, sendcount, recvbuff, recvcounts, rdispls,
        static_cast<int>(datatype), comm, stream);
    return BackendResultToNcclCode(result);
}

int ncclAllToAllV(
    const void* sendbuff, const size_t* sendcounts, const size_t* sdispls,
    void* recvbuff, const size_t* recvcounts, const size_t* rdispls,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    ampccl::BackendResult result = ampccl::OriginalCollectiveAPI::NcclAllToAllV(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
        static_cast<int>(datatype), comm, stream);
    return BackendResultToNcclCode(result);
}

int ncclReduceScatterV(
    const void* sendbuff, const size_t* sendcounts, const size_t* sdispls,
    void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    ampccl::BackendResult result = ampccl::OriginalCollectiveAPI::NcclReduceScatterV(
        sendbuff, sendcounts, sdispls, recvbuff, recvcount,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);
    return BackendResultToNcclCode(result);
}

int ncclBroadcast(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (orig_ncclBroadcast) {
        return orig_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    }
    return -1;
}

}  // extern "C"
