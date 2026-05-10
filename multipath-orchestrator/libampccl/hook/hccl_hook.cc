#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// HCCL hook for LD_PRELOAD
// This file intercepts HCCL calls and routes them through AMP-CCL

#include "core/virtual_collective.h"
#include "core/domain_manager.h"
#include "core/comm_init.h"
#include "core/node_scope.h"
#include "core/stream_sync.h"
#include "backend/original_collective_api.h"
#include "common/op_key.h"
#include "common/config.h"
#include "common/intercept_policy.h"
#include "backend/dtype/native_dtype.h"
#include <dlfcn.h>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <cstddef>
#include <link.h>

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
#define HCCL_ROOT_INFO_BYTES 4108

typedef enum {
    HCCL_SUCCESS = 0,               /* success */
    HCCL_E_PARA = 1,                /* parameter error */
    HCCL_E_PTR = 2,                 /* empty pointer */
    HCCL_E_MEMORY = 3,              /* memory error */
    HCCL_E_INTERNAL = 4,            /* internal error */
    HCCL_E_NOT_SUPPORT = 5,         /* not support feature */
    HCCL_E_NOT_FOUND = 6,           /* not found specific resource */
    HCCL_E_UNAVAIL = 7,             /* resource unavailable */
    HCCL_E_SYSCALL = 8,             /* call system interface error */
    HCCL_E_TIMEOUT = 9,             /* timeout */
    HCCL_E_OPEN_FILE_FAILURE = 10,  /* open file fail */
    HCCL_E_TCP_CONNECT = 11,        /* tcp connect fail */
    HCCL_E_ROCE_CONNECT = 12,       /* roce connect fail */
    HCCL_E_TCP_TRANSFER = 13,       /* tcp transfer fail */
    HCCL_E_ROCE_TRANSFER = 14,      /* roce transfer fail */
    HCCL_E_RUNTIME = 15,            /* call runtime api fail */
    HCCL_E_DRV = 16,                /* call driver api fail */
    HCCL_E_PROFILING = 17,          /* call profiling api fail */
    HCCL_E_CCE = 18,                /* call cce api fail */
    HCCL_E_NETWORK = 19,            /* call network api fail */
    HCCL_E_AGAIN = 20,              /* try again */
    HCCL_E_REMOTE = 21,             /* error cqe */
    HCCL_E_SUSPENDING = 22,         /* error communicator suspending */
    HCCL_E_OPRETRY_FAIL = 23,       /* retry constraint */
    HCCL_E_OOM = 24,                /* out of memory */
} hcclResult_t;

typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,
    HCCL_DATA_TYPE_INT16 = 1,
    HCCL_DATA_TYPE_INT32 = 2,
    HCCL_DATA_TYPE_FP16 = 3,
    HCCL_DATA_TYPE_FP32 = 4,
    HCCL_DATA_TYPE_INT64 = 5,
    HCCL_DATA_TYPE_UINT64 = 6,
    HCCL_DATA_TYPE_UINT8 = 7,
    HCCL_DATA_TYPE_UINT16 = 8,
    HCCL_DATA_TYPE_UINT32 = 9,
    HCCL_DATA_TYPE_FP64 = 10,
    HCCL_DATA_TYPE_BFP16 = 11,
    HCCL_DATA_TYPE_INT128 = 12,
    HCCL_DATA_TYPE_FLOAT = HCCL_DATA_TYPE_FP32,
    HCCL_DATA_TYPE_FLOAT16 = HCCL_DATA_TYPE_FP16,
    HCCL_DATA_TYPE_RESERVED = 255
} HcclDataType;

typedef enum {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_PROD = 1,
    HCCL_REDUCE_MAX = 2,
    HCCL_REDUCE_MIN = 3,
    HCCL_REDUCE_RESERVED = 255
} HcclReduceOp;

typedef void* HcclComm;
typedef void* aclrtStream;

typedef struct { char internal[HCCL_ROOT_INFO_BYTES]; } HcclRootInfo;
typedef struct HcclCommConfig HcclCommConfig;

// Forward declarations for original HCCL functions
typedef hcclResult_t (*hcclCommInitRootInfo_t)(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
typedef hcclResult_t (*hcclCommInitRootInfoConfig_t)(uint32_t nRanks, const HcclRootInfo* rootInfo,
                                                     uint32_t rank, const HcclCommConfig* config, HcclComm* comm);
typedef hcclResult_t (*hcclCommInitClusterInfo_t)(const char* clusterInfo, uint32_t rank, HcclComm* comm);
typedef hcclResult_t (*hcclCommInitClusterInfoConfig_t)(const char* clusterInfo, uint32_t rank,
                                                        HcclCommConfig* config, HcclComm* comm);
typedef hcclResult_t (*hcclCommInitAll_t)(unsigned int ndev, int* devices, HcclComm* comms);
typedef hcclResult_t (*hcclCommDestroy_t)(HcclComm comm);
typedef hcclResult_t (*hcclGetRootInfo_t)(HcclRootInfo* rootInfo);
typedef hcclResult_t (*hcclCreateSubCommConfig_t)(
    HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId,
    uint32_t subCommRankId, HcclCommConfig* config, HcclComm* subComm);
typedef hcclResult_t (*hcclGetRankSize_t)(HcclComm comm, uint32_t* rankSize);
typedef hcclResult_t (*hcclGetRankId_t)(HcclComm comm, uint32_t* rank);

typedef hcclResult_t (*hcclAllReduce_t)(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclAllGather_t)(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream);

// Matches hccl.h: HcclReduceScatter(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream)
typedef hcclResult_t (*hcclReduceScatter_t)(
    void* sendbuff, void* recvbuff, unsigned long recvcount,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream);

// Matches hccl.h: HcclAlltoAll(const void*, uint64_t, HcclDataType, const void*, uint64_t, HcclDataType, HcclComm, aclrtStream)
typedef hcclResult_t (*hcclAlltoAll_t)(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream);
// Legacy 6-arg HcclAllToAll: symmetric per-peer count/dtype for send and recv.
typedef hcclResult_t (*hcclAlltoAll_6arg_t)(
    const void* sendBuf, void* recvBuf, uint64_t count,
    HcclDataType datatype, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclAllGatherV_t)(void* sendBuf, uint64_t sendCount, void* recvBuf,
    const void* recvCounts, const void* recvDispls, HcclDataType dataType, HcclComm comm,
    aclrtStream stream);

typedef hcclResult_t (*hcclReduceScatterV_t)(void* sendBuf, const void* sendCounts,
    const void* sendDispls, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclAlltoAllV_t)(const void* sendBuf, const void* sendCounts,
    const void* sdispls, HcclDataType sendType, const void* recvBuf, const void* recvCounts,
    const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream);

// ACL runtime: dlsym aclrtSynchronizeStream for SyncDeviceStreamRaw only (not exported from this .so).
typedef int (*aclrtSynchronizeStream_t)(aclrtStream stream);

// Function pointers to original HCCL functions
static hcclCommInitRootInfo_t orig_hcclCommInitRootInfo = nullptr;
static hcclCommInitRootInfoConfig_t orig_hcclCommInitRootInfoConfig = nullptr;
static hcclCommInitClusterInfo_t orig_hcclCommInitClusterInfo = nullptr;
static hcclCommInitClusterInfoConfig_t orig_hcclCommInitClusterInfoConfig = nullptr;
static hcclCommInitAll_t orig_hcclCommInitAll = nullptr;
static hcclCommDestroy_t orig_hcclCommDestroy = nullptr;
static hcclGetRootInfo_t orig_hcclGetRootInfo = nullptr;
static hcclCreateSubCommConfig_t orig_hcclCreateSubCommConfig = nullptr;
static hcclGetRankSize_t orig_hcclGetRankSize = nullptr;
static hcclGetRankId_t orig_hcclGetRankId = nullptr;
static hcclAllReduce_t orig_hcclAllReduce = nullptr;
static hcclAllGather_t orig_hcclAllGather = nullptr;
static hcclReduceScatter_t orig_hcclReduceScatter = nullptr;
static hcclAlltoAll_t orig_hcclAlltoAll = nullptr;
static hcclAlltoAll_6arg_t orig_hcclAlltoAll_6arg = nullptr;
static hcclAllGatherV_t orig_hcclAllGatherV = nullptr;
static hcclReduceScatterV_t orig_hcclReduceScatterV = nullptr;
static hcclAlltoAllV_t orig_hcclAlltoAllV = nullptr;
static aclrtSynchronizeStream_t orig_aclrtSynchronizeStream = nullptr;

// ACL runtime: dlsym aclrtSynchronizeEvent for SyncPCIeEndEventBeforeCallbackIfEnabled.
typedef int (*aclrtSynchronizeEvent_t)(void* event);
static aclrtSynchronizeEvent_t orig_aclrtSynchronizeEvent = nullptr;
static std::atomic<uint64_t> g_hccl_init_all_seq{0};

using real_dlsym_t = void* (*)(void*, const char*);

static real_dlsym_t GetRealDlsym() {
    static real_dlsym_t fn = reinterpret_cast<real_dlsym_t>(
        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17"));
    return fn;
}

static void* RawDlsym(void* handle, const char* symbol) {
    real_dlsym_t fn = GetRealDlsym();
    return fn == nullptr ? nullptr : fn(handle, symbol);
}

// Load original HCCL functions
static void LoadOriginalFunctions() {
    static bool loaded = false;
    if (loaded) return;

    void* handle = dlopen("libhccl.so", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("libhccl.so.1", RTLD_LAZY);
    }

    if (handle) {
        AMPCCL_LOG(INFO,"find libhccl.so, load hccl comm api success!");
        orig_hcclCommInitAll = (hcclCommInitAll_t)RawDlsym(handle, "HcclCommInitAll");
        orig_hcclCommInitRootInfo = (hcclCommInitRootInfo_t)RawDlsym(handle, "HcclCommInitRootInfo");
        orig_hcclCommInitRootInfoConfig =
            (hcclCommInitRootInfoConfig_t)RawDlsym(handle, "HcclCommInitRootInfoConfig");
        orig_hcclCommInitClusterInfo =
            (hcclCommInitClusterInfo_t)RawDlsym(handle, "HcclCommInitClusterInfo");
        orig_hcclCommInitClusterInfoConfig =
            (hcclCommInitClusterInfoConfig_t)RawDlsym(handle, "HcclCommInitClusterInfoConfig");
        orig_hcclCommDestroy = (hcclCommDestroy_t)RawDlsym(handle, "HcclCommDestroy");
        orig_hcclGetRootInfo = (hcclGetRootInfo_t)RawDlsym(handle, "HcclGetRootInfo");
        orig_hcclCreateSubCommConfig =
            (hcclCreateSubCommConfig_t)RawDlsym(handle, "HcclCreateSubCommConfig");
        if (orig_hcclCreateSubCommConfig == nullptr) {
            orig_hcclCreateSubCommConfig =
                (hcclCreateSubCommConfig_t)RawDlsym(handle, "hcclCreateSubCommConfig");
        }
        orig_hcclGetRankSize = (hcclGetRankSize_t)RawDlsym(handle, "HcclGetRankSize");
        orig_hcclGetRankId = (hcclGetRankId_t)RawDlsym(handle, "HcclGetRankId");
        orig_hcclAllReduce = (hcclAllReduce_t)RawDlsym(handle, "HcclAllReduce");
        orig_hcclAllGather = (hcclAllGather_t)RawDlsym(handle, "HcclAllGather");
        orig_hcclReduceScatter = (hcclReduceScatter_t)RawDlsym(handle, "HcclReduceScatter");
        // Load 6-arg legacy first (older HCCL SDKs only export this name),
        // then canonical 8-arg HcclAlltoAll. Each stored in its own typed pointer.
        orig_hcclAlltoAll_6arg = (hcclAlltoAll_6arg_t)RawDlsym(handle, "HcclAllToAll");
        orig_hcclAlltoAll = (hcclAlltoAll_t)RawDlsym(handle, "HcclAlltoAll");
        orig_hcclAllGatherV = (hcclAllGatherV_t)RawDlsym(handle, "HcclAllGatherV");
        orig_hcclReduceScatterV = (hcclReduceScatterV_t)RawDlsym(handle, "HcclReduceScatterV");
        orig_hcclAlltoAllV = (hcclAlltoAllV_t)RawDlsym(handle, "HcclAlltoAllV");
    }

    // ACL runtime for internal SyncDeviceStreamRaw (may be in libascendcl/libacl)
    if (!orig_aclrtSynchronizeStream) {
        void* acl_handle = dlopen("libascendcl.so", RTLD_LAZY);
        if (!acl_handle) {
            acl_handle = dlopen("libacl.so", RTLD_LAZY);
        }
        if (acl_handle) {
            AMPCCL_LOG(INFO,"find libascendcl.so, load aclrt api success!");
            orig_aclrtSynchronizeStream =
                (aclrtSynchronizeStream_t)RawDlsym(acl_handle, "aclrtSynchronizeStream");
            orig_aclrtSynchronizeEvent =
                (aclrtSynchronizeEvent_t)RawDlsym(acl_handle, "aclrtSynchronizeEvent");
        }
    }

    if (orig_hcclAllReduce || orig_hcclAllGather) {
        ampccl::OriginalCollectiveAPI::RegisterHccl(
            reinterpret_cast<void*>(orig_hcclAllReduce),
            reinterpret_cast<void*>(orig_hcclAllGather),
            reinterpret_cast<void*>(orig_hcclReduceScatter),
            reinterpret_cast<void*>(orig_hcclAlltoAll),         // 8-arg (may be null)
            reinterpret_cast<void*>(orig_hcclAlltoAll_6arg),    // 6-arg (may be null)
            reinterpret_cast<void*>(orig_hcclAllGatherV),
            reinterpret_cast<void*>(orig_hcclReduceScatterV),
            reinterpret_cast<void*>(orig_hcclAlltoAllV));
    }

    // Register synchronize API (intercepted but not re-exported)
    if (orig_aclrtSynchronizeStream) {
        ampccl::OriginalCollectiveAPI::RegisterSynchronize(
            reinterpret_cast<void*>(orig_aclrtSynchronizeStream));
    }
    if (orig_aclrtSynchronizeEvent) {
        ampccl::OriginalCollectiveAPI::RegisterSynchronizeEvent(
            reinterpret_cast<void*>(orig_aclrtSynchronizeEvent));
    }

    AMPCCL_LOG(INFO, "register hccl comm api success!\n"
        "allgather, allreduce, reducescatter, alltoall; V via dlsym for hybrid split (not LD_PRELOAD-exported).\n"
        "adaptive enabled: %d\n"
        "enable pcie: %d\n",
        ampccl::Config::IsAdaptiveEnabled(), ampccl::Config::IsPCIeEnabled());

    loaded = true;
}

// Helper to convert HCCL result to BackendResult
static ampccl::BackendResult ConvertHCCLResult(hcclResult_t hccl_result) {
    if (hccl_result == HCCL_SUCCESS) {
        return ampccl::BackendResult::Success;
    }
    return ampccl::BackendResult(ampccl::BackendResult::UnhandledError, static_cast<int>(hccl_result));
}

// Helper to convert BackendResult back to HCCL return code (for App-facing return)
static hcclResult_t BackendResultToHcclCode(const ampccl::BackendResult& r) {
    if (r.IsOk()) return HCCL_SUCCESS;
    return (r.original != 0) ? static_cast<hcclResult_t>(r.original) : HCCL_E_PARA;
}

// Look up domain by raw HCCL communicator (registered at CommInit).
static ampccl::CommDomain* GetDomainByRawComm(HcclComm comm) {
    return ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
}

static bool IsAmpcclRuntimeEnabled() {
    return ampccl::Config::IsAdaptiveEnabled() && !ampccl::Config::IsForceDisable();
}

static uint64_t HashHcclRankIds(const uint32_t* rank_ids, uint32_t rank_num) {
    if (rank_ids == nullptr || rank_num == 0) {
        return 0;
    }
    return ampccl::HashBytesForComm(rank_ids, sizeof(uint32_t) * static_cast<size_t>(rank_num));
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

static void RegisterHcclDomain(HcclComm comm, int nranks, int rank,
                               const ampccl::LogicalCommKey& logical_key,
                               const void* pcie_bootstrap, size_t pcie_bootstrap_len,
                               const char* init_api) {
    if (!IsAmpcclRuntimeEnabled() || comm == nullptr || nranks < 2 || rank < 0 || rank >= nranks) {
        return;
    }

    const char* env_target_tp = std::getenv("AMPCCL_TARGET_TP_SIZE");
    if (env_target_tp != nullptr && env_target_tp[0] != '\0') {
        int target_tp = std::atoi(env_target_tp);
        if (target_tp > 0 && nranks != target_tp) {
            AMPCCL_LOG(INFO,
                       "[Rank %d] Skipping HCCL domain registration: world_size=%d != target_tp=%d",
                       rank, nranks, target_tp);
            return;
        }
    }

    ampccl::CommDomainKey domain_key;
    domain_key.world_size = nranks;
    domain_key.ranks.assign(1, rank);
    domain_key.topology_hash = logical_key.comm_hash;
    ampccl::DomainManager::GetInstance().RegisterRawComm(comm, domain_key, logical_key);
    ampccl::CommDomain* domain =
        ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
    bool allow = ApplyNodeScopeDecision(domain, logical_key, rank, nranks, false);
    bool pcie_init = false;
    if (allow) {
        ampccl::InitPCIeForDomain(domain, rank, nranks, pcie_bootstrap, pcie_bootstrap_len);
        pcie_init = domain != nullptr && domain->pcie_comm() != nullptr;
    }
    AMPCCL_LOG(INFO,
               "[Rank %d] HCCL domain registered: api=%s raw=%p world_size=%d logical_hash=%llu "
               "allow=%d pcie_init=%d",
               rank, init_api == nullptr ? "unknown" : init_api, comm, nranks,
               static_cast<unsigned long long>(logical_key.comm_hash), allow ? 1 : 0,
               pcie_init ? 1 : 0);
}

static bool QueryHcclCommInfo(HcclComm comm, uint32_t* nranks, uint32_t* rank) {
    if (comm == nullptr || nranks == nullptr || rank == nullptr ||
        orig_hcclGetRankSize == nullptr || orig_hcclGetRankId == nullptr) {
        return false;
    }
    return orig_hcclGetRankSize(comm, nranks) == HCCL_SUCCESS &&
           orig_hcclGetRankId(comm, rank) == HCCL_SUCCESS;
}

void ampccl::SyncDeviceStreamRaw(void* stream) {
    if (orig_aclrtSynchronizeStream && stream) {
        (void)orig_aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
    }
}

// Hooked HCCL functions
extern "C" {

hcclResult_t HcclGetRootInfo(HcclRootInfo* rootInfo) {
    LoadOriginalFunctions();
    AMPCCL_LOG(INFO, "HcclGetRootInfo begin: rootInfo=%p", static_cast<void*>(rootInfo));
    if (!orig_hcclGetRootInfo) {
        AMPCCL_LOG(ERROR, "HcclGetRootInfo missing original symbol");
        return HCCL_E_PARA;
    }
    hcclResult_t ret = orig_hcclGetRootInfo(rootInfo);
    AMPCCL_LOG(INFO, "HcclGetRootInfo end: ret=%d rootInfo=%p", static_cast<int>(ret),
               static_cast<void*>(rootInfo));
    return ret;
}

hcclResult_t HcclCreateSubCommConfig(HcclComm* comm, uint32_t rankNum, uint32_t* rankIds,
                                     uint64_t subCommId, uint32_t subCommRankId,
                                     HcclCommConfig* config, HcclComm* subComm) {
    LoadOriginalFunctions();
    AMPCCL_LOG(INFO,
               "[Rank %u] HcclCreateSubCommConfig begin: parent_ptr=%p parent=%p rankNum=%u "
               "rankIds=%p subCommId=%llu subComm_out=%p",
               subCommRankId, static_cast<void*>(comm), comm != nullptr ? *comm : nullptr, rankNum,
               static_cast<void*>(rankIds), static_cast<unsigned long long>(subCommId),
               static_cast<void*>(subComm));
    if (!orig_hcclCreateSubCommConfig) {
        AMPCCL_LOG(ERROR, "HcclCreateSubCommConfig missing original symbol");
        return HCCL_E_NOT_SUPPORT;
    }

    hcclResult_t ret = orig_hcclCreateSubCommConfig(
        comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
    AMPCCL_LOG(INFO,
               "[Rank %u] HcclCreateSubCommConfig original end: ret=%d child=%p",
               subCommRankId, static_cast<int>(ret), subComm != nullptr ? *subComm : nullptr);
    if (ret != HCCL_SUCCESS || subComm == nullptr || *subComm == nullptr) {
        return ret;
    }
    if (!IsAmpcclRuntimeEnabled()) {
        return ret;
    }
    if (rankNum < 2) {
        AMPCCL_LOG(INFO,
                   "[Rank %u] HCCL singleton subcomm remains on original HCCL path: child=%p",
                   subCommRankId, static_cast<void*>(*subComm));
        return ret;
    }
    if (comm == nullptr || *comm == nullptr || rankNum == 0 || subCommRankId >= rankNum) {
        AMPCCL_LOG(WARN,
                   "[Rank %u] HCCL subcomm child is not registered: invalid args parent=%p "
                   "rankNum=%u child_rank=%u",
                   subCommRankId, comm != nullptr ? *comm : nullptr, rankNum, subCommRankId);
        return ret;
    }

    ampccl::CommDomain* parent_domain =
        ampccl::DomainManager::GetInstance().GetDomainByRawComm(*comm);
    if (parent_domain == nullptr) {
        AMPCCL_LOG(WARN,
                   "[Rank %u] HCCL subcomm child is not registered: parent raw comm %p has no AMP-CCL domain",
                   subCommRankId, *comm);
        return ret;
    }

    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromHcclSubComm(
        parent_domain->logical_key, rankNum, rankIds, subCommId);
    ampccl::CommDomainKey domain_key = ampccl::BuildKeyFromHcclSubComm(
        parent_domain->logical_key, rankNum, rankIds, subCommId, subCommRankId);
    ampccl::DomainManager::GetInstance().RegisterRawComm(*subComm, domain_key, logical_key);
    ampccl::CommDomain* child_domain =
        ampccl::DomainManager::GetInstance().GetDomainByRawComm(*subComm);

    const uint64_t rank_ids_hash = HashHcclRankIds(rankIds, rankNum);
    bool allow = ApplyNodeScopeDecision(child_domain, logical_key, static_cast<int>(subCommRankId),
                                        static_cast<int>(rankNum), false);
    bool pcie_init = false;
    if (allow) {
        AMPCCL_LOG(INFO,
                   "[Rank %u] HCCL subcomm PCCL init begin: child=%p rankNum=%u child_hash=%llu",
                   subCommRankId, *subComm, rankNum,
                   static_cast<unsigned long long>(logical_key.comm_hash));
        uint64_t pcie_bootstrap[2] = {
            static_cast<uint64_t>(logical_key.world_size),
            logical_key.comm_hash,
        };
        ampccl::InitPCIeForDomain(child_domain, static_cast<int>(subCommRankId),
                                  static_cast<int>(rankNum), pcie_bootstrap, sizeof(pcie_bootstrap));
        pcie_init = child_domain != nullptr && child_domain->pcie_comm() != nullptr;
        AMPCCL_LOG(INFO,
                   "[Rank %u] HCCL subcomm PCCL init end: child=%p rankNum=%u child_hash=%llu pcie_init=%d",
                   subCommRankId, *subComm, rankNum,
                   static_cast<unsigned long long>(logical_key.comm_hash), pcie_init ? 1 : 0);
    }

    AMPCCL_LOG(INFO,
               "[Rank %u] HCCL subcomm registered: parent=%p child=%p parent_hash=%llu "
               "child_hash=%llu rankNum=%u subCommRankId=%u subCommId=%llu rankIdsHash=%llu "
               "allow=%d pcie_init=%d",
               subCommRankId, *comm, *subComm,
               static_cast<unsigned long long>(parent_domain->logical_key.comm_hash),
               static_cast<unsigned long long>(logical_key.comm_hash), rankNum, subCommRankId,
               static_cast<unsigned long long>(subCommId),
               static_cast<unsigned long long>(rank_ids_hash), allow ? 1 : 0, pcie_init ? 1 : 0);
    return ret;
}

// Older integrations used the lowercase spelling. Keep it as a compatibility
// entry point, while CANN 9.x and torch_npu use HcclCreateSubCommConfig.
hcclResult_t hcclCreateSubCommConfig(HcclComm* comm, uint32_t rankNum, uint32_t* rankIds,
                                     uint64_t subCommId, uint32_t subCommRankId,
                                     HcclCommConfig* config, HcclComm* subComm) {
    return HcclCreateSubCommConfig(comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
}

hcclResult_t HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm) {
    LoadOriginalFunctions();
    AMPCCL_LOG(INFO, "[Rank %u] HcclCommInitRootInfo begin: nRanks=%u rootInfo=%p comm_out=%p",
               rank, nRanks, static_cast<const void*>(rootInfo), static_cast<void*>(comm));
    if (!orig_hcclCommInitRootInfo) {
        AMPCCL_LOG(ERROR, "[Rank %u] HcclCommInitRootInfo missing original symbol", rank);
        return HCCL_E_PARA;
    }
    AMPCCL_LOG(INFO, "[Rank %u] HcclCommInitRootInfo original begin", rank);
    hcclResult_t ret = orig_hcclCommInitRootInfo(nRanks, rootInfo, rank, comm);
    AMPCCL_LOG(INFO, "[Rank %u] HcclCommInitRootInfo original end: ret=%d comm=%p",
               rank, static_cast<int>(ret), (comm != nullptr ? *comm : nullptr));
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    // AMPCCL_LOG(INFO,"[Rank %d] Exec original HcclCommInitRootInfo finished", rank);
    if (!IsAmpcclRuntimeEnabled()) {
        return ret;
    }
    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromHcclInitRootInfo(
        static_cast<int>(nRanks), rootInfo, sizeof(HcclRootInfo));
    RegisterHcclDomain(*comm, static_cast<int>(nRanks), static_cast<int>(rank), logical_key,
                       rootInfo, rootInfo == nullptr ? 0 : sizeof(HcclRootInfo),
                       "HcclCommInitRootInfo");
    AMPCCL_LOG(INFO,"[Rank %d] AMP-CCL HcclCommInitRootInfo finished", rank);
    return ret;
}

hcclResult_t HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo,
                                        uint32_t rank, const HcclCommConfig* config, HcclComm* comm) {
    LoadOriginalFunctions();
    AMPCCL_LOG(INFO, "[Rank %u] HcclCommInitRootInfoConfig begin: nRanks=%u comm_out=%p",
               rank, nRanks, static_cast<void*>(comm));
    if (orig_hcclCommInitRootInfoConfig == nullptr) {
        AMPCCL_LOG(ERROR, "HcclCommInitRootInfoConfig missing original symbol");
        return HCCL_E_NOT_SUPPORT;
    }
    hcclResult_t ret = orig_hcclCommInitRootInfoConfig(nRanks, rootInfo, rank, config, comm);
    AMPCCL_LOG(INFO, "[Rank %u] HcclCommInitRootInfoConfig original end: ret=%d comm=%p",
               rank, static_cast<int>(ret), comm == nullptr ? nullptr : *comm);
    if (ret == HCCL_SUCCESS && comm != nullptr && *comm != nullptr) {
        ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromHcclInitRootInfo(
            static_cast<int>(nRanks), rootInfo, sizeof(HcclRootInfo));
        RegisterHcclDomain(*comm, static_cast<int>(nRanks), static_cast<int>(rank), logical_key,
                           rootInfo, rootInfo == nullptr ? 0 : sizeof(HcclRootInfo),
                           "HcclCommInitRootInfoConfig");
    }
    return ret;
}

hcclResult_t HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm) {
    LoadOriginalFunctions();
    if (orig_hcclCommInitClusterInfo == nullptr) {
        AMPCCL_LOG(ERROR, "HcclCommInitClusterInfo missing original symbol");
        return HCCL_E_NOT_SUPPORT;
    }
    hcclResult_t ret = orig_hcclCommInitClusterInfo(clusterInfo, rank, comm);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    uint32_t nranks = 0;
    uint32_t comm_rank = rank;
    if (!QueryHcclCommInfo(*comm, &nranks, &comm_rank)) {
        AMPCCL_LOG(WARN, "[Rank %u] HcclCommInitClusterInfo cannot query communicator size", rank);
        return ret;
    }
    size_t info_len = clusterInfo == nullptr ? 0 : std::strlen(clusterInfo);
    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromBootstrap(
        static_cast<int>(nranks), clusterInfo, info_len, 0x48434c5f434c5553ULL);
    RegisterHcclDomain(*comm, static_cast<int>(nranks), static_cast<int>(comm_rank), logical_key,
                       clusterInfo, info_len, "HcclCommInitClusterInfo");
    return ret;
}

hcclResult_t HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank,
                                           HcclCommConfig* config, HcclComm* comm) {
    LoadOriginalFunctions();
    if (orig_hcclCommInitClusterInfoConfig == nullptr) {
        AMPCCL_LOG(ERROR, "HcclCommInitClusterInfoConfig missing original symbol");
        return HCCL_E_NOT_SUPPORT;
    }
    hcclResult_t ret = orig_hcclCommInitClusterInfoConfig(clusterInfo, rank, config, comm);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    uint32_t nranks = 0;
    uint32_t comm_rank = rank;
    if (!QueryHcclCommInfo(*comm, &nranks, &comm_rank)) {
        AMPCCL_LOG(WARN, "[Rank %u] HcclCommInitClusterInfoConfig cannot query communicator size", rank);
        return ret;
    }
    size_t info_len = clusterInfo == nullptr ? 0 : std::strlen(clusterInfo);
    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromBootstrap(
        static_cast<int>(nranks), clusterInfo, info_len, 0x48434c5f434c4346ULL);
    RegisterHcclDomain(*comm, static_cast<int>(nranks), static_cast<int>(comm_rank), logical_key,
                       clusterInfo, info_len, "HcclCommInitClusterInfoConfig");
    return ret;
}

//   HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms);
hcclResult_t HcclCommInitAll(unsigned int ndev, int* devices, HcclComm* comms) {
    LoadOriginalFunctions();
    AMPCCL_LOG(INFO, "HcclCommInitAll begin: ndev=%u devices=%p comms=%p", ndev,
               static_cast<void*>(devices), static_cast<void*>(comms));
    if (!orig_hcclCommInitAll) {
        AMPCCL_LOG(ERROR, "HcclCommInitAll missing original symbol");
        return HCCL_E_PARA;
    }

    AMPCCL_LOG(INFO, "HcclCommInitAll original begin");
    hcclResult_t ret = orig_hcclCommInitAll(ndev, devices, comms);
    AMPCCL_LOG(INFO, "HcclCommInitAll original end: ret=%d", static_cast<int>(ret));
    if (ret != HCCL_SUCCESS || comms == nullptr || ndev < 2) {
        return ret;
    }
    if (!IsAmpcclRuntimeEnabled()) {
        return ret;
    }

    uint64_t init_all_nonce = g_hccl_init_all_seq.fetch_add(1, std::memory_order_relaxed);
    ampccl::LogicalCommKey logical_key = ampccl::BuildLogicalKeyFromInitAll(static_cast<int>(ndev), init_all_nonce);
    for (unsigned int r = 0; r < ndev; ++r) {
        HcclComm comm = comms[r];
        if (comm == nullptr) {
            continue;
        }
        ampccl::CommDomainKey key = ampccl::BuildKeyFromHcclInitAll(static_cast<int>(ndev), static_cast<int>(r));
        key.topology_hash = logical_key.comm_hash;
        ampccl::DomainManager::GetInstance().RegisterRawComm(comm, key, logical_key);
        (void)ampccl::RegisterNodeScopeRank(logical_key, static_cast<int>(r), static_cast<int>(ndev));
    }
    for (unsigned int r = 0; r < ndev; ++r) {
        HcclComm comm = comms[r];
        if (comm == nullptr) {
            continue;
        }
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
        if (ApplyNodeScopeDecision(domain, logical_key, static_cast<int>(r), static_cast<int>(ndev), true)) {
            ampccl::InitPCIeForDomain(domain, static_cast<int>(r), static_cast<int>(ndev), nullptr, 0);
        }
    }

    return ret;
}

hcclResult_t HcclCommDestroy(HcclComm comm) {
    LoadOriginalFunctions();
    AMPCCL_LOG(INFO, "HcclCommDestroy begin: comm=%p", comm);
    if (IsAmpcclRuntimeEnabled()) {
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
        if (domain) {
            AMPCCL_LOG(INFO, "[Rank %d] HcclCommDestroy PrintAvgBw begin", domain->pcie_rank());
            domain->PrintAvgBw();
            AMPCCL_LOG(INFO, "[Rank %d] HcclCommDestroy PrintAvgBw end", domain->pcie_rank());
            AMPCCL_LOG(INFO, "[Rank %d] HcclCommDestroy DestroyPCIeForDomain begin",
                       domain->pcie_rank());
            ampccl::DestroyPCIeForDomain(domain);
            AMPCCL_LOG(INFO, "HcclCommDestroy DestroyPCIeForDomain end");
        }
        AMPCCL_LOG(INFO, "HcclCommDestroy unregister raw comm begin: comm=%p", comm);
        ampccl::DomainManager::GetInstance().UnregisterRawComm(comm);
        AMPCCL_LOG(INFO, "HcclCommDestroy unregister raw comm end: comm=%p", comm);
    }
    if (orig_hcclCommDestroy) {
        AMPCCL_LOG(INFO, "HcclCommDestroy original begin: comm=%p", comm);
        hcclResult_t ret = orig_hcclCommDestroy(comm);
        AMPCCL_LOG(INFO, "HcclCommDestroy original end: ret=%d comm=%p", static_cast<int>(ret), comm);
        return ret;
    }
    AMPCCL_LOG(ERROR, "HcclCommDestroy missing original symbol");
    return HCCL_E_PARA;
}

hcclResult_t HcclAllReduce(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_hcclAllReduce) {
        return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        if (orig_hcclAllReduce) {
            return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }
        return HCCL_E_PARA;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::AllReduce, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)), 0, 0,
            static_cast<size_t>(count))) {
        return orig_hcclAllReduce
            ? orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
            : HCCL_E_PARA;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllReduce(
        domain, sendbuff, recvbuff, count,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return BackendResultToHcclCode(result);
}

hcclResult_t HcclAllGather(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_hcclAllGather) {
        return orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        if (orig_hcclAllGather) {
            return orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        }
        return HCCL_E_PARA;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::AllGather, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)),
            static_cast<size_t>(sendcount), 0, 0)) {
        return orig_hcclAllGather
            ? orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
            : HCCL_E_PARA;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllGather(
        domain, sendbuff, recvbuff, sendcount,
        static_cast<int>(datatype), comm, stream);

    return BackendResultToHcclCode(result);
}

hcclResult_t HcclReduceScatter(
    void* sendbuff, void* recvbuff, unsigned long recvcount,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (!IsAmpcclRuntimeEnabled() && orig_hcclReduceScatter) {
        return orig_hcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        if (orig_hcclReduceScatter) {
            return orig_hcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
        }
        return HCCL_E_PARA;
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::ReduceScatter, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(datatype)), 0,
            static_cast<size_t>(recvcount), 0)) {
        return orig_hcclReduceScatter
            ? orig_hcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
            : HCCL_E_PARA;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::ReduceScatter(
        domain, static_cast<const void*>(sendbuff), recvbuff, recvcount,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return BackendResultToHcclCode(result);
}

// Canonical HCCL name and signature (see ascend-toolkit .../hccl/hccl.h).
hcclResult_t HcclAlltoAll(
    const void* sendbuff, uint64_t sendCount, HcclDataType sendType, const void* recvbuff, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream) {
    LoadOriginalFunctions();

    auto fallback = [&]() -> hcclResult_t {
        if (orig_hcclAlltoAll) {
            return orig_hcclAlltoAll(sendbuff, sendCount, sendType, recvbuff, recvCount, recvType, comm, stream);
        }
        // 8-arg original unavailable; try 6-arg (symmetric only).
        if (orig_hcclAlltoAll_6arg && sendCount == recvCount && sendType == recvType) {
            return orig_hcclAlltoAll_6arg(sendbuff, const_cast<void*>(recvbuff),
                                          sendCount, sendType, comm, stream);
        }
        return HCCL_E_PARA;
    };

    if (!IsAmpcclRuntimeEnabled()) {
        return fallback();
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain || !domain->allow_ampccl()) {
        return fallback();
    }

    // Adaptive path requires symmetric (per-peer count, same dtype); asymmetric falls back.
    if (sendCount != recvCount || sendType != recvType) {
        return fallback();
    }

    if (!ampccl::ShouldInterceptCollective(
            ampccl::CollectiveType::AllToAll, static_cast<size_t>(domain->pcie_nranks()),
            ampccl::NativeDataTypeElemSize(static_cast<int>(sendType)), 0, 0,
            static_cast<size_t>(sendCount))) {
        return fallback();
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllToAll(
        domain, sendbuff, const_cast<void*>(recvbuff), static_cast<size_t>(sendCount),
        static_cast<int>(sendType), comm, stream);

    return BackendResultToHcclCode(result);
}

// Legacy 6-argument name/order used in older samples: (send, recv, count, dtype, comm, stream) — symmetric AlltoAll.
hcclResult_t HcclAllToAll(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclComm comm, aclrtStream stream) {
    return HcclAlltoAll(sendbuff, static_cast<uint64_t>(count), datatype, recvbuff, static_cast<uint64_t>(count),
                        datatype, comm, stream);
}

}  // extern "C"

static bool IsHcclDlopenHandle(void* handle) {
    if (handle == nullptr || handle == RTLD_DEFAULT || handle == RTLD_NEXT) {
        return false;
    }
    struct link_map* map = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || map == nullptr || map->l_name == nullptr) {
        return false;
    }
    const char* name = std::strrchr(map->l_name, '/');
    name = name == nullptr ? map->l_name : name + 1;
    return std::strcmp(name, "libhccl.so") == 0 || std::strcmp(name, "libhccl.so.1") == 0;
}

static void* AmpcclHcclHookAddress(const char* symbol) {
    if (symbol == nullptr) {
        return nullptr;
    }
    if (std::strcmp(symbol, "HcclGetRootInfo") == 0) return reinterpret_cast<void*>(&HcclGetRootInfo);
    if (std::strcmp(symbol, "HcclCommInitAll") == 0) return reinterpret_cast<void*>(&HcclCommInitAll);
    if (std::strcmp(symbol, "HcclCommInitRootInfo") == 0) return reinterpret_cast<void*>(&HcclCommInitRootInfo);
    if (std::strcmp(symbol, "HcclCommInitRootInfoConfig") == 0) return reinterpret_cast<void*>(&HcclCommInitRootInfoConfig);
    if (std::strcmp(symbol, "HcclCommInitClusterInfo") == 0) return reinterpret_cast<void*>(&HcclCommInitClusterInfo);
    if (std::strcmp(symbol, "HcclCommInitClusterInfoConfig") == 0) return reinterpret_cast<void*>(&HcclCommInitClusterInfoConfig);
    if (std::strcmp(symbol, "HcclCreateSubCommConfig") == 0) return reinterpret_cast<void*>(&HcclCreateSubCommConfig);
    if (std::strcmp(symbol, "HcclCommDestroy") == 0) return reinterpret_cast<void*>(&HcclCommDestroy);
    if (std::strcmp(symbol, "HcclAllReduce") == 0) return reinterpret_cast<void*>(&HcclAllReduce);
    if (std::strcmp(symbol, "HcclAllGather") == 0) return reinterpret_cast<void*>(&HcclAllGather);
    if (std::strcmp(symbol, "HcclReduceScatter") == 0) return reinterpret_cast<void*>(&HcclReduceScatter);
    if (std::strcmp(symbol, "HcclAlltoAll") == 0) return reinterpret_cast<void*>(&HcclAlltoAll);
    if (std::strcmp(symbol, "HcclAllToAll") == 0) return reinterpret_cast<void*>(&HcclAllToAll);
    return nullptr;
}

// torch_npu's FunctionLoader calls dlsym on the libhccl.so handle directly.
// That lookup does not consult LD_PRELOAD's global symbol scope, so return our
// wrappers only for the supported HCCL API names. All other lookups remain
// owned by the real loader.
extern "C" void* dlsym(void* handle, const char* symbol) {
    static thread_local bool resolving = false;
    if (!resolving && ampccl::Config::IsAdaptiveEnabled() &&
        ampccl::Config::IsDlsymInterposeEnabled() && IsHcclDlopenHandle(handle)) {
        resolving = true;
        void* hook = AmpcclHcclHookAddress(symbol);
        resolving = false;
        if (hook != nullptr) {
            return hook;
        }
    }
    return RawDlsym(handle, symbol);
}
