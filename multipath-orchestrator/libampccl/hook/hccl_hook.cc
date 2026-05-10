// HCCL hook for LD_PRELOAD
// This file intercepts HCCL calls and routes them through AMP-CCL

#include "core/virtual_collective.h"
#include "core/domain_manager.h"
#include "core/comm_init.h"
#include "core/stream_sync.h"
#include "backend/original_collective_api.h"
#include "common/op_key.h"
#include "common/config.h"
#include <dlfcn.h>
#include <cstring>

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
#define HCCL_UNIQUE_ID_BYTES 128
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
    HCCL_DATA_TYPE_FLOAT = 0,
    HCCL_DATA_TYPE_FLOAT16,
    HCCL_DATA_TYPE_INT32
} HcclDataType;

typedef enum {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_MAX,
    HCCL_REDUCE_MIN
} HcclReduceOp;

typedef void* HcclComm;
typedef void* aclrtStream;

typedef struct { char internal[HCCL_UNIQUE_ID_BYTES]; } hcclUniqueId;
typedef struct { char internal[HCCL_ROOT_INFO_BYTES]; } HcclRootInfo;

// Forward declarations for original HCCL functions
typedef hcclResult_t (*hcclGetUniqueId_t)(hcclUniqueId* uniqueId);
typedef hcclResult_t (*hcclCommInitRank_t)(HcclComm* comm, unsigned int nranks, hcclUniqueId commId, unsigned int rank);
typedef hcclResult_t (*hcclCommInitRootInfo_t)(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
typedef hcclResult_t (*hcclCommInitAll_t)(unsigned int ndev, int* devices, HcclComm* comms);
typedef hcclResult_t (*hcclCommDestroy_t)(HcclComm comm);

typedef hcclResult_t (*hcclAllReduce_t)(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclAllGather_t)(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclReduceScatter_t)(
    const void* sendbuff, void* recvbuff, unsigned long recvcount,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream);

typedef hcclResult_t (*hcclBroadcast_t)(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, unsigned int root, HcclComm comm, aclrtStream stream);

// ACL runtime (for stream sync)
typedef int (*aclrtSynchronizeStream_t)(aclrtStream stream);

// Function pointers to original HCCL functions
static hcclGetUniqueId_t orig_hcclGetUniqueId = nullptr;
static hcclCommInitRank_t orig_hcclCommInitRank = nullptr;
static hcclCommInitRootInfo_t orig_hcclCommInitRootInfo = nullptr;
static hcclCommInitAll_t orig_hcclCommInitAll = nullptr;
static hcclCommDestroy_t orig_hcclCommDestroy = nullptr;
static hcclAllReduce_t orig_hcclAllReduce = nullptr;
static hcclAllGather_t orig_hcclAllGather = nullptr;
static hcclReduceScatter_t orig_hcclReduceScatter = nullptr;
static hcclBroadcast_t orig_hcclBroadcast = nullptr;
static aclrtSynchronizeStream_t orig_aclrtSynchronizeStream = nullptr;

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
        orig_hcclGetUniqueId = (hcclGetUniqueId_t)dlsym(handle, "HcclGetUniqueId");
        orig_hcclCommInitRank = (hcclCommInitRank_t)dlsym(handle, "HcclCommInitRank");
        orig_hcclCommInitAll = (hcclCommInitAll_t)dlsym(handle, "HcclCommInitAll");
        orig_hcclCommInitRootInfo = (hcclCommInitRootInfo_t)dlsym(handle, "HcclCommInitRootInfo");
        orig_hcclCommDestroy = (hcclCommDestroy_t)dlsym(handle, "HcclCommDestroy");
        orig_hcclAllReduce = (hcclAllReduce_t)dlsym(handle, "HcclAllReduce");
        orig_hcclAllGather = (hcclAllGather_t)dlsym(handle, "HcclAllGather");
        orig_hcclReduceScatter = (hcclReduceScatter_t)dlsym(handle, "HcclReduceScatter");
        orig_hcclBroadcast = (hcclBroadcast_t)dlsym(handle, "HcclBroadcast");
    }

    // ACL runtime for aclrtSynchronizeStream (may be in same lib or libascendcl/libacl)
    if (!orig_aclrtSynchronizeStream) {
        void* acl_handle = dlopen("libascendcl.so", RTLD_LAZY);
        if (!acl_handle) {
            acl_handle = dlopen("libacl.so", RTLD_LAZY);
        }
        if (acl_handle) {
            AMPCCL_LOG(INFO,"find libascendcl.so, load aclrt api success!");
            orig_aclrtSynchronizeStream = (aclrtSynchronizeStream_t)dlsym(acl_handle, "aclrtSynchronizeStream");
        }
    }

    if (orig_hcclAllReduce || orig_hcclAllGather) {
        ampccl::OriginalCollectiveAPI::RegisterHccl(
            reinterpret_cast<void*>(orig_hcclAllReduce),
            reinterpret_cast<void*>(orig_hcclAllGather),
            reinterpret_cast<void*>(orig_hcclReduceScatter),
            reinterpret_cast<void*>(orig_hcclBroadcast));
    }
    AMPCCL_LOG(INFO, "register hccl comm api success!\n"
        "allgather, allreduce, broadcast, reducescatter.\n"
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
    return ampccl::BackendResult::UnhandledError;
}

// Look up domain by raw HCCL communicator (registered at CommInit).
static ampccl::CommDomain* GetDomainByRawComm(HcclComm comm) {
    return ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
}

void ampccl::SyncDeviceStreamRaw(void* stream) {
    if (orig_aclrtSynchronizeStream && stream) {
        (void)orig_aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
    }
}

// Hooked HCCL functions
extern "C" {

hcclResult_t HcclGetUniqueId(hcclUniqueId* uniqueId) {
    LoadOriginalFunctions();
    if (orig_hcclGetUniqueId) {
        return orig_hcclGetUniqueId(uniqueId);
    }
    return HCCL_E_PARA;
}

hcclResult_t HcclCommInitRank(HcclComm* comm, unsigned int nranks, hcclUniqueId commId, unsigned int rank) {
    LoadOriginalFunctions();
    if (!orig_hcclCommInitRank) {
        return HCCL_E_PARA;
    }
    hcclResult_t ret = orig_hcclCommInitRank(comm, nranks, commId, rank);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    if (!ampccl::Config::IsAdaptiveEnabled()) {
        return ret;
    }
    ampccl::CommDomainKey key = ampccl::BuildKeyFromHcclInit(
        static_cast<int>(nranks), &commId, HCCL_UNIQUE_ID_BYTES, static_cast<int>(rank));
    ampccl::DomainManager::GetInstance().RegisterRawComm(*comm, key);
    ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(*comm);
    if (domain) {
        ampccl::InitPCIeForDomain(domain, static_cast<int>(rank), static_cast<int>(nranks));
    }
    return ret;
}

hcclResult_t HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm) {
    LoadOriginalFunctions();
    if (!orig_hcclCommInitRootInfo) {
        return HCCL_E_PARA;
    }
    hcclResult_t ret = orig_hcclCommInitRootInfo(nRanks, rootInfo, rank, comm);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }
    // AMPCCL_LOG(INFO,"[Rank %d] Exec original HcclCommInitRootInfo finished", rank);
    if (!ampccl::Config::IsAdaptiveEnabled()) {
        return ret;
    }
    ampccl::CommDomainKey key = ampccl::BuildKeyFromHcclInitRootInfo(
        static_cast<int>(nRanks), rootInfo, sizeof(HcclRootInfo), static_cast<int>(rank));
    ampccl::DomainManager::GetInstance().RegisterRawComm(*comm, key);
    ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(*comm);
    if (domain) {
        // printf("domain: %p on rank %d\n", domain, rank);
        ampccl::InitPCIeForDomain(domain, rank, nRanks);
        // printf("domain is %p, rank %d, ranks %d\n", domain, domain->pcie_rank(), domain->pcie_nranks());
        AMPCCL_LOG(INFO,"[Rank %d] InitPCIeForDomain finished", rank);
    }
    AMPCCL_LOG(INFO,"[Rank %d] AMP-CCL HcclCommInitRootInfo finished", rank);
    return ret;
}

//   HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms);
hcclResult_t HcclCommInitAll(unsigned int ndev, int* devices, HcclComm* comms) {
    LoadOriginalFunctions();
    if (!orig_hcclCommInitAll) {
        return HCCL_E_PARA;
    }

    hcclResult_t ret = orig_hcclCommInitAll(ndev, devices, comms);
    if (ret != HCCL_SUCCESS || comms == nullptr || ndev == 0) {
        return ret;
    }
    if (!ampccl::Config::IsAdaptiveEnabled()) {
        return ret;
    }

    for (unsigned int r = 0; r < ndev; ++r) {
        HcclComm comm = comms[r];
        if (comm == nullptr) {
            continue;
        }
        ampccl::CommDomainKey key = ampccl::BuildKeyFromHcclInitAll(static_cast<int>(ndev), static_cast<int>(r));
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetOrCreateDomainByKey(key);
        if (domain == nullptr) {
            continue;
        }
        ampccl::DomainManager::GetInstance().RegisterRawComm(comm, key);
        ampccl::InitPCIeForDomain(domain, static_cast<int>(r), static_cast<int>(ndev));
    }

    return ret;
}

hcclResult_t HcclCommDestroy(HcclComm comm) {
    LoadOriginalFunctions();
    if (ampccl::Config::IsAdaptiveEnabled()) {
        ampccl::DomainManager::GetInstance().UnregisterRawComm(comm);
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
        if (domain) {
            ampccl::DestroyPCIeForDomain(domain);
        }
    }
    if (orig_hcclCommDestroy) {
        return orig_hcclCommDestroy(comm);
    }
    return HCCL_E_PARA;
}

hcclResult_t HcclAllReduce(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (!ampccl::Config::IsAdaptiveEnabled() && orig_hcclAllReduce) {
        return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain) {
        if (orig_hcclAllReduce) {
            return orig_hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }
        return HCCL_E_PARA;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllReduce(
        domain, sendbuff, recvbuff, count,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return (result == ampccl::BackendResult::Success) ? HCCL_SUCCESS : HCCL_E_PARA;
}

hcclResult_t HcclAllGather(
    const void* sendbuff, void* recvbuff, unsigned long sendcount,
    HcclDataType datatype, HcclComm comm, aclrtStream stream) {
    LoadOriginalFunctions();
    if (!ampccl::Config::IsAdaptiveEnabled() && orig_hcclAllGather) {
        return orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain) {
        if (orig_hcclAllGather) {
            return orig_hcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        }
        return HCCL_E_PARA;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllGather(
        domain, sendbuff, recvbuff, sendcount,
        static_cast<int>(datatype), comm, stream);

    return (result == ampccl::BackendResult::Success) ? HCCL_SUCCESS : HCCL_E_PARA;
}

hcclResult_t HcclReduceScatter(
    const void* sendbuff, void* recvbuff, unsigned long recvcount,
    HcclDataType datatype, HcclReduceOp op, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (orig_hcclReduceScatter) {
        return orig_hcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    }
    return HCCL_E_PARA;
}

hcclResult_t HcclBroadcast(
    const void* sendbuff, void* recvbuff, unsigned long count,
    HcclDataType datatype, unsigned int root, HcclComm comm, aclrtStream stream) {

    LoadOriginalFunctions();
    if (orig_hcclBroadcast) {
        return orig_hcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    }
    return HCCL_E_PARA;
}

int aclrtSynchronizeStream(aclrtStream stream) {
    LoadOriginalFunctions();
    if (orig_aclrtSynchronizeStream) {
        // AMPCCL_LOG(INFO,"invoke aclrtSynchronizeStream to sync fast ccl stream");
        int ret = orig_aclrtSynchronizeStream(stream);
        if (ret == 0 && ampccl::Config::IsAdaptiveEnabled()) {
            // AMPCCL_LOG(INFO,"invoke OnStreamSynchronized to sync pcie ccl stream");
            ampccl::OnStreamSynchronized(stream);
        }
        return ret;
    }
    return -1;
}


}  // extern "C"
