// NCCL hook for LD_PRELOAD
// This file intercepts NCCL calls and routes them through AMP-CCL

#include "core/virtual_collective.h"
#include "core/domain_manager.h"
#include "core/comm_init.h"
#include "core/stream_sync.h"
#include "backend/original_collective_api.h"
#include "common/op_key.h"
#include "common/config.h"
#include "common/log.h"
#include <dlfcn.h>
#include <cstring>

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
typedef int (*ncclCommInitAll_t)(ncclComm_t* comms, int ndev, int* devlist);
typedef int (*ncclCommDestroy_t)(ncclComm_t comm);

typedef int (*ncclAllReduce_t)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclAllGather_t)(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclReduceScatter_t)(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

typedef int (*ncclBroadcast_t)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

// CUDA runtime (for stream sync)
typedef ampccl_cudaError_t (*cudaStreamSynchronize_t)(ampccl_cudaStream_t stream);

// Function pointers to original NCCL functions
static ncclGetUniqueId_t orig_ncclGetUniqueId = nullptr;
static ncclCommInitRank_t orig_ncclCommInitRank = nullptr;
static ncclCommInitAll_t orig_ncclCommInitAll = nullptr;
static ncclCommDestroy_t orig_ncclCommDestroy = nullptr;
static ncclAllReduce_t orig_ncclAllReduce = nullptr;
static ncclAllGather_t orig_ncclAllGather = nullptr;
static ncclReduceScatter_t orig_ncclReduceScatter = nullptr;
static ncclBroadcast_t orig_ncclBroadcast = nullptr;
static cudaStreamSynchronize_t orig_cudaStreamSynchronize = nullptr;

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
        orig_ncclCommInitAll = (ncclCommInitAll_t)dlsym(handle, "ncclCommInitAll");
        orig_ncclCommDestroy = (ncclCommDestroy_t)dlsym(handle, "ncclCommDestroy");
        orig_ncclAllReduce = (ncclAllReduce_t)dlsym(handle, "ncclAllReduce");
        orig_ncclAllGather = (ncclAllGather_t)dlsym(handle, "ncclAllGather");
        orig_ncclReduceScatter = (ncclReduceScatter_t)dlsym(handle, "ncclReduceScatter");
        orig_ncclBroadcast = (ncclBroadcast_t)dlsym(handle, "ncclBroadcast");
    }

    // CUDA runtime for cudaStreamSynchronize
    if (!orig_cudaStreamSynchronize) {
        void* cuda_handle = dlopen("libcudart.so", RTLD_LAZY);
        if (cuda_handle) {
            orig_cudaStreamSynchronize = (cudaStreamSynchronize_t)dlsym(cuda_handle, "cudaStreamSynchronize");
            AMPCCL_LOG(INFO, "find libcudart.so, load cudaStreamSynchronize success!");
        }
    }

    if (orig_ncclAllReduce || orig_ncclAllGather) {
        ampccl::OriginalCollectiveAPI::RegisterNccl(
            reinterpret_cast<void*>(orig_ncclAllReduce),
            reinterpret_cast<void*>(orig_ncclAllGather),
            reinterpret_cast<void*>(orig_ncclReduceScatter),
            reinterpret_cast<void*>(orig_ncclBroadcast));
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
    return ampccl::BackendResult::UnhandledError;
}

// Look up domain by raw NCCL communicator (registered at CommInit).
static ampccl::CommDomain* GetDomainByRawComm(ncclComm_t comm) {
    return ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
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
    if (!ampccl::Config::IsAdaptiveEnabled()) {
        return ret;
    }
    printf("rank %d: register nccl comm\n", myrank);
    ampccl::CommDomainKey key = ampccl::BuildKeyFromNcclInit(
        nranks, &commId, NCCL_UNIQUE_ID_BYTES, myrank);
    ampccl::DomainManager::GetInstance().RegisterRawComm(*comm, key);
    ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(*comm);
    if (domain) {
        ampccl::InitPCIeForDomain(domain, myrank, nranks);
    }
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
    if (!ampccl::Config::IsAdaptiveEnabled()) {
        return ret;
    }
    for (int r = 0; r < ndev; ++r) {
        ncclComm_t comm = comms[r];
        if (comm == nullptr) {
            continue;
        }
        ampccl::CommDomainKey key = ampccl::BuildKeyFromNcclInitAll(ndev, r);
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetOrCreateDomainByKey(key);
        if (domain == nullptr) {
            continue;
        }
        ampccl::DomainManager::GetInstance().RegisterRawComm(comm, key);
        ampccl::InitPCIeForDomain(domain, r, ndev);
    }
    return ret;
}

int ncclCommDestroy(ncclComm_t comm) {
    LoadOriginalFunctions();
    if (ampccl::Config::IsAdaptiveEnabled()) {
        ampccl::CommDomain* domain = ampccl::DomainManager::GetInstance().GetDomainByRawComm(comm);
        ampccl::DomainManager::GetInstance().UnregisterRawComm(comm);
        if (domain) {
            ampccl::DestroyPCIeForDomain(domain);
        }
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
    if (!ampccl::Config::IsAdaptiveEnabled() && orig_ncclAllReduce) {
        return orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain) {
        if (orig_ncclAllReduce) {
            return orig_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }
        return -1;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllReduce(
        domain, sendbuff, recvbuff, count,
        static_cast<int>(datatype), static_cast<int>(op), comm, stream);

    return (result == ampccl::BackendResult::Success) ? 0 : -1;
}

int ncclAllGather(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (!ampccl::Config::IsAdaptiveEnabled() && orig_ncclAllGather) {
        return orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }

    ampccl::CommDomain* domain = GetDomainByRawComm(comm);
    if (!domain) {
        if (orig_ncclAllGather) {
            return orig_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        }
        return -1;
    }

    ampccl::BackendResult result = ampccl::VirtualCollective::AllGather(
        domain, sendbuff, recvbuff, sendcount,
        static_cast<int>(datatype), comm, stream);

    return (result == ampccl::BackendResult::Success) ? 0 : -1;
}

int ncclReduceScatter(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {

    LoadOriginalFunctions();
    if (orig_ncclReduceScatter) {
        return orig_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    }
    return -1;
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

ampccl_cudaError_t cudaStreamSynchronize(ampccl_cudaStream_t stream) {
    LoadOriginalFunctions();
    if (orig_cudaStreamSynchronize) {
        ampccl_cudaError_t ret = orig_cudaStreamSynchronize(stream);
        if (ret == 0 && ampccl::Config::IsAdaptiveEnabled()) {
            ampccl::OnStreamSynchronized(stream);
        }
        return ret;
    }
    return static_cast<ampccl_cudaError_t>(-1);  // cudaErrorUnknown or similar
}

}  // extern "C"
