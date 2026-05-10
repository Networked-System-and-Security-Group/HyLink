#include "comm_init.h"
#include "common/config.h"
#include "backend/memory/device_mem.h"

#ifdef AMPCCL_ENABLE_PCIE
#include "comm.hpp"
#if defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#elif defined(AMPCCL_USE_CUDA_TIMER)
#include <cuda_runtime.h>
#endif
#endif

namespace ampccl {

void InitPCIeForDomain(CommDomain* domain, int rank, int nranks) {
    AMPCCL_LOG(INFO,"[Rank %d] InitPCIeForDomain begin", rank);
    if (!domain || nranks <= 0 || rank < 0 || rank >= nranks) {
        return;
    }
#ifdef AMPCCL_ENABLE_PCIE
    // AMPCCL_LOG(INFO,"[Rank %d] InitPCIeForDomain, IsPCIeEnabled=%d", rank, Config::IsPCIeEnabled());
    domain->set_pcie_rank(rank);
    domain->set_pcie_nranks(nranks);
    if (Config::IsPCIeEnabled()) {
        pcclComm_t pcie_comm = nullptr;
        pcclResult_t ret = pcclInit(rank, nranks, &pcie_comm);
        if (ret != pcclSuccess || !pcie_comm) {
            return;
        }
        domain->set_pcie_comm(pcie_comm);

#if defined(AMPCCL_USE_ACL_TIMER)
        aclrtStream pcie_stream = nullptr;
        if (aclrtCreateStream(&pcie_stream) == ACL_SUCCESS && pcie_stream) {
            domain->set_pcie_stream(pcie_stream);
            AMPCCL_LOG(INFO,"[Rank %d] InitPCIeForDomain end: pcie_comm %p, pcie_stream %p", rank, pcie_comm, pcie_stream);
        }
#elif defined(AMPCCL_USE_CUDA_TIMER)
        cudaStream_t pcie_stream = nullptr;
        if (cudaStreamCreate(&pcie_stream) == cudaSuccess && pcie_stream) {
            domain->set_pcie_stream(pcie_stream);
            AMPCCL_LOG(INFO,"[Rank %d] InitPCIeForDomain end: pcie_comm %p, pcie_stream %p", rank, pcie_comm, pcie_stream);
        }
#endif
        size_t recvbuf_size = Config::GetMaxPCIeRecvBufBytes();
        if (recvbuf_size > 0) {
            void* recvbuf = AllocDeviceBuffer(recvbuf_size);
            if (recvbuf) {
                domain->set_pcie_recvbuf(recvbuf);
                domain->set_pcie_recvbuf_size(recvbuf_size);
                AMPCCL_LOG(INFO, "[Rank %d] InitPCIeForDomain: pcie_recvbuf %p size=%zu", rank, recvbuf, recvbuf_size);
            }
        }
    }
#endif
}

void DestroyPCIeForDomain(CommDomain* domain) {
    if (!domain) {
        return;
    }
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t pcie_comm = static_cast<pcclComm_t>(domain->pcie_comm());
    void* pcie_stream = domain->pcie_stream();
    if (pcie_comm) {
        pcclSynchronizeInternalStreams(pcie_comm);
    }
#if defined(AMPCCL_USE_ACL_TIMER)
    if (pcie_stream) {
        aclrtDestroyStream(static_cast<aclrtStream>(pcie_stream));
        domain->set_pcie_stream(nullptr);
    }
#elif defined(AMPCCL_USE_CUDA_TIMER)
    if (pcie_stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(pcie_stream));
        domain->set_pcie_stream(nullptr);
    }
#endif
    void* pcie_recvbuf = domain->pcie_recvbuf();
    if (pcie_recvbuf) {
        FreeDeviceBuffer(pcie_recvbuf);
        domain->set_pcie_recvbuf(nullptr);
        domain->set_pcie_recvbuf_size(0);
    }
    if (pcie_comm) {
        pcclDestroy(pcie_comm);
        domain->set_pcie_comm(nullptr);
    }
    domain->set_pcie_rank(-1);
    domain->set_pcie_nranks(0);
#endif
}

}  // namespace ampccl
