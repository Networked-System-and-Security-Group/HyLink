#include "pcie_backend.h"
#include "core/domain.h"
#include "backend/pcie_kernels/allreduce_pcie.h"
#include "backend/pcie_kernels/allgather_pcie.h"

namespace ampccl {

BackendResult BackendBase<PCIeBackend>::AllReduce(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int op,
    void* stream) {
    return pcie_kernels::RunAllReducePcie(domain, sendbuff, recvbuff, count, datatype, op, stream);
}

BackendResult BackendBase<PCIeBackend>::AllGather(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* stream) {
    return pcie_kernels::RunAllGatherPcie(domain, sendbuff, recvbuff, sendcount, datatype, stream);
}

BackendResult BackendBase<PCIeBackend>::ReduceScatter(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    void* stream) {
    // (void)domain;
    // (void)sendbuff;
    // (void)recvbuff;
    // (void)recvcount;
    // (void)datatype;
    // (void)op;
    // (void)stream;
    return BackendResult::Success;
}

BackendResult BackendBase<PCIeBackend>::Broadcast(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int root,
    void* stream) {
    // (void)domain;
    // (void)sendbuff;
    // (void)recvbuff;
    // (void)count;
    // (void)datatype;
    // (void)root;
    // (void)stream;
    return BackendResult::Success;
}

}  // namespace ampccl
