#ifndef AMPCCL_BACKEND_PCIE_KERNELS_ALLGATHER_PCIE_H_
#define AMPCCL_BACKEND_PCIE_KERNELS_ALLGATHER_PCIE_H_

#include "backend/backend_base.h"
#include <cstddef>

namespace ampccl {

class CommDomain;

namespace pcie_kernels {

BackendResult RunAllGatherPcie(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* stream);

}  // namespace pcie_kernels

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_PCIE_KERNELS_ALLGATHER_PCIE_H_

