#ifndef AMPCCL_BACKEND_PCIE_BACKEND_H_
#define AMPCCL_BACKEND_PCIE_BACKEND_H_

#include "backend_base.h"

namespace ampccl {

class CommDomain;

// PCIe backend tag
struct PCIeBackend {};

// PCIe backend uses CommDomain (pcie_comm, pcie_rank, pcie_nranks); raw comm not used.
template<>
class BackendBase<PCIeBackend> {
public:
    static BackendResult AllReduce(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* stream
    );

    static BackendResult AllGather(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* stream
    );

    static BackendResult ReduceScatter(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* stream
    );

    static BackendResult Broadcast(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int root,
        void* stream
    );
};

using PCIeBackendImpl = BackendBase<PCIeBackend>;

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_PCIE_BACKEND_H_
