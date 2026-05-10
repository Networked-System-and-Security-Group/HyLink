#ifndef AMPCCL_BACKEND_PCIE_BACKEND_H_
#define AMPCCL_BACKEND_PCIE_BACKEND_H_

#include "backend_base.h"

namespace ampccl {

class CommDomain;

struct PCIeBackend {};

template<>
class BackendBase<PCIeBackend> {
public:
    static bool SupportsAllReduce(int datatype, int op);
    static bool SupportsAllGather(int datatype);
    static bool SupportsReduceScatter(int datatype, int op);
    static bool SupportsAllToAll(int datatype);

    static BackendResult AllReduce(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* stream);

    static BackendResult AllGather(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* stream);

    static BackendResult AllGatherStrided(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        size_t stride_elems,
        int datatype,
        void* stream);

    static BackendResult ReduceScatter(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* stream);

    static BackendResult ReduceScatterStrided(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t total_elems,
        size_t stride_elems,
        int datatype,
        int op,
        void* stream);

    static BackendResult AllToAll(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        void* stream);

    static BackendResult AllToAllStrided(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t total_elems,
        size_t stride_elems,
        int datatype,
        void* stream);

    static BackendResult Broadcast(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int root,
        void* stream);
};

using PCIeBackendImpl = BackendBase<PCIeBackend>;

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_PCIE_BACKEND_H_
