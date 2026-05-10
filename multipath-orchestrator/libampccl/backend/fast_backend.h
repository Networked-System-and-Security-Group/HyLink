#ifndef AMPCCL_BACKEND_FAST_BACKEND_H_
#define AMPCCL_BACKEND_FAST_BACKEND_H_

#include "backend_base.h"

namespace ampccl {

// Fast backend tag (NCCL/HCCL)
struct FastBackend {};

// Specialization for fast backend (NCCL/HCCL)
// This will be implemented to call actual NCCL/HCCL APIs
template<>
class BackendBase<FastBackend> {
public:
    static BackendResult AllReduce(
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* comm,
        void* stream
    );

    static BackendResult AllGather(
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* comm,
        void* stream
    );

    static BackendResult ReduceScatter(
        const void* sendbuff,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* comm,
        void* stream
    );

    static BackendResult Broadcast(
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int root,
        void* comm,
        void* stream
    );
};

// Type alias for convenience
using FastBackendImpl = BackendBase<FastBackend>;

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_FAST_BACKEND_H_
