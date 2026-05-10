#ifndef AMPCCL_BACKEND_BACKEND_BASE_H_
#define AMPCCL_BACKEND_BACKEND_BASE_H_

#include <cstddef>
#include <cstdint>

namespace ampccl {

// Result type matching NCCL/HCCL conventions
enum class BackendResult {
    Success = 0,
    InvalidArgument = 1,
    UnhandledError = 2,
    InternalError = 3
};

// Base interface for all backends
template<typename BackendType>
class BackendBase {
public:
    // AllReduce operation
    static BackendResult AllReduce(
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,  // reduce op (sum, max, min, etc.)
        void* comm,  // communicator handle
        void* stream  // CUDA stream
    );

    // AllGather operation
    static BackendResult AllGather(
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* comm,
        void* stream
    );

    // ReduceScatter operation
    static BackendResult ReduceScatter(
        const void* sendbuff,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* comm,
        void* stream
    );

    // Broadcast operation
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

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_BACKEND_BASE_H_
