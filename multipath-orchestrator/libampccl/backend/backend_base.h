#ifndef AMPCCL_BACKEND_BACKEND_BASE_H_
#define AMPCCL_BACKEND_BACKEND_BASE_H_

#include <cstddef>
#include <cstdint>

namespace ampccl {

// Result type matching NCCL/HCCL conventions.
// Carries the original native error code (HCCL/NCCL/PCCL) for diagnostics.
struct BackendResult {
    enum Code : int {
        kSuccess = 0,
        kInvalidArgument = 1,
        kUnhandledError = 2,
        kInternalError = 3
    };

    Code code = kSuccess;
    int original = 0;  // raw HCCL/NCCL/PCCL return value (0 when no native call was made)

    // Shorthand constants so existing `BackendResult::Success` comparisons keep compiling.
    static constexpr Code Success = kSuccess;
    static constexpr Code UnhandledError = kUnhandledError;
    static constexpr Code InvalidArgument = kInvalidArgument;
    static constexpr Code InternalError = kInternalError;

    BackendResult() = default;
    /*implicit*/ BackendResult(Code c) : code(c), original(0) {}
    BackendResult(Code c, int orig) : code(c), original(orig) {}

    bool operator==(Code c) const { return code == c; }
    bool operator!=(Code c) const { return code != c; }
    bool IsOk() const { return code == kSuccess; }
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
