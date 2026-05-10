#ifndef AMPCCL_COMMON_INTERCEPT_POLICY_H_
#define AMPCCL_COMMON_INTERCEPT_POLICY_H_

#include "common/config.h"
#include "common/op_key.h"

#include <cstddef>
#include <limits>

namespace ampccl {

namespace {

constexpr size_t kEightRankInterceptMinBytes = 256ULL * 1024ULL * 1024ULL;

inline bool MultiplySizeT(size_t lhs, size_t rhs, size_t* result) {
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

inline size_t HookInterceptMinBytes(CollectiveType op, size_t nranks) {
    // Keep this rank/op dispatch explicit: the current policy has one 8-rank
    // threshold for all standard collectives. The individual cases are the
    // extension points for future per-op thresholds.
    if (nranks == 8) {
        switch (op) {
            case CollectiveType::AllReduce:
            case CollectiveType::AllGather:
            case CollectiveType::ReduceScatter:
            case CollectiveType::AllToAll:
                return kEightRankInterceptMinBytes;
        }
    }
    return Config::GetMinMsgSize();
}

inline bool LargestCollectiveBufferBytes(CollectiveType op, size_t nranks,
                                         size_t dtype_size, size_t sendcount,
                                         size_t recvcount, size_t count,
                                         size_t* bytes) {
    if (bytes == nullptr || nranks == 0 || dtype_size == 0) {
        return false;
    }

    size_t elements = 0;
    switch (op) {
        case CollectiveType::AllReduce:
            elements = count;
            break;
        case CollectiveType::AllGather:
            if (!MultiplySizeT(sendcount, nranks, &elements)) {
                return false;
            }
            break;
        case CollectiveType::ReduceScatter:
            if (!MultiplySizeT(recvcount, nranks, &elements)) {
                return false;
            }
            break;
        case CollectiveType::AllToAll:
            if (!MultiplySizeT(count, nranks, &elements)) {
                return false;
            }
            break;
    }

    return MultiplySizeT(elements, dtype_size, bytes);
}

}  // namespace

// Return true only when the hook should enter AMP-CCL. The byte count is the
// largest user-visible buffer, not the per-peer payload used by the planner.
inline bool ShouldInterceptCollective(CollectiveType op, size_t nranks,
                                      size_t dtype_size, size_t sendcount,
                                      size_t recvcount, size_t count) {
    size_t bytes = 0;
    if (!LargestCollectiveBufferBytes(op, nranks, dtype_size, sendcount, recvcount,
                                      count, &bytes)) {
        return false;
    }
    return bytes >= HookInterceptMinBytes(op, nranks);
}

}  // namespace ampccl

#endif  // AMPCCL_COMMON_INTERCEPT_POLICY_H_
