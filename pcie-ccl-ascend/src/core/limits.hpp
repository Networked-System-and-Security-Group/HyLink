#ifndef PCCL_LIMITS_HPP
#define PCCL_LIMITS_HPP

#include <cstddef>

namespace pccl {

constexpr int PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT = 512;
constexpr int PCCL_EXECUTOR_QUEUE_DEPTH_MAX = 512;
constexpr int PCCL_MAX_CONCURRENT_COMMS = 32;
constexpr std::size_t PCCL_INTERNAL_BUFFER_BYTES = 4ULL << 30;

}  // namespace pccl

#endif /* PCCL_LIMITS_HPP */
