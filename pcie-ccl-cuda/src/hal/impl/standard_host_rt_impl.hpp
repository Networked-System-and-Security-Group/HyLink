#ifndef PCCL_STANDARD_HOST_RT_IMPL_HPP
#define PCCL_STANDARD_HOST_RT_IMPL_HPP

#include <cstring>

#include "../host_rt.hpp"

namespace pccl {
namespace detail {

inline hostStatus hostMemcpyImpl(host_tag::StandardTag, void* dst, const void* src, size_t size) {
  std::memcpy(dst, src, size);
  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceImpl(host_tag::StandardTag, float* dst, const float* src1, const float* src2, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = src1[i] + src2[i];
  }
  return hostStatus::SUCCESS;
}

// Temporal variants: standard stores are already cache-friendly
inline hostStatus hostMemcpyTemporalImpl(host_tag::StandardTag, void* dst, const void* src, size_t size) {
  return hostMemcpyImpl(host_tag::StandardTag{}, dst, src, size);
}

inline hostStatus hostReduceTemporalImpl(host_tag::StandardTag, float* dst, const float* src1, const float* src2, size_t count) {
  return hostReduceImpl(host_tag::StandardTag{}, dst, src1, src2, count);
}

}  // namespace detail
}  // namespace pccl

#endif /* PCCL_STANDARD_HOST_RT_IMPL_HPP */
