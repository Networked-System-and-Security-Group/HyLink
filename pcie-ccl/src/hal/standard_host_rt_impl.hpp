#ifndef STANDARD_HOST_RT_IMPL_HPP
#define STANDARD_HOST_RT_IMPL_HPP

#include <cstring>

#include "../../include/hal/host_rt.hpp"
#include "../../include/hal/host_tags.hpp"

namespace pccl {
namespace detail {

inline hostStatus hostMemcpyImpl(host_tag::StandardTag, void *dst, const void *src, size_t size) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (size == 0) {
    return hostStatus::SUCCESS;
  }

  std::memcpy(dst, src, size);
  return hostStatus::SUCCESS;
}

inline hostStatus hostMemcpyAddImpl(host_tag::StandardTag, float *dst, const float *src, size_t count) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (count == 0) {
    return hostStatus::SUCCESS;
  }

  for (size_t i = 0; i < count; i++) {
    dst[i] += src[i];
  }
  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* STANDARD_HOST_RT_IMPL_HPP */
