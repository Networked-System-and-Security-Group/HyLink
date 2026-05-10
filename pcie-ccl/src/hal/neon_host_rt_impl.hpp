#ifndef NEON_HOST_RT_IMPL_HPP
#define NEON_HOST_RT_IMPL_HPP

#include <arm_neon.h>

#include "../../include/hal/host_rt.hpp"
#include "../../include/hal/host_tags.hpp"

namespace pccl {
namespace detail {

inline hostStatus hostMemcpyImpl(host_tag::NeonTag, void *dst, const void *src, size_t size) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (size == 0) {
    return hostStatus::SUCCESS;
  }

  uint8_t *d = static_cast<uint8_t *>(dst);
  const uint8_t *s = static_cast<const uint8_t *>(src);

  // Process 16-byte chunks with NEON
  size_t vec_count = size / 16;
  size_t vec_bytes = vec_count * 16;

  for (size_t i = 0; i < vec_count; i++) {
    uint8x16_t data = vld1q_u8(s + i * 16);
    vst1q_u8(d + i * 16, data);
  }

  // Handle remaining bytes with scalar copy
  for (size_t i = vec_bytes; i < size; i++) {
    d[i] = s[i];
  }

  return hostStatus::SUCCESS;
}

inline hostStatus hostMemcpyAddImpl(host_tag::NeonTag, float *dst, const float *src, size_t count) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (count == 0) {
    return hostStatus::SUCCESS;
  }

  size_t vec_count = count / 4;
  size_t vec_processed = vec_count * 4;

  for (size_t i = 0; i < vec_count; i++) {
    float32x4_t s = vld1q_f32(src + i * 4);
    float32x4_t d = vld1q_f32(dst + i * 4);
    float32x4_t r = vaddq_f32(d, s);
    vst1q_f32(dst + i * 4, r);
  }

  // Handle remaining elements with scalar operations
  for (size_t i = vec_processed; i < count; i++) {
    dst[i] += src[i];
  }

  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* NEON_HOST_RT_IMPL_HPP */
