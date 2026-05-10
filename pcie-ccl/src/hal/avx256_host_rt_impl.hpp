#ifndef AVX256_HOST_RT_IMPL_HPP
#define AVX256_HOST_RT_IMPL_HPP

#include <immintrin.h>

#include "../../include/hal/host_rt.hpp"
#include "../../include/hal/host_tags.hpp"

namespace pccl {
namespace detail {

inline hostStatus hostMemcpyImpl(host_tag::Avx256Tag, void *dst, const void *src, size_t size) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (size == 0) {
    return hostStatus::SUCCESS;
  }

  char *d = static_cast<char *>(dst);
  const char *s = static_cast<const char *>(src);

  // Process 32-byte chunks with AVX2
  size_t vec_count = size / 32;
  size_t vec_bytes = vec_count * 32;

  for (size_t i = 0; i < vec_count; i++) {
    __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i * 32));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i * 32), data);
  }

  // Handle remaining bytes with scalar copy
  for (size_t i = vec_bytes; i < size; i++) {
    d[i] = s[i];
  }

  return hostStatus::SUCCESS;
}

inline hostStatus hostMemcpyAddImpl(host_tag::Avx256Tag, float *dst, const float *src, size_t count) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (count == 0) {
    return hostStatus::SUCCESS;
  }

  bool aligned = (reinterpret_cast<uintptr_t>(dst) % 32 == 0) && (reinterpret_cast<uintptr_t>(src) % 32 == 0);

  size_t vec_count = count / 8;
  size_t vec_processed = vec_count * 8;

  if (aligned) {
    for (size_t i = 0; i < vec_count; i++) {
      __m256 s = _mm256_load_ps(src + i * 8);
      __m256 d = _mm256_load_ps(dst + i * 8);
      __m256 r = _mm256_add_ps(d, s);
      _mm256_store_ps(dst + i * 8, r);
    }
  } else {
    for (size_t i = 0; i < vec_count; i++) {
      __m256 s = _mm256_loadu_ps(src + i * 8);
      __m256 d = _mm256_loadu_ps(dst + i * 8);
      __m256 r = _mm256_add_ps(d, s);
      _mm256_storeu_ps(dst + i * 8, r);
    }
  }

  // Handle remaining elements with scalar operations
  for (size_t i = vec_processed; i < count; i++) {
    dst[i] += src[i];
  }

  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* AVX256_HOST_RT_IMPL_HPP */
