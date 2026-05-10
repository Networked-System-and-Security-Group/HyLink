#ifndef AVX512_HOST_RT_IMPL_HPP
#define AVX512_HOST_RT_IMPL_HPP

#include <immintrin.h>

#include <stdexcept>

#include "../../include/hal/host_rt.hpp"
#include "../../include/hal/host_tags.hpp"

namespace pccl {
namespace detail {

inline hostStatus hostMemcpyImpl(host_tag::Avx512Tag, void *dst, const void *src, size_t size) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (size == 0) {
    return hostStatus::SUCCESS;
  }

  // Alignment check
  if (reinterpret_cast<uintptr_t>(dst) % 64 != 0 || reinterpret_cast<uintptr_t>(src) % 64 != 0) {
    throw std::runtime_error("simd_memcpy: Start address is not aligned to 64 bytes.");
  }

  auto *d = static_cast<char *>(dst);
  auto *s = static_cast<const char *>(src);
  size_t offset = 0;

  // Main loop: Process 256 bytes per iteration (4 vectors)
  const size_t block_size = 64 * 4;
  if (size >= block_size) {
    size_t limit = size - block_size;
    for (; offset <= limit; offset += block_size) {
      __m512i z0 = _mm512_load_si512(reinterpret_cast<const void *>(s + offset));
      __m512i z1 = _mm512_load_si512(reinterpret_cast<const void *>(s + offset + 64));
      __m512i z2 = _mm512_load_si512(reinterpret_cast<const void *>(s + offset + 128));
      __m512i z3 = _mm512_load_si512(reinterpret_cast<const void *>(s + offset + 192));

      _mm512_stream_si512(reinterpret_cast<__m512i *>(d + offset), z0);
      _mm512_stream_si512(reinterpret_cast<__m512i *>(d + offset + 64), z1);
      _mm512_stream_si512(reinterpret_cast<__m512i *>(d + offset + 128), z2);
      _mm512_stream_si512(reinterpret_cast<__m512i *>(d + offset + 192), z3);
    }
  }

  // Handle remaining full 64-byte chunks
  for (; offset + 64 <= size; offset += 64) {
    __m512i z = _mm512_load_si512(reinterpret_cast<const void *>(s + offset));
    _mm512_stream_si512(reinterpret_cast<__m512i *>(d + offset), z);
  }

  // Handle the final tail (0-63 bytes) using masking
  size_t remaining = size - offset;
  if (remaining > 0) {
    __mmask64 mask = (1ULL << remaining) - 1;
    __m512i tail_data = _mm512_maskz_loadu_epi8(mask, s + offset);
    _mm512_mask_storeu_epi8(d + offset, mask, tail_data);
  }

  _mm_sfence();
  return hostStatus::SUCCESS;
}

inline hostStatus hostMemcpyAddImpl(host_tag::Avx512Tag, float *dst, const float *src, size_t count) {
  if (!dst || !src) {
    return hostStatus::ERROR_INVALID_VALUE;
  }
  if (count == 0) {
    return hostStatus::SUCCESS;
  }

  // Alignment check
  if (reinterpret_cast<uintptr_t>(dst) % 64 != 0 || reinterpret_cast<uintptr_t>(src) % 64 != 0) {
    throw std::runtime_error("simd_add: Start address is not aligned to 64 bytes.");
  }

  size_t idx = 0;

  // Main loop: Process 64 floats (256 bytes) per iteration
  const size_t block_count = 16 * 4;
  if (count >= block_count) {
    size_t limit = count - block_count;
    for (; idx <= limit; idx += block_count) {
      __m512 s0 = _mm512_load_ps(src + idx);
      __m512 d0 = _mm512_load_ps(dst + idx);
      __m512 s1 = _mm512_load_ps(src + idx + 16);
      __m512 d1 = _mm512_load_ps(dst + idx + 16);
      __m512 s2 = _mm512_load_ps(src + idx + 32);
      __m512 d2 = _mm512_load_ps(dst + idx + 32);
      __m512 s3 = _mm512_load_ps(src + idx + 48);
      __m512 d3 = _mm512_load_ps(dst + idx + 48);

      __m512 r0 = _mm512_add_ps(d0, s0);
      __m512 r1 = _mm512_add_ps(d1, s1);
      __m512 r2 = _mm512_add_ps(d2, s2);
      __m512 r3 = _mm512_add_ps(d3, s3);

      _mm512_store_ps(dst + idx, r0);
      _mm512_store_ps(dst + idx + 16, r1);
      _mm512_store_ps(dst + idx + 32, r2);
      _mm512_store_ps(dst + idx + 48, r3);
    }
  }

  // Handle remaining full 16-float vectors
  for (; idx + 16 <= count; idx += 16) {
    __m512 s = _mm512_load_ps(src + idx);
    __m512 d = _mm512_load_ps(dst + idx);
    _mm512_store_ps(dst + idx, _mm512_add_ps(d, s));
  }

  // Handle the tail (0-15 floats) using masking
  size_t remaining = count - idx;
  if (remaining > 0) {
    __mmask16 mask = (1U << remaining) - 1;
    __m512 s = _mm512_maskz_load_ps(mask, src + idx);
    __m512 d = _mm512_maskz_load_ps(mask, dst + idx);
    __m512 r = _mm512_add_ps(d, s);
    _mm512_mask_store_ps(dst + idx, mask, r);
  }

  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* AVX512_HOST_RT_IMPL_HPP */
