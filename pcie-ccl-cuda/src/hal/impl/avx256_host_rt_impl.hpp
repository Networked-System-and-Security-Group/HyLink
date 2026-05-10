#ifndef PCCL_AVX256_HOST_RT_IMPL_HPP
#define PCCL_AVX256_HOST_RT_IMPL_HPP

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "../host_rt.hpp"

namespace pccl {
namespace detail {

// ── Non-temporal (default): bypass L3, write directly to DRAM ─────────────
// Matches NEON's STNP behavior.

inline hostStatus hostMemcpyImpl(host_tag::Avx256Tag, void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return hostStatus::SUCCESS;
  }

  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t*       d = static_cast<uint8_t*>(dst);
  size_t i = 0;

  // Align dst to 32-byte boundary (required by _mm256_stream_ps)
  size_t align_offset = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
  if (align_offset > 0) {
    size_t head = (align_offset < size) ? align_offset : size;
    std::memcpy(d, s, head);
    i = head;
  }

  // Main loop: 256 bytes (8 × 32B) per iteration, NT stores.
  // Prefetch 2KB ahead on src.
  for (; i + 256 <= size; i += 256) {
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 192), _MM_HINT_NTA);

    __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i));
    __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 32));
    __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 64));
    __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 96));
    __m256 v4 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 128));
    __m256 v5 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 160));
    __m256 v6 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 192));
    __m256 v7 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 224));

    _mm256_stream_ps(reinterpret_cast<float*>(d + i),       v0);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 32),  v1);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 64),  v2);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 96),  v3);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 128), v4);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 160), v5);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 192), v6);
    _mm256_stream_ps(reinterpret_cast<float*>(d + i + 224), v7);
  }

  if (i < size)
    std::memcpy(d + i, s + i, size - i);

  _mm_sfence();
  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceImpl(host_tag::Avx256Tag, float* dst, const float* src1, const float* src2, size_t count) {
  size_t i = 0;

  // Align dst to 32-byte boundary (8 floats) for stream stores
  size_t align_floats = (8 - ((reinterpret_cast<uintptr_t>(dst) >> 2) & 7)) & 7;
  if (align_floats > 0) {
    size_t head = (align_floats < count) ? align_floats : count;
    for (size_t j = 0; j < head; ++j)
      dst[j] = src1[j] + src2[j];
    i = head;
  }

  // Main loop: 8× unrolled, 64 floats (256 bytes) per iteration.
  const uint8_t* s1 = reinterpret_cast<const uint8_t*>(src1);
  const uint8_t* s2 = reinterpret_cast<const uint8_t*>(src2);

  for (; i + 64 <= count; i += 64) {
    size_t bi = i * sizeof(float);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048 + 192), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048 + 192), _MM_HINT_NTA);

    __m256 a0 = _mm256_loadu_ps(src1 + i);
    __m256 b0 = _mm256_loadu_ps(src2 + i);
    __m256 a1 = _mm256_loadu_ps(src1 + i + 8);
    __m256 b1 = _mm256_loadu_ps(src2 + i + 8);
    __m256 a2 = _mm256_loadu_ps(src1 + i + 16);
    __m256 b2 = _mm256_loadu_ps(src2 + i + 16);
    __m256 a3 = _mm256_loadu_ps(src1 + i + 24);
    __m256 b3 = _mm256_loadu_ps(src2 + i + 24);
    __m256 a4 = _mm256_loadu_ps(src1 + i + 32);
    __m256 b4 = _mm256_loadu_ps(src2 + i + 32);
    __m256 a5 = _mm256_loadu_ps(src1 + i + 40);
    __m256 b5 = _mm256_loadu_ps(src2 + i + 40);
    __m256 a6 = _mm256_loadu_ps(src1 + i + 48);
    __m256 b6 = _mm256_loadu_ps(src2 + i + 48);
    __m256 a7 = _mm256_loadu_ps(src1 + i + 56);
    __m256 b7 = _mm256_loadu_ps(src2 + i + 56);

    _mm256_stream_ps(dst + i,      _mm256_add_ps(a0, b0));
    _mm256_stream_ps(dst + i + 8,  _mm256_add_ps(a1, b1));
    _mm256_stream_ps(dst + i + 16, _mm256_add_ps(a2, b2));
    _mm256_stream_ps(dst + i + 24, _mm256_add_ps(a3, b3));
    _mm256_stream_ps(dst + i + 32, _mm256_add_ps(a4, b4));
    _mm256_stream_ps(dst + i + 40, _mm256_add_ps(a5, b5));
    _mm256_stream_ps(dst + i + 48, _mm256_add_ps(a6, b6));
    _mm256_stream_ps(dst + i + 56, _mm256_add_ps(a7, b7));
  }

  for (; i + 8 <= count; i += 8) {
    __m256 a = _mm256_loadu_ps(src1 + i);
    __m256 b = _mm256_loadu_ps(src2 + i);
    _mm256_stream_ps(dst + i, _mm256_add_ps(a, b));
  }
  for (; i < count; ++i)
    dst[i] = src1[i] + src2[i];

  _mm_sfence();
  return hostStatus::SUCCESS;
}

// ── Temporal: data stays in L3 for downstream CPU consumers ───────────────
// Matches NEON's STP/vst1q_f32 behavior.

inline hostStatus hostMemcpyTemporalImpl(host_tag::Avx256Tag, void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return hostStatus::SUCCESS;
  }

  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t*       d = static_cast<uint8_t*>(dst);
  size_t i = 0;

  for (; i + 256 <= size; i += 256) {
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 192), _MM_HINT_NTA);

    __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i));
    __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 32));
    __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 64));
    __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 96));
    __m256 v4 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 128));
    __m256 v5 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 160));
    __m256 v6 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 192));
    __m256 v7 = _mm256_loadu_ps(reinterpret_cast<const float*>(s + i + 224));

    _mm256_storeu_ps(reinterpret_cast<float*>(d + i),       v0);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 32),  v1);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 64),  v2);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 96),  v3);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 128), v4);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 160), v5);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 192), v6);
    _mm256_storeu_ps(reinterpret_cast<float*>(d + i + 224), v7);
  }

  if (i < size)
    std::memcpy(d + i, s + i, size - i);

  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceTemporalImpl(host_tag::Avx256Tag, float* dst, const float* src1, const float* src2, size_t count) {
  size_t i = 0;

  const uint8_t* s1 = reinterpret_cast<const uint8_t*>(src1);
  const uint8_t* s2 = reinterpret_cast<const uint8_t*>(src2);

  for (; i + 64 <= count; i += 64) {
    size_t bi = i * sizeof(float);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s1 + bi + 2048 + 192), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s2 + bi + 2048 + 192), _MM_HINT_NTA);

    __m256 a0 = _mm256_loadu_ps(src1 + i);
    __m256 b0 = _mm256_loadu_ps(src2 + i);
    __m256 a1 = _mm256_loadu_ps(src1 + i + 8);
    __m256 b1 = _mm256_loadu_ps(src2 + i + 8);
    __m256 a2 = _mm256_loadu_ps(src1 + i + 16);
    __m256 b2 = _mm256_loadu_ps(src2 + i + 16);
    __m256 a3 = _mm256_loadu_ps(src1 + i + 24);
    __m256 b3 = _mm256_loadu_ps(src2 + i + 24);
    __m256 a4 = _mm256_loadu_ps(src1 + i + 32);
    __m256 b4 = _mm256_loadu_ps(src2 + i + 32);
    __m256 a5 = _mm256_loadu_ps(src1 + i + 40);
    __m256 b5 = _mm256_loadu_ps(src2 + i + 40);
    __m256 a6 = _mm256_loadu_ps(src1 + i + 48);
    __m256 b6 = _mm256_loadu_ps(src2 + i + 48);
    __m256 a7 = _mm256_loadu_ps(src1 + i + 56);
    __m256 b7 = _mm256_loadu_ps(src2 + i + 56);

    _mm256_storeu_ps(dst + i,      _mm256_add_ps(a0, b0));
    _mm256_storeu_ps(dst + i + 8,  _mm256_add_ps(a1, b1));
    _mm256_storeu_ps(dst + i + 16, _mm256_add_ps(a2, b2));
    _mm256_storeu_ps(dst + i + 24, _mm256_add_ps(a3, b3));
    _mm256_storeu_ps(dst + i + 32, _mm256_add_ps(a4, b4));
    _mm256_storeu_ps(dst + i + 40, _mm256_add_ps(a5, b5));
    _mm256_storeu_ps(dst + i + 48, _mm256_add_ps(a6, b6));
    _mm256_storeu_ps(dst + i + 56, _mm256_add_ps(a7, b7));
  }

  for (; i + 8 <= count; i += 8) {
    __m256 a = _mm256_loadu_ps(src1 + i);
    __m256 b = _mm256_loadu_ps(src2 + i);
    _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
  }
  for (; i < count; ++i)
    dst[i] = src1[i] + src2[i];

  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* PCCL_AVX256_HOST_RT_IMPL_HPP */
