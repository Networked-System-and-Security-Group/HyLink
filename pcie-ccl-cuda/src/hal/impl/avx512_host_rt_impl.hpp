#ifndef PCCL_AVX512_HOST_RT_IMPL_HPP
#define PCCL_AVX512_HOST_RT_IMPL_HPP

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "../host_rt.hpp"

namespace pccl {
namespace detail {

// ── Non-temporal (default): bypass L3, write directly to DRAM ─────────────
// Matches NEON's STNP behavior. Use for data consumed by DMA (H2D) or
// cross-NUMA readers that would evict useful L3 lines.

inline hostStatus hostMemcpyImpl(host_tag::Avx512Tag, void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return hostStatus::SUCCESS;
  }

  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t*       d = static_cast<uint8_t*>(dst);
  size_t i = 0;

  // Align dst to 64-byte boundary (required by _mm512_stream_ps)
  size_t align_offset = (64 - (reinterpret_cast<uintptr_t>(d) & 63)) & 63;
  if (align_offset > 0) {
    size_t head = (align_offset < size) ? align_offset : size;
    std::memcpy(d, s, head);
    i = head;
  }

  // Main loop: 256 bytes (4 × 64B cache lines) per iteration, NT stores.
  // Prefetch 2KB ahead on src (matches NEON pattern).
  for (; i + 256 <= size; i += 256) {
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 192), _MM_HINT_NTA);

    __m512 v0 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i));
    __m512 v1 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i + 64));
    __m512 v2 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i + 128));
    __m512 v3 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i + 192));

    _mm512_stream_ps(reinterpret_cast<float*>(d + i),       v0);
    _mm512_stream_ps(reinterpret_cast<float*>(d + i + 64),  v1);
    _mm512_stream_ps(reinterpret_cast<float*>(d + i + 128), v2);
    _mm512_stream_ps(reinterpret_cast<float*>(d + i + 192), v3);
  }

  if (i < size)
    std::memcpy(d + i, s + i, size - i);

  _mm_sfence();
  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceImpl(host_tag::Avx512Tag, float* dst, const float* src1, const float* src2, size_t count) {
  size_t i = 0;

  // Align dst to 64-byte boundary (16 floats) for stream stores
  size_t align_floats = (16 - ((reinterpret_cast<uintptr_t>(dst) >> 2) & 15)) & 15;
  if (align_floats > 0) {
    size_t head = (align_floats < count) ? align_floats : count;
    for (size_t j = 0; j < head; ++j)
      dst[j] = src1[j] + src2[j];
    i = head;
  }

  // Main loop: 4× unrolled, 64 floats (256 bytes) per iteration.
  // NT stores bypass L3; prefetch 2KB ahead on both sources.
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

    __m512 a0 = _mm512_loadu_ps(src1 + i);
    __m512 b0 = _mm512_loadu_ps(src2 + i);
    __m512 a1 = _mm512_loadu_ps(src1 + i + 16);
    __m512 b1 = _mm512_loadu_ps(src2 + i + 16);
    __m512 a2 = _mm512_loadu_ps(src1 + i + 32);
    __m512 b2 = _mm512_loadu_ps(src2 + i + 32);
    __m512 a3 = _mm512_loadu_ps(src1 + i + 48);
    __m512 b3 = _mm512_loadu_ps(src2 + i + 48);

    _mm512_stream_ps(dst + i,      _mm512_add_ps(a0, b0));
    _mm512_stream_ps(dst + i + 16, _mm512_add_ps(a1, b1));
    _mm512_stream_ps(dst + i + 32, _mm512_add_ps(a2, b2));
    _mm512_stream_ps(dst + i + 48, _mm512_add_ps(a3, b3));
  }

  for (; i + 16 <= count; i += 16) {
    __m512 a = _mm512_loadu_ps(src1 + i);
    __m512 b = _mm512_loadu_ps(src2 + i);
    _mm512_stream_ps(dst + i, _mm512_add_ps(a, b));
  }
  for (; i < count; ++i)
    dst[i] = src1[i] + src2[i];

  _mm_sfence();
  return hostStatus::SUCCESS;
}

// ── Temporal: data stays in L3 for downstream CPU consumers ───────────────
// Matches NEON's STP/vst1q_f32 behavior. Use for intermediate H2H results
// that will be read by another H2H stage on the same or nearby NUMA node.

inline hostStatus hostMemcpyTemporalImpl(host_tag::Avx512Tag, void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return hostStatus::SUCCESS;
  }

  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t*       d = static_cast<uint8_t*>(dst);
  size_t i = 0;

  // Main loop: 256 bytes per iteration, temporal stores (cache-friendly).
  // Prefetch src 2KB ahead with NTA hint (avoid polluting reader L3 with remote data).
  for (; i + 256 <= size; i += 256) {
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048),       _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 64),  _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 128), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(s + i + 2048 + 192), _MM_HINT_NTA);

    __m512 v0 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i));
    __m512 v1 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i + 64));
    __m512 v2 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i + 128));
    __m512 v3 = _mm512_loadu_ps(reinterpret_cast<const float*>(s + i + 192));

    _mm512_storeu_ps(reinterpret_cast<float*>(d + i),       v0);
    _mm512_storeu_ps(reinterpret_cast<float*>(d + i + 64),  v1);
    _mm512_storeu_ps(reinterpret_cast<float*>(d + i + 128), v2);
    _mm512_storeu_ps(reinterpret_cast<float*>(d + i + 192), v3);
  }

  if (i < size)
    std::memcpy(d + i, s + i, size - i);

  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceTemporalImpl(host_tag::Avx512Tag, float* dst, const float* src1, const float* src2, size_t count) {
  size_t i = 0;

  // Temporal stores: data stays in L3 for downstream CPU readers.
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

    __m512 a0 = _mm512_loadu_ps(src1 + i);
    __m512 b0 = _mm512_loadu_ps(src2 + i);
    __m512 a1 = _mm512_loadu_ps(src1 + i + 16);
    __m512 b1 = _mm512_loadu_ps(src2 + i + 16);
    __m512 a2 = _mm512_loadu_ps(src1 + i + 32);
    __m512 b2 = _mm512_loadu_ps(src2 + i + 32);
    __m512 a3 = _mm512_loadu_ps(src1 + i + 48);
    __m512 b3 = _mm512_loadu_ps(src2 + i + 48);

    _mm512_storeu_ps(dst + i,      _mm512_add_ps(a0, b0));
    _mm512_storeu_ps(dst + i + 16, _mm512_add_ps(a1, b1));
    _mm512_storeu_ps(dst + i + 32, _mm512_add_ps(a2, b2));
    _mm512_storeu_ps(dst + i + 48, _mm512_add_ps(a3, b3));
  }

  for (; i + 16 <= count; i += 16) {
    __m512 a = _mm512_loadu_ps(src1 + i);
    __m512 b = _mm512_loadu_ps(src2 + i);
    _mm512_storeu_ps(dst + i, _mm512_add_ps(a, b));
  }
  for (; i < count; ++i)
    dst[i] = src1[i] + src2[i];

  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* PCCL_AVX512_HOST_RT_IMPL_HPP */
