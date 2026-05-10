#ifndef PCCL_NEON_HOST_RT_IMPL_HPP
#define PCCL_NEON_HOST_RT_IMPL_HPP

#include <arm_neon.h>

#include <cstring>

#include "../host_rt.hpp"

namespace pccl {
namespace detail {

inline hostStatus hostMemcpyImpl(host_tag::NeonTag, void* dst, const void* src, size_t size) {
  // For small copies, fall back to memcpy — the NT overhead is not worth it.
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return hostStatus::SUCCESS;
  }

  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t*       d = static_cast<uint8_t*>(dst);
  size_t i = 0;

  // Main loop: 256 bytes (4 cache lines) per iteration, fully unrolled.
  // Uses 16 caller-saved Q registers (q0-q7, q16-q23) in a single asm block
  // to maximize memory-level parallelism: all 8 ldnp issued before any stnp.
  //
  // LDNP: non-temporal load pair — avoids cache pollution from remote data.
  // STNP: non-temporal store pair — bypasses write-allocate RFO round-trip.
  //
  // Software prefetch 2KB ahead covers ~200ns cross-NUMA latency at ~16 B/ns.
  // (A/B tested: 2KB > 4KB > no-prefetch on Kunpeng 920 HCCS.)
  for (; i + 256 <= size; i += 256) {
    __builtin_prefetch(s + i + 2048,       0, 0);
    __builtin_prefetch(s + i + 2048 + 64,  0, 0);
    __builtin_prefetch(s + i + 2048 + 128, 0, 0);
    __builtin_prefetch(s + i + 2048 + 192, 0, 0);

    __asm__ volatile(
        "ldp q0,  q1,  [%[s]]        \n"
        "ldp q2,  q3,  [%[s], #32]   \n"
        "ldp q4,  q5,  [%[s], #64]   \n"
        "ldp q6,  q7,  [%[s], #96]   \n"
        "ldp q16, q17, [%[s], #128]  \n"
        "ldp q18, q19, [%[s], #160]  \n"
        "ldp q20, q21, [%[s], #192]  \n"
        "ldp q22, q23, [%[s], #224]  \n"
        "stnp q0,  q1,  [%[d]]        \n"
        "stnp q2,  q3,  [%[d], #32]   \n"
        "stnp q4,  q5,  [%[d], #64]   \n"
        "stnp q6,  q7,  [%[d], #96]   \n"
        "stnp q16, q17, [%[d], #128]  \n"
        "stnp q18, q19, [%[d], #160]  \n"
        "stnp q20, q21, [%[d], #192]  \n"
        "stnp q22, q23, [%[d], #224]  \n"
        :
        : [s] "r" (s + i), [d] "r" (d + i)
        : "memory",
          "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
    );
  }

  // Tail: remaining bytes (< 256) with standard memcpy.
  if (i < size) {
    std::memcpy(d + i, s + i, size - i);
  }

  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceImpl(host_tag::NeonTag, float* dst, const float* src1, const float* src2, size_t count) {
  size_t i = 0;

  // Main loop: 8× unrolled, 32 floats (128 bytes) per iteration.
  // Two read streams (src1, src2) + one write stream (dst).
  // No read from dst — pure three-address operation.
  //
  // Software prefetch 2KB ahead (4 cache lines × 4 prefetches) on both src
  // streams covers ~200ns cross-NUMA latency at ~16 B/ns.
  // (Same distance as hostMemcpy, A/B tested on Kunpeng 920 HCCS.)
  const uint8_t* s1 = reinterpret_cast<const uint8_t*>(src1);
  const uint8_t* s2 = reinterpret_cast<const uint8_t*>(src2);

  for (; i + 32 <= count; i += 32) {
    size_t bi = i * sizeof(float);
    __builtin_prefetch(s1 + bi + 2048,       0, 0);
    __builtin_prefetch(s1 + bi + 2048 + 64,  0, 0);
    __builtin_prefetch(s1 + bi + 2048 + 128, 0, 0);
    __builtin_prefetch(s1 + bi + 2048 + 192, 0, 0);
    __builtin_prefetch(s2 + bi + 2048,       0, 0);
    __builtin_prefetch(s2 + bi + 2048 + 64,  0, 0);
    __builtin_prefetch(s2 + bi + 2048 + 128, 0, 0);
    __builtin_prefetch(s2 + bi + 2048 + 192, 0, 0);

    float32x4_t a0 = vld1q_f32(src1 + i);
    float32x4_t b0 = vld1q_f32(src2 + i);
    float32x4_t a1 = vld1q_f32(src1 + i +  4);
    float32x4_t b1 = vld1q_f32(src2 + i +  4);
    float32x4_t a2 = vld1q_f32(src1 + i +  8);
    float32x4_t b2 = vld1q_f32(src2 + i +  8);
    float32x4_t a3 = vld1q_f32(src1 + i + 12);
    float32x4_t b3 = vld1q_f32(src2 + i + 12);
    float32x4_t a4 = vld1q_f32(src1 + i + 16);
    float32x4_t b4 = vld1q_f32(src2 + i + 16);
    float32x4_t a5 = vld1q_f32(src1 + i + 20);
    float32x4_t b5 = vld1q_f32(src2 + i + 20);
    float32x4_t a6 = vld1q_f32(src1 + i + 24);
    float32x4_t b6 = vld1q_f32(src2 + i + 24);
    float32x4_t a7 = vld1q_f32(src1 + i + 28);
    float32x4_t b7 = vld1q_f32(src2 + i + 28);

    float32x4_t c0 = vaddq_f32(a0, b0);
    float32x4_t c1 = vaddq_f32(a1, b1);
    float32x4_t c2 = vaddq_f32(a2, b2);
    float32x4_t c3 = vaddq_f32(a3, b3);
    float32x4_t c4 = vaddq_f32(a4, b4);
    float32x4_t c5 = vaddq_f32(a5, b5);
    float32x4_t c6 = vaddq_f32(a6, b6);
    float32x4_t c7 = vaddq_f32(a7, b7);

    float* p = dst + i;
    __asm__ volatile("stnp %q[a], %q[b], [%[p]]"        :: [a]"w"(c0),[b]"w"(c1),[p]"r"(p) : "memory");
    __asm__ volatile("stnp %q[a], %q[b], [%[p], #32]"   :: [a]"w"(c2),[b]"w"(c3),[p]"r"(p) : "memory");
    __asm__ volatile("stnp %q[a], %q[b], [%[p], #64]"   :: [a]"w"(c4),[b]"w"(c5),[p]"r"(p) : "memory");
    __asm__ volatile("stnp %q[a], %q[b], [%[p], #96]"   :: [a]"w"(c6),[b]"w"(c7),[p]"r"(p) : "memory");
  }

  for (; i + 4 <= count; i += 4)
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(src1 + i), vld1q_f32(src2 + i)));
  for (; i < count; ++i)
    dst[i] = src1[i] + src2[i];

  return hostStatus::SUCCESS;
}

// ── Temporal-store variants: data stays in L3 for downstream CPU readers ──

inline hostStatus hostMemcpyTemporalImpl(host_tag::NeonTag, void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return hostStatus::SUCCESS;
  }

  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t*       d = static_cast<uint8_t*>(dst);
  size_t i = 0;

  // Same as hostMemcpyImpl but uses STP (temporal store pair) instead of STNP.
  // LDNP kept for reads (avoid polluting reader L3 with remote NUMA data).
  for (; i + 256 <= size; i += 256) {
    __builtin_prefetch(s + i + 2048,       0, 0);
    __builtin_prefetch(s + i + 2048 + 64,  0, 0);
    __builtin_prefetch(s + i + 2048 + 128, 0, 0);
    __builtin_prefetch(s + i + 2048 + 192, 0, 0);

    __asm__ volatile(
        "ldnp q0,  q1,  [%[s]]        \n"
        "ldnp q2,  q3,  [%[s], #32]   \n"
        "ldnp q4,  q5,  [%[s], #64]   \n"
        "ldnp q6,  q7,  [%[s], #96]   \n"
        "ldnp q16, q17, [%[s], #128]  \n"
        "ldnp q18, q19, [%[s], #160]  \n"
        "ldnp q20, q21, [%[s], #192]  \n"
        "ldnp q22, q23, [%[s], #224]  \n"
        "stp  q0,  q1,  [%[d]]        \n"
        "stp  q2,  q3,  [%[d], #32]   \n"
        "stp  q4,  q5,  [%[d], #64]   \n"
        "stp  q6,  q7,  [%[d], #96]   \n"
        "stp  q16, q17, [%[d], #128]  \n"
        "stp  q18, q19, [%[d], #160]  \n"
        "stp  q20, q21, [%[d], #192]  \n"
        "stp  q22, q23, [%[d], #224]  \n"
        :
        : [s] "r" (s + i), [d] "r" (d + i)
        : "memory",
          "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
    );
  }

  if (i < size) {
    std::memcpy(d + i, s + i, size - i);
  }

  return hostStatus::SUCCESS;
}

inline hostStatus hostReduceTemporalImpl(host_tag::NeonTag, float* dst, const float* src1, const float* src2, size_t count) {
  size_t i = 0;

  // Same as hostReduceImpl but uses vst1q_f32 (temporal stores) instead of stnp.
  // Data stays in L3 for downstream CPU consumers (H2H REDUCE/COPY).
  const uint8_t* s1 = reinterpret_cast<const uint8_t*>(src1);
  const uint8_t* s2 = reinterpret_cast<const uint8_t*>(src2);

  for (; i + 32 <= count; i += 32) {
    size_t bi = i * sizeof(float);
    __builtin_prefetch(s1 + bi + 2048,       0, 0);
    __builtin_prefetch(s1 + bi + 2048 + 64,  0, 0);
    __builtin_prefetch(s1 + bi + 2048 + 128, 0, 0);
    __builtin_prefetch(s1 + bi + 2048 + 192, 0, 0);
    __builtin_prefetch(s2 + bi + 2048,       0, 0);
    __builtin_prefetch(s2 + bi + 2048 + 64,  0, 0);
    __builtin_prefetch(s2 + bi + 2048 + 128, 0, 0);
    __builtin_prefetch(s2 + bi + 2048 + 192, 0, 0);

    float32x4_t a0 = vld1q_f32(src1 + i);
    float32x4_t b0 = vld1q_f32(src2 + i);
    float32x4_t a1 = vld1q_f32(src1 + i +  4);
    float32x4_t b1 = vld1q_f32(src2 + i +  4);
    float32x4_t a2 = vld1q_f32(src1 + i +  8);
    float32x4_t b2 = vld1q_f32(src2 + i +  8);
    float32x4_t a3 = vld1q_f32(src1 + i + 12);
    float32x4_t b3 = vld1q_f32(src2 + i + 12);
    float32x4_t a4 = vld1q_f32(src1 + i + 16);
    float32x4_t b4 = vld1q_f32(src2 + i + 16);
    float32x4_t a5 = vld1q_f32(src1 + i + 20);
    float32x4_t b5 = vld1q_f32(src2 + i + 20);
    float32x4_t a6 = vld1q_f32(src1 + i + 24);
    float32x4_t b6 = vld1q_f32(src2 + i + 24);
    float32x4_t a7 = vld1q_f32(src1 + i + 28);
    float32x4_t b7 = vld1q_f32(src2 + i + 28);

    // Temporal stores: data participates in L3 cache
    vst1q_f32(dst + i,      vaddq_f32(a0, b0));
    vst1q_f32(dst + i +  4, vaddq_f32(a1, b1));
    vst1q_f32(dst + i +  8, vaddq_f32(a2, b2));
    vst1q_f32(dst + i + 12, vaddq_f32(a3, b3));
    vst1q_f32(dst + i + 16, vaddq_f32(a4, b4));
    vst1q_f32(dst + i + 20, vaddq_f32(a5, b5));
    vst1q_f32(dst + i + 24, vaddq_f32(a6, b6));
    vst1q_f32(dst + i + 28, vaddq_f32(a7, b7));
  }

  for (; i + 4 <= count; i += 4)
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(src1 + i), vld1q_f32(src2 + i)));
  for (; i < count; ++i)
    dst[i] = src1[i] + src2[i];

  return hostStatus::SUCCESS;
}

}  // namespace detail
}  // namespace pccl

#endif /* PCCL_NEON_HOST_RT_IMPL_HPP */
