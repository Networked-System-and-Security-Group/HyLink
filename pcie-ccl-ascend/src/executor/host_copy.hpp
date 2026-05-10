#ifndef PCCL_HOST_COPY_HPP
#define PCCL_HOST_COPY_HPP

#include <cstddef>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pccl {

namespace detail {

#if defined(__ARM_NEON)
inline void copyHostNeonNonTemporal(void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return;
  }

  const unsigned char* s = static_cast<const unsigned char*>(src);
  unsigned char* d = static_cast<unsigned char*>(dst);
  size_t i = 0;
  for (; i + 256 <= size; i += 256) {
    __builtin_prefetch(s + i + 2048, 0, 0);
    __builtin_prefetch(s + i + 2048 + 64, 0, 0);
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
        "stnp q0,  q1,  [%[d]]       \n"
        "stnp q2,  q3,  [%[d], #32]  \n"
        "stnp q4,  q5,  [%[d], #64]  \n"
        "stnp q6,  q7,  [%[d], #96]  \n"
        "stnp q16, q17, [%[d], #128] \n"
        "stnp q18, q19, [%[d], #160] \n"
        "stnp q20, q21, [%[d], #192] \n"
        "stnp q22, q23, [%[d], #224] \n"
        :
        : [s] "r"(s + i), [d] "r"(d + i)
        : "memory",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
  }
  if (i < size)
    std::memcpy(d + i, s + i, size - i);
}

inline void copyHostNeonTemporal(void* dst, const void* src, size_t size) {
  if (size < 2048) {
    std::memcpy(dst, src, size);
    return;
  }

  const unsigned char* s = static_cast<const unsigned char*>(src);
  unsigned char* d = static_cast<unsigned char*>(dst);
  size_t i = 0;
  for (; i + 256 <= size; i += 256) {
    __builtin_prefetch(s + i + 2048, 0, 0);
    __builtin_prefetch(s + i + 2048 + 64, 0, 0);
    __builtin_prefetch(s + i + 2048 + 128, 0, 0);
    __builtin_prefetch(s + i + 2048 + 192, 0, 0);

    __asm__ volatile(
        "ldnp q0,  q1,  [%[s]]       \n"
        "ldnp q2,  q3,  [%[s], #32]  \n"
        "ldnp q4,  q5,  [%[s], #64]  \n"
        "ldnp q6,  q7,  [%[s], #96]  \n"
        "ldnp q16, q17, [%[s], #128] \n"
        "ldnp q18, q19, [%[s], #160] \n"
        "ldnp q20, q21, [%[s], #192] \n"
        "ldnp q22, q23, [%[s], #224] \n"
        "stp  q0,  q1,  [%[d]]       \n"
        "stp  q2,  q3,  [%[d], #32]  \n"
        "stp  q4,  q5,  [%[d], #64]  \n"
        "stp  q6,  q7,  [%[d], #96]  \n"
        "stp  q16, q17, [%[d], #128] \n"
        "stp  q18, q19, [%[d], #160] \n"
        "stp  q20, q21, [%[d], #192] \n"
        "stp  q22, q23, [%[d], #224] \n"
        :
        : [s] "r"(s + i), [d] "r"(d + i)
        : "memory",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
  }
  if (i < size)
    std::memcpy(d + i, s + i, size - i);
}
#endif

}  // namespace detail

inline void copyHost(void* dst, const void* src, size_t size, bool temporal_write) {
#if defined(__ARM_NEON)
  if (temporal_write)
    detail::copyHostNeonTemporal(dst, src, size);
  else
    detail::copyHostNeonNonTemporal(dst, src, size);
#else
  (void)temporal_write;
  std::memcpy(dst, src, size);
#endif
}

}  // namespace pccl

#endif /* PCCL_HOST_COPY_HPP */
