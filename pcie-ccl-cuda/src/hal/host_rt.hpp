#ifndef PCCL_HOST_RT_HPP
#define PCCL_HOST_RT_HPP

#include <cstddef>
#include <cstdint>

namespace pccl {

enum class hostStatus { SUCCESS = 0, ERROR_INVALID_VALUE, ERROR_OUT_OF_MEMORY, ERROR_NOT_SUPPORTED, ERROR_UNKNOWN };

// Host tag dispatch
namespace host_tag {
struct Avx512Tag {};
struct Avx256Tag {};
struct NeonTag {};
struct StandardTag {};
}  // namespace host_tag

#if defined(__AVX512F__)
using HostTag = host_tag::Avx512Tag;
#elif defined(__AVX2__)
using HostTag = host_tag::Avx256Tag;
#elif defined(__ARM_NEON)
using HostTag = host_tag::NeonTag;
#else
using HostTag = host_tag::StandardTag;
#endif

}  // namespace pccl

// Include the appropriate implementation based on CPU features
#if defined(__AVX512F__)
#include "impl/avx512_host_rt_impl.hpp"
#elif defined(__AVX2__)
#include "impl/avx256_host_rt_impl.hpp"
#elif defined(__ARM_NEON)
#include "impl/neon_host_rt_impl.hpp"
#else
#include "impl/standard_host_rt_impl.hpp"
#endif

// Public API
namespace pccl {

inline hostStatus hostMemcpy(void* dst, const void* src, size_t size) {
  return detail::hostMemcpyImpl(HostTag{}, dst, src, size);
}

inline hostStatus hostReduce(float* dst, const float* src1, const float* src2, size_t count) {
  return detail::hostReduceImpl(HostTag{}, dst, src1, src2, count);
}

// Temporal-store variants: use cache-friendly stores (stp/vst1q_f32 instead of stnp)
// so that the written data stays in L3 for downstream CPU consumers.
inline hostStatus hostMemcpyTemporal(void* dst, const void* src, size_t size) {
  return detail::hostMemcpyTemporalImpl(HostTag{}, dst, src, size);
}

inline hostStatus hostReduceTemporal(float* dst, const float* src1, const float* src2, size_t count) {
  return detail::hostReduceTemporalImpl(HostTag{}, dst, src1, src2, count);
}

}  // namespace pccl

#endif /* PCCL_HOST_RT_HPP */
