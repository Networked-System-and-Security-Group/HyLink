#ifndef HOST_RT_HPP
#define HOST_RT_HPP

#include <cstddef>
#include <cstdint>

namespace pccl {

// Host runtime status codes
enum class hostStatus { SUCCESS = 0, ERROR_INVALID_VALUE, ERROR_OUT_OF_MEMORY, ERROR_NOT_SUPPORTED, ERROR_UNKNOWN };

}  // namespace pccl

// Include tag definitions
#include "host_tags.hpp"

// Include the appropriate implementation based on CPU features
#if defined(__AVX512F__)
#include "../../src/hal/avx512_host_rt_impl.hpp"
#elif defined(__AVX2__)
#include "../../src/hal/avx256_host_rt_impl.hpp"
#elif defined(__ARM_NEON)
#include "../../src/hal/neon_host_rt_impl.hpp"
#else
#include "../../src/hal/standard_host_rt_impl.hpp"
#endif

// Public API: wrapper functions that hide the tag parameter
namespace pccl {

inline hostStatus hostMemcpy(void *dst, const void *src, size_t size) {
  return detail::hostMemcpyImpl(HostTag{}, dst, src, size);
}

inline hostStatus hostMemcpyAdd(float *dst, const float *src, size_t count) {
  return detail::hostMemcpyAddImpl(HostTag{}, dst, src, count);
}

}  // namespace pccl

#endif /* HOST_RT_HPP */
