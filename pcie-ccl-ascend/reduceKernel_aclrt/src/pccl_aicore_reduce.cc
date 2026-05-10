#include "pccl_aicore_reduce.h"

#include "launch_reduce_aicore.h"

#include <algorithm>
#include <cstdlib>

namespace {

static bool SkipPointerCheck() {
  static int val = -1;
  if (val == -1) {
    const char* env = std::getenv("PCCL_AICORE_SKIP_POINTER_CHECK");
    val = (env && std::atoi(env) != 0) ? 1 : 0;
  }
  return val == 1;
}

struct VerifiedRange {
  uintptr_t start = 0;
  uintptr_t end = 0;
  int32_t device = -1;
};

static thread_local VerifiedRange g_cached_dst{};
static thread_local VerifiedRange g_cached_src{};

bool PointersAreOnCurrentDevice(void* dst, const void* src, uint64_t count_bytes) {
  if (SkipPointerCheck()) {
    return true;
  }

  int32_t current_device = -1;
  if (aclrtGetDevice(&current_device) != ACL_SUCCESS)
    return false;

  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);

  bool dst_ok = (g_cached_dst.device == current_device &&
                 dst_addr >= g_cached_dst.start &&
                 (dst_addr + count_bytes) <= g_cached_dst.end);
  bool src_ok = (g_cached_src.device == current_device &&
                 src_addr >= g_cached_src.start &&
                 (src_addr + count_bytes) <= g_cached_src.end);

  if (dst_ok && src_ok) {
    return true;
  }

  if (!dst_ok) {
    aclrtPtrAttributes dst_attributes{};
    if (aclrtPointerGetAttributes(dst, &dst_attributes) != ACL_SUCCESS ||
        dst_attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE ||
        static_cast<int32_t>(dst_attributes.location.id) != current_device) {
      return false;
    }
    g_cached_dst.start = dst_addr;
    g_cached_dst.end = dst_addr + std::max(count_bytes, uint64_t{1024 * 1024 * 1024});
    g_cached_dst.device = current_device;
  }

  if (!src_ok) {
    aclrtPtrAttributes src_attributes{};
    if (aclrtPointerGetAttributes(src, &src_attributes) != ACL_SUCCESS ||
        src_attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE ||
        static_cast<int32_t>(src_attributes.location.id) != current_device) {
      return false;
    }
    g_cached_src.start = src_addr;
    g_cached_src.end = src_addr + std::max(count_bytes, uint64_t{1024 * 1024 * 1024});
    g_cached_src.device = current_device;
  }

  return true;
}

aclError LaunchSelectedAicoreKernel(void* dst, const void* src, uint64_t count_bytes,
                                    aclrtReduceKind kind, aclDataType type, aclrtStream stream) {
  if (type == ACL_FLOAT16) {
    switch (kind) {
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM:
        return LaunchPcclAicoreReduceFloat16Sum(dst, src, count_bytes, stream);
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX:
        return LaunchPcclAicoreReduceFloat16Max(dst, src, count_bytes, stream);
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN:
        return LaunchPcclAicoreReduceFloat16Min(dst, src, count_bytes, stream);
      default:
        return ACL_ERROR_API_NOT_SUPPORT;
    }
  }

  if (type == ACL_FLOAT) {
    switch (kind) {
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM:
        return LaunchPcclAicoreReduceFloatSum(dst, src, count_bytes, stream);
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX:
        return LaunchPcclAicoreReduceFloatMax(dst, src, count_bytes, stream);
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN:
        return LaunchPcclAicoreReduceFloatMin(dst, src, count_bytes, stream);
      default:
        return ACL_ERROR_API_NOT_SUPPORT;
    }
  }

  if (type == ACL_BF16) {
    switch (kind) {
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM:
        return LaunchPcclAicoreReduceBfloat16Sum(dst, src, count_bytes, stream);
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX:
        return LaunchPcclAicoreReduceBfloat16Max(dst, src, count_bytes, stream);
      case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN:
        return LaunchPcclAicoreReduceBfloat16Min(dst, src, count_bytes, stream);
      default:
        return ACL_ERROR_API_NOT_SUPPORT;
    }
  }

  return ACL_ERROR_API_NOT_SUPPORT;
}

}  // namespace

extern "C" aclError pcclAicoreReduceAsync(void* dst, const void* src, uint64_t count_bytes,
                                            aclrtReduceKind kind, aclDataType type,
                                            aclrtStream stream) {
  if (!dst || !src || !pccl::aicore_reduce::IsAicoreReduceSupported(kind, type, count_bytes) ||
      !PointersAreOnCurrentDevice(dst, src, count_bytes)) {
    return ACL_ERROR_API_NOT_SUPPORT;
  }

  return LaunchSelectedAicoreKernel(dst, src, count_bytes, kind, type, stream);
}
