#include "pccl_aicore_reduce.h"

#include <acl/acl.h>

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t kElements = 65539;
constexpr size_t kLargeFloatElements = (1024 * 1024 / sizeof(float)) + 3;
constexpr size_t kLargeBfloat16Elements = (1024 * 1024 / sizeof(uint16_t)) + 3;

void Check(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(status));
    std::abort();
  }
}

void RunFloatCase(aclrtStream stream, aclrtReduceKind kind, float dst_value, float src_value,
                  float expected) {
  std::vector<float> dst(kElements, dst_value);
  std::vector<float> src(kElements, src_value);
  void* device_dst = nullptr;
  void* device_src = nullptr;
  const size_t bytes = kElements * sizeof(float);
  Check(aclrtMalloc(&device_dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(dst)");
  Check(aclrtMalloc(&device_src, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(src)");
  Check(aclrtMemcpy(device_dst, bytes, dst.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy dst to device");
  Check(aclrtMemcpy(device_src, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy src to device");
  Check(pcclAicoreReduceAsync(device_dst, device_src, bytes, kind, ACL_FLOAT, stream),
        "pcclAicoreReduceAsync(float)");
  Check(aclrtSynchronizeStream(stream), "synchronize float stream");
  Check(aclrtMemcpy(dst.data(), bytes, device_dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy float result");
  for (size_t index = 0; index < dst.size(); ++index) {
    if (dst[index] != expected) {
      std::fprintf(stderr, "float mismatch kind=%d index=%zu expected=%f actual=%f\n",
                   static_cast<int>(kind), index, expected, dst[index]);
      std::abort();
    }
  }
  Check(aclrtFree(device_src), "aclrtFree(src)");
  Check(aclrtFree(device_dst), "aclrtFree(dst)");
}

void RunFloat16Case(aclrtStream stream, aclrtReduceKind kind, uint16_t dst_value, uint16_t src_value,
                    uint16_t expected) {
  std::vector<uint16_t> dst(kElements, dst_value);
  std::vector<uint16_t> src(kElements, src_value);
  void* device_dst = nullptr;
  void* device_src = nullptr;
  const size_t bytes = kElements * sizeof(uint16_t);
  Check(aclrtMalloc(&device_dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(fp16 dst)");
  Check(aclrtMalloc(&device_src, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(fp16 src)");
  Check(aclrtMemcpy(device_dst, bytes, dst.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy fp16 dst");
  Check(aclrtMemcpy(device_src, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy fp16 src");
  Check(pcclAicoreReduceAsync(device_dst, device_src, bytes, kind, ACL_FLOAT16, stream),
        "pcclAicoreReduceAsync(fp16)");
  Check(aclrtSynchronizeStream(stream), "synchronize fp16 stream");
  Check(aclrtMemcpy(dst.data(), bytes, device_dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy fp16 result");
  for (size_t index = 0; index < dst.size(); ++index) {
    if (dst[index] != expected) {
      std::fprintf(stderr, "fp16 mismatch kind=%d index=%zu expected=0x%04x actual=0x%04x\n",
                   static_cast<int>(kind), index, expected, dst[index]);
      std::abort();
    }
  }
  Check(aclrtFree(device_src), "aclrtFree(fp16 src)");
  Check(aclrtFree(device_dst), "aclrtFree(fp16 dst)");
}

void RunBfloat16Case(aclrtStream stream, aclrtReduceKind kind, uint16_t dst_value,
                     uint16_t src_value, uint16_t expected) {
  std::vector<uint16_t> dst(kElements, dst_value);
  std::vector<uint16_t> src(kElements, src_value);
  void* device_dst = nullptr;
  void* device_src = nullptr;
  const size_t bytes = kElements * sizeof(uint16_t);
  Check(aclrtMalloc(&device_dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(bf16 dst)");
  Check(aclrtMalloc(&device_src, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(bf16 src)");
  Check(aclrtMemcpy(device_dst, bytes, dst.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy bf16 dst to device");
  Check(aclrtMemcpy(device_src, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy bf16 src to device");
  Check(pcclAicoreReduceAsync(device_dst, device_src, bytes, kind, ACL_BF16, stream),
        "pcclAicoreReduceAsync(bf16)");
  Check(aclrtSynchronizeStream(stream), "synchronize bf16 stream");
  Check(aclrtMemcpy(dst.data(), bytes, device_dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "copy bf16 result");
  for (size_t index = 0; index < dst.size(); ++index) {
    if (dst[index] != expected) {
      std::fprintf(stderr, "bf16 mismatch kind=%d index=%zu expected=0x%04x actual=0x%04x\n",
                   static_cast<int>(kind), index, expected, dst[index]);
      std::abort();
    }
  }
  Check(aclrtFree(device_src), "aclrtFree(bf16 src)");
  Check(aclrtFree(device_dst), "aclrtFree(bf16 dst)");
}

void RunLargeBfloat16Case(aclrtStream stream) {
  std::vector<uint16_t> dst(kLargeBfloat16Elements, 0x3f80);
  std::vector<uint16_t> src(kLargeBfloat16Elements, 0x4000);
  void* device_dst = nullptr;
  void* device_src = nullptr;
  const size_t bytes = kLargeBfloat16Elements * sizeof(uint16_t);
  Check(aclrtMalloc(&device_dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(large bf16 dst)");
  Check(aclrtMalloc(&device_src, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(large bf16 src)");
  Check(aclrtMemcpy(device_dst, bytes, dst.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy large bf16 dst");
  Check(aclrtMemcpy(device_src, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy large bf16 src");
  Check(pcclAicoreReduceAsync(device_dst, device_src, bytes, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM,
                              ACL_BF16, stream),
        "pcclAicoreReduceAsync(large bf16)");
  Check(aclrtSynchronizeStream(stream), "synchronize large bf16 stream");
  Check(aclrtMemcpy(dst.data(), bytes, device_dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "copy large bf16 result");
  for (size_t index = 0; index < dst.size(); ++index) {
    if (dst[index] != 0x4040) {
      std::fprintf(stderr, "large bf16 mismatch index=%zu expected=0x4040 actual=0x%04x\n",
                   index, dst[index]);
      std::abort();
    }
  }
  Check(aclrtFree(device_src), "aclrtFree(large bf16 src)");
  Check(aclrtFree(device_dst), "aclrtFree(large bf16 dst)");
}

void RunLargeFloatCase(aclrtStream stream, aclrtReduceKind kind) {
  std::vector<float> dst(kLargeFloatElements);
  std::vector<float> src(kLargeFloatElements);
  for (size_t index = 0; index < kLargeFloatElements; ++index) {
    dst[index] = static_cast<float>(index + 1);
    src[index] = 2.0F * static_cast<float>(index + 1);
  }

  void* device_dst = nullptr;
  void* device_src = nullptr;
  const size_t bytes = kLargeFloatElements * sizeof(float);
  Check(aclrtMalloc(&device_dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(large float dst)");
  Check(aclrtMalloc(&device_src, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(large float src)");
  Check(aclrtMemcpy(device_dst, bytes, dst.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy large float dst");
  Check(aclrtMemcpy(device_src, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy large float src");
  Check(pcclAicoreReduceAsync(device_dst, device_src, bytes, kind, ACL_FLOAT, stream),
        "pcclAicoreReduceAsync(large float)");
  Check(aclrtSynchronizeStream(stream), "synchronize large float stream");
  Check(aclrtMemcpy(dst.data(), bytes, device_dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy large float result");

  for (size_t index = 0; index < kLargeFloatElements; ++index) {
    const float expected = kind == ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM
                               ? 3.0F * static_cast<float>(index + 1)
                               : 2.0F * static_cast<float>(index + 1);
    if (dst[index] != expected) {
      std::fprintf(stderr, "large float mismatch kind=%d index=%zu expected=%f actual=%f\n",
                   static_cast<int>(kind), index, expected, dst[index]);
      std::abort();
    }
  }

  Check(aclrtFree(device_src), "aclrtFree(large float src)");
  Check(aclrtFree(device_dst), "aclrtFree(large float dst)");
}

}  // namespace

int main() {
  Check(aclInit(nullptr), "aclInit");
  Check(aclrtSetDevice(0), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  Check(aclrtCreateStream(&stream), "aclrtCreateStream");

  RunFloatCase(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, 1.0F, 2.0F, 3.0F);
  RunFloatCase(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, 1.0F, 2.0F, 2.0F);
  RunFloatCase(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, 2.0F, 1.0F, 1.0F);
  RunLargeFloatCase(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM);
  RunLargeFloatCase(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX);

  RunFloat16Case(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, 0x3c00, 0x4000, 0x4200);
  RunFloat16Case(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, 0x3c00, 0x4000, 0x4000);
  RunFloat16Case(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, 0x4000, 0x3c00, 0x3c00);

  RunBfloat16Case(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, 0x3f80, 0x4000, 0x4040);
  RunBfloat16Case(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, 0x3f80, 0x4000, 0x4000);
  RunBfloat16Case(stream, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, 0x4000, 0x3f80, 0x3f80);
  RunLargeBfloat16Case(stream);

  assert(pcclAicoreReduceAsync(reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000), 16,
                                ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_INT32, stream) ==
         ACL_ERROR_API_NOT_SUPPORT);

  Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
  Check(aclrtResetDevice(0), "aclrtResetDevice");
  Check(aclFinalize(), "aclFinalize");
  return 0;
}
