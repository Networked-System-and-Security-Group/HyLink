#include "pccl_aicore_reduce.h"

#include <acl/acl.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr size_t kBytes = 1024 * 1024;
constexpr int kWarmupIterations = 50;
constexpr int kMeasuredIterations = 200;

void Check(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(status));
    std::abort();
  }
}

float Percentile(std::vector<float> samples, float percentile) {
  assert(!samples.empty());
  std::sort(samples.begin(), samples.end());
  const size_t index = static_cast<size_t>((samples.size() - 1) * percentile);
  return samples[index];
}

void BenchmarkReduce(aclrtStream stream, aclDataType type, const char* label) {
  std::vector<uint8_t> host_dst(kBytes, 0);
  std::vector<uint8_t> host_src(kBytes, 0);
  void* device_dst = nullptr;
  void* device_src = nullptr;
  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;

  Check(aclrtMalloc(&device_dst, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(benchmark dst)");
  Check(aclrtMalloc(&device_src, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(benchmark src)");
  Check(aclrtMemcpy(device_dst, kBytes, host_dst.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy benchmark dst");
  Check(aclrtMemcpy(device_src, kBytes, host_src.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy benchmark src");
  Check(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE), "create benchmark start event");
  Check(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE), "create benchmark end event");

  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    Check(pcclAicoreReduceAsync(device_dst, device_src, kBytes, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, type, stream),
          "warmup reduce");
  }
  Check(aclrtSynchronizeStream(stream), "synchronize warmup stream");

  std::vector<float> samples_us;
  samples_us.reserve(kMeasuredIterations);
  for (int iteration = 0; iteration < kMeasuredIterations; ++iteration) {
    Check(aclrtRecordEvent(start, stream), "record benchmark start");
    Check(pcclAicoreReduceAsync(device_dst, device_src, kBytes, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, type, stream),
          "benchmark reduce");
    Check(aclrtRecordEvent(end, stream), "record benchmark end");
    Check(aclrtSynchronizeStream(stream), "synchronize benchmark stream");
    float elapsed_ms = 0;
    Check(aclrtEventElapsedTime(&elapsed_ms, start, end), "measure benchmark elapsed time");
    samples_us.push_back(elapsed_ms * 1000.0F);
  }

  std::printf("dtype=%s bytes=%zu block_dim_policy=auto p50_us=%.3f p80_us=%.3f\n", label, kBytes,
              Percentile(samples_us, 0.50F), Percentile(samples_us, 0.80F));

  Check(aclrtDestroyEvent(end), "destroy benchmark end event");
  Check(aclrtDestroyEvent(start), "destroy benchmark start event");
  Check(aclrtFree(device_src), "aclrtFree(benchmark src)");
  Check(aclrtFree(device_dst), "aclrtFree(benchmark dst)");
}

}  // namespace

int main() {
  Check(aclInit(nullptr), "aclInit");
  Check(aclrtSetDevice(0), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  Check(aclrtCreateStream(&stream), "aclrtCreateStream");

  BenchmarkReduce(stream, ACL_FLOAT, "fp32");
  BenchmarkReduce(stream, ACL_BF16, "bf16");

  Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
  Check(aclrtResetDevice(0), "aclrtResetDevice");
  Check(aclFinalize(), "aclFinalize");
  return 0;
}
