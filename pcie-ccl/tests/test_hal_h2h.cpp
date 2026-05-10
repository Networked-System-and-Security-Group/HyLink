#include <numa.h>

#include <thread>
#include <vector>

#include "include/hal/host.hpp"
#include "test_utils.hpp"

using namespace pccl;

constexpr size_t TEST_SIZE = 512ULL * 1024 * 1024;  // 512MB

struct TestConfig {
  int num_threads;
  bool same_numa;
  bool is_add;
  bool bidirectional;
};

void runUnidirectionalCopy(void* dst, void* src, size_t size, int num_threads) {
  size_t chunk_size = size / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&, t]() {
      char* dst_ptr = (char*)dst + t * chunk_size;
      char* src_ptr = (char*)src + t * chunk_size;
      hostMemcpy(dst_ptr, src_ptr, chunk_size);
    });
  }

  for (auto& th : threads)
    th.join();
}

void runUnidirectionalAdd(float* dst, float* src, size_t count, int num_threads) {
  size_t chunk_count = count / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&, t]() {
      float* dst_ptr = dst + t * chunk_count;
      float* src_ptr = src + t * chunk_count;
      hostMemcpyAdd(dst_ptr, src_ptr, chunk_count);
    });
  }

  for (auto& th : threads)
    th.join();
}

void runBidirectionalCopy(void* dst1, void* src1, void* dst2, void* src2, size_t size, int num_threads) {
  size_t chunk_size = size / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&, t]() {
      char* d1 = (char*)dst1 + t * chunk_size;
      char* s1 = (char*)src1 + t * chunk_size;
      hostMemcpy(d1, s1, chunk_size);
    });
    threads.emplace_back([&, t]() {
      char* d2 = (char*)dst2 + t * chunk_size;
      char* s2 = (char*)src2 + t * chunk_size;
      hostMemcpy(d2, s2, chunk_size);
    });
  }

  for (auto& th : threads)
    th.join();
}

void runBidirectionalAdd(float* dst1, float* src1, float* dst2, float* src2, size_t count, int num_threads) {
  size_t chunk_count = count / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&, t]() {
      float* d1 = dst1 + t * chunk_count;
      float* s1 = src1 + t * chunk_count;
      hostMemcpyAdd(d1, s1, chunk_count);
    });
    threads.emplace_back([&, t]() {
      float* d2 = dst2 + t * chunk_count;
      float* s2 = src2 + t * chunk_count;
      hostMemcpyAdd(d2, s2, chunk_count);
    });
  }

  for (auto& th : threads)
    th.join();
}

void runTest(const TestConfig& cfg) {
  int numa0 = 0, numa1 = cfg.same_numa ? 0 : 1;

  if (!cfg.same_numa && numa_max_node() < 1) {
    printf("SKIP: threads=%d numa=%s op=%s dir=%s (only 1 NUMA node)\n", cfg.num_threads,
           cfg.same_numa ? "same" : "diff", cfg.is_add ? "add" : "copy", cfg.bidirectional ? "bidir" : "unidir");
    return;
  }

  void* src1 = numa_alloc_onnode(TEST_SIZE, numa0);
  void* dst1 = numa_alloc_onnode(TEST_SIZE, numa1);
  void *src2 = nullptr, *dst2 = nullptr;

  if (cfg.bidirectional) {
    src2 = numa_alloc_onnode(TEST_SIZE, numa1);
    dst2 = numa_alloc_onnode(TEST_SIZE, numa0);
  }

  printf("Buffer addresses: src1=%p dst1=%p", src1, dst1);
  if (cfg.bidirectional)
    printf(" src2=%p dst2=%p", src2, dst2);
  printf("\n");

  if (cfg.is_add) {
    float *s1 = (float*)src1, *d1 = (float*)dst1;
    size_t count = TEST_SIZE / sizeof(float);
    for (size_t i = 0; i < count; i++) {
      s1[i] = 1.0f;
      d1[i] = 2.0f;
    }

    if (cfg.bidirectional) {
      float *s2 = (float*)src2, *d2 = (float*)dst2;
      for (size_t i = 0; i < count; i++) {
        s2[i] = 1.0f;
        d2[i] = 2.0f;
      }
    }
  } else {
    memset(src1, 1, TEST_SIZE);
    memset(dst1, 0, TEST_SIZE);
    if (cfg.bidirectional) {
      memset(src2, 1, TEST_SIZE);
      memset(dst2, 0, TEST_SIZE);
    }
  }

  HostTimer timer;
  std::vector<float> times;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer.start();

    if (cfg.bidirectional) {
      if (cfg.is_add) {
        runBidirectionalAdd((float*)dst1, (float*)src1, (float*)dst2, (float*)src2, TEST_SIZE / sizeof(float),
                            cfg.num_threads);
      } else {
        runBidirectionalCopy(dst1, src1, dst2, src2, TEST_SIZE, cfg.num_threads);
      }
    } else {
      if (cfg.is_add) {
        runUnidirectionalAdd((float*)dst1, (float*)src1, TEST_SIZE / sizeof(float), cfg.num_threads);
      } else {
        runUnidirectionalCopy(dst1, src1, TEST_SIZE, cfg.num_threads);
      }
    }

    timer.end();
    float ms = timer.getElapsedMs();
    if (i >= WARMUP_ITERS)
      times.push_back(ms);

    if (cfg.is_add && i < VERIFY_ITERS) {
      float* d1 = (float*)dst1;
      size_t count = TEST_SIZE / sizeof(float);
      for (size_t j = 0; j < count; j++)
        d1[j] = 2.0f;
      if (cfg.bidirectional) {
        float* d2 = (float*)dst2;
        for (size_t j = 0; j < count; j++)
          d2[j] = 2.0f;
      }
    }
  }

  auto stats = calculateStats(times, cfg.bidirectional ? 2 * TEST_SIZE : TEST_SIZE);
  printf("threads=%2d numa=%s op=%s dir=%s: %.2f GB/s\n", cfg.num_threads, cfg.same_numa ? "same" : "diff",
         cfg.is_add ? "add " : "copy", cfg.bidirectional ? "bidir" : "unidir", stats.avg_bw_gbps);

  numa_free(src1, TEST_SIZE);
  numa_free(dst1, TEST_SIZE);
  if (cfg.bidirectional) {
    numa_free(src2, TEST_SIZE);
    numa_free(dst2, TEST_SIZE);
  }
}

int main() {
  if (numa_available() == -1) {
    printf("NUMA not available\n");
    return 1;
  }

  printf("Test size: %zu MB\n", TEST_SIZE / (1024 * 1024));
  printf("Warmup: %d, Measure: %d\n\n", WARMUP_ITERS, MEASURE_ITERS);

  int thread_counts[] = {1, 2, 4, 8, 16, 32};

  for (int threads : thread_counts) {
    for (bool same_numa : {true, false}) {
      for (bool is_add : {false, true}) {
        for (bool bidir : {false, true}) {
          runTest({threads, same_numa, is_add, bidir});
        }
      }
    }
  }

  return 0;
}
