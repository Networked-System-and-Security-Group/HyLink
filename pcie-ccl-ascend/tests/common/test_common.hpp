#ifndef PCCL_TEST_COMMON_HPP
#define PCCL_TEST_COMMON_HPP

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "runtime/ascend_runtime.hpp"
#include "pccl/pccl.h"
#include "topology/topology.hpp"

using namespace pccl;

/* ------------------------------------------------------------------ */
/*  Single-process UniqueId (all zeros = single-process mode)          */
/* ------------------------------------------------------------------ */

static inline pcclUniqueId singleProcessId() {
  pcclUniqueId id;
  std::memset(&id, 0, sizeof(id));
  return id;
}

/* ------------------------------------------------------------------ */
/*  Error-checking macros                                              */
/* ------------------------------------------------------------------ */

#define CHECK_PCCL(call)                                                                             \
  do {                                                                                               \
    pcclResult_t _r = (call);                                                                        \
    if (_r != pcclSuccess) {                                                                         \
      std::fprintf(stderr, "PCCL error at %s:%d: %s\n", __FILE__, __LINE__, pcclGetErrorString(_r)); \
      std::exit(1);                                                                                  \
    }                                                                                                \
  } while (0)

#define CHECK_DEV(call)                                                                   \
  do {                                                                                    \
    pccl::devStatus _s = (call);                                                          \
    if (_s != pccl::devSuccess) {                                                         \
      std::fprintf(stderr, "Device error at %s:%d: status %d\n", __FILE__, __LINE__, _s); \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

/* ------------------------------------------------------------------ */
/*  Correctness size set                                               */
/* ------------------------------------------------------------------ */

struct TestSize {
  const char* label;
  size_t bytes;
};

static const TestSize TEST_SIZES[] = {
    {"1 KB",   1u << 10}, {"2 KB",   1u << 11}, {"4 KB",   1u << 12}, {"8 KB",   1u << 13},
    {"16 KB",  1u << 14}, {"32 KB",  1u << 15}, {"64 KB",  1u << 16}, {"128 KB", 1u << 17},
    {"256 KB", 1u << 18}, {"512 KB", 1u << 19}, {"1 MB",   1u << 20}, {"2 MB",   1u << 21},
    {"4 MB",   1u << 22}, {"8 MB",   1u << 23}, {"16 MB",  1u << 24}, {"32 MB",  1u << 25},
    {"64 MB",  1u << 26}, {"128 MB", 1u << 27}, {"256 MB", 1u << 28}, {"512 MB", 1u << 29},
    {"1 GB",   1ull << 30},
};
static const int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

inline size_t testMaxBytes() {
  const char* env = std::getenv("PCCL_TEST_MAX_BYTES");
  if (!env || env[0] == '\0')
    return TEST_SIZES[NUM_TEST_SIZES - 1].bytes;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed == 0)
    return TEST_SIZES[NUM_TEST_SIZES - 1].bytes;
  size_t scale = 1;
  if (*end != '\0') {
    if ((end[0] == 'K' || end[0] == 'k') && end[1] == '\0')
      scale = 1ull << 10;
    else if ((end[0] == 'M' || end[0] == 'm') && end[1] == '\0')
      scale = 1ull << 20;
    else if ((end[0] == 'G' || end[0] == 'g') && end[1] == '\0')
      scale = 1ull << 30;
    else
      return TEST_SIZES[NUM_TEST_SIZES - 1].bytes;
  }
  return static_cast<size_t>(parsed) * scale;
}

inline int numTestSizes() {
  size_t max_bytes = testMaxBytes();
  int count = 0;
  while (count < NUM_TEST_SIZES && TEST_SIZES[count].bytes <= max_bytes)
    ++count;
  return count > 0 ? count : 1;
}

/* ------------------------------------------------------------------ */
/*  Correctness table printing helpers                                 */
/* ------------------------------------------------------------------ */

inline void printTestHeader(const char* title) {
  std::printf("\n=== %s correctness ===\n\n", title);
  std::printf("%-22s  %s\n", "Case", "Correct");
  std::printf("%-22s  %s\n", "----------------------", "-------");
}

inline void printTestRow(const char* label, bool correct) {
  std::printf("%-22s  %s\n", label, correct ? "PASS" : "FAIL");
}

inline int stridedPerfIters() {
  const char* env = std::getenv("PCCL_STRIDED_PERF_ITERS");
  if (!env || env[0] == '\0')
    return 5;
  char* end = nullptr;
  long parsed = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || parsed <= 0)
    return 5;
  return static_cast<int>(parsed);
}

inline int stridedPerfWarmupIters() {
  const char* env = std::getenv("PCCL_STRIDED_PERF_WARMUP");
  if (!env || env[0] == '\0')
    return 2;
  char* end = nullptr;
  long parsed = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || parsed < 0)
    return 2;
  return static_cast<int>(parsed);
}

inline double stridedBusFactor(int nranks) {
  return (static_cast<double>(nranks) - 1.0) / static_cast<double>(nranks);
}

inline void printStridedPerfHeader(int iters, int warmup_iters) {
  std::printf("Perf: measured_iters=%d warmup_iters=%d, bandwidth uses active bytes only\n", iters, warmup_iters);
  std::printf("%-22s  %-7s  %-13s  %-14s  %-14s\n", "Case", "Correct", "avg_time(us)", "alg_bw(GB/s)",
              "bus_bw(GB/s)");
  std::printf("%-22s  %-7s  %-13s  %-14s  %-14s\n", "----------------------", "-------", "------------",
              "-------------", "-------------");
}

inline void printStridedPerfRow(const char* label, bool correct, double avg_us, double alg_bw, double bus_bw) {
  std::printf("%-22s  %-7s  %-13.2f  %-14.5f  %-14.5f\n", label, correct ? "PASS" : "FAIL", avg_us, alg_bw,
              bus_bw);
}

/* ------------------------------------------------------------------ */
/*  Correctness verification (float comparison)                        */
/* ------------------------------------------------------------------ */

inline bool verifyBuffer(const float* buf, size_t count, float expected, int rank, const char* tag) {
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(buf[i] - expected) > 1e-5f) {
      std::fprintf(stderr, "  FAIL rank %d %s: buf[%zu] = %f, expected %f\n", rank, tag, i, buf[i], expected);
      return false;
    }
  }
  return true;
}

/* ------------------------------------------------------------------ */
/*  Persistent per-rank thread pool for parallel collective submit     */
/* ------------------------------------------------------------------ */

// Creates nranks persistent worker threads. Each call to run(fn) dispatches
// fn(r) on thread r and blocks until all ranks complete.
class RankThreadPool {
 public:
  explicit RankThreadPool(int nranks)
      : nranks_(nranks), generation_(0), done_count_(0), stop_(false) {
    threads_.reserve(nranks);
    for (int r = 0; r < nranks; ++r)
      threads_.emplace_back([this, r] { workerLoop(r); });
  }

  ~RankThreadPool() {
    {
      std::lock_guard<std::mutex> lk(dispatch_mu_);
      stop_ = true;
    }
    dispatch_cv_.notify_all();
    for (auto& t : threads_) t.join();
  }

  RankThreadPool(const RankThreadPool&) = delete;
  RankThreadPool& operator=(const RankThreadPool&) = delete;

  // Dispatch fn(r) to each worker thread r and wait for all to finish.
  template <typename Fn>
  void run(Fn&& fn) {
    fn_ = std::forward<Fn>(fn);
    {
      std::lock_guard<std::mutex> lk(dispatch_mu_);
      done_count_.store(0, std::memory_order_relaxed);
      ++generation_;
    }
    dispatch_cv_.notify_all();
    std::unique_lock<std::mutex> lk(main_mu_);
    main_cv_.wait(lk, [this] {
      return done_count_.load(std::memory_order_acquire) == nranks_;
    });
  }

 private:
  void workerLoop(int r) {
    int seen_gen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lk(dispatch_mu_);
        dispatch_cv_.wait(lk, [&] { return generation_ > seen_gen || stop_; });
        if (stop_) break;
        seen_gen = generation_;
      }
      fn_(r);
      // The worker that brings done_count to nranks_ notifies the main thread.
      if (done_count_.fetch_add(1, std::memory_order_acq_rel) + 1 == nranks_)
        main_cv_.notify_one();
    }
  }

  int nranks_;
  std::function<void(int)> fn_;

  std::mutex dispatch_mu_;
  std::condition_variable dispatch_cv_;
  int generation_;
  bool stop_;

  std::mutex main_mu_;
  std::condition_variable main_cv_;
  std::atomic<int> done_count_;

  std::vector<std::thread> threads_;
};

/* ------------------------------------------------------------------ */
/*  Strided test support                                               */
/* ------------------------------------------------------------------ */

struct StrideCfg {
  size_t seg_count; /* floats per segment */
  size_t stride;    /* reserved-prefix floats per segment */
  const char* label;
};

static const StrideCfg STRIDE_CONFIGS[] = {
    {32u << 20, 24u << 20, "25% active region"},
    {32u << 20, 16u << 20, "50% active region"},
    {32u << 20, 8u << 20, "75% active region"},
    {32u << 20, 0, "contiguous"},
};
static const int NUM_STRIDE_CONFIGS = sizeof(STRIDE_CONFIGS) / sizeof(STRIDE_CONFIGS[0]);

/* ------------------------------------------------------------------ */
/*  NUMA topology printing                                             */
/* ------------------------------------------------------------------ */

inline void printNumaTopology(int nranks) {
  pccl::LocalTopology topo = pccl::discoverLocalTopology(nranks);
  std::printf("\n--- NPU NUMA Topology (%d NPUs, %d NUMA nodes) ---\n", nranks, topo.num_numa_nodes);
  for (int npu = 0; npu < nranks; ++npu) {
    int logical = topo.getNUMAForDevice(npu);
    int physical = topo.getPhysicalNUMA(logical);
    char bus_id[32] = {};
    pccl::devGetPciBusId(npu, bus_id, sizeof(bus_id));
    std::printf("  NPU %d  PCI %s  -> NUMA %d (physical %d)\n", npu, bus_id, logical, physical);
  }
  std::printf("\n");
}

#endif /* PCCL_TEST_COMMON_HPP */
