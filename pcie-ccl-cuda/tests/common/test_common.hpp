#ifndef PCCL_TEST_COMMON_HPP
#define PCCL_TEST_COMMON_HPP

#include <algorithm>
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
#include <vector>

#include "hal/device_rt.hpp"
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
/*  Benchmark constants                                                */
/* ------------------------------------------------------------------ */

static const int WARMUP_ITERS = 2;
static const int MEASURED_ITERS = 10;

struct BenchSize {
  const char* label;
  size_t bytes;
};

static const BenchSize BENCH_SIZES[] = {
    {"1 KB",   1u << 10}, {"2 KB",   1u << 11}, {"4 KB",   1u << 12}, {"8 KB",   1u << 13},
    {"16 KB",  1u << 14}, {"32 KB",  1u << 15}, {"64 KB",  1u << 16}, {"128 KB", 1u << 17},
    {"256 KB", 1u << 18}, {"512 KB", 1u << 19}, {"1 MB",   1u << 20}, {"2 MB",   1u << 21},
    {"4 MB",   1u << 22}, {"8 MB",   1u << 23}, {"16 MB",  1u << 24}, {"32 MB",  1u << 25},
    {"64 MB",  1u << 26}, {"128 MB", 1u << 27}, {"256 MB", 1u << 28}, {"512 MB", 1u << 29},
    {"1 GB",   1ull << 30},
};
static const int NUM_BENCH_SIZES = sizeof(BENCH_SIZES) / sizeof(BENCH_SIZES[0]);

inline size_t benchMaxBytes() {
  const char* env = std::getenv("PCCL_BENCH_MAX_BYTES");
  if (!env || env[0] == '\0')
    return BENCH_SIZES[NUM_BENCH_SIZES - 1].bytes;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed == 0)
    return BENCH_SIZES[NUM_BENCH_SIZES - 1].bytes;
  return static_cast<size_t>(parsed);
}

inline int numBenchSizes() {
  size_t max_bytes = benchMaxBytes();
  int count = 0;
  while (count < NUM_BENCH_SIZES && BENCH_SIZES[count].bytes <= max_bytes)
    ++count;
  return count > 0 ? count : 1;
}

/* ------------------------------------------------------------------ */
/*  Bus bandwidth factor (per NCCL doc)                                */
/* ------------------------------------------------------------------ */

enum CollType {
  COLL_ALLREDUCE,
  COLL_ALLGATHER,
  COLL_REDUCESCATTER,
  COLL_ALLTOALL,
};

inline double busBwFactor(CollType coll, int n) {
  double d = static_cast<double>(n);
  switch (coll) {
    case COLL_ALLREDUCE:
      return 2.0 * (d - 1.0) / d;
    case COLL_ALLGATHER:
      return (d - 1.0) / d;
    case COLL_REDUCESCATTER:
      return (d - 1.0) / d;
    case COLL_ALLTOALL:
      return (d - 1.0) / d;
  }
  return 1.0;
}

/* ------------------------------------------------------------------ */
/*  Table printing helpers                                             */
/* ------------------------------------------------------------------ */

inline void printBenchHeader(const char* title) {
  std::printf("\n=== %s | Warmup %d | Measured %d ===\n\n", title, WARMUP_ITERS, MEASURED_ITERS);
  std::printf("%-10s  %12s  %14s  %14s  %s\n", "Size", "Avg Lat(ms)", "Alg BW(GB/s)", "Bus BW(GB/s)", "Correct");
  std::printf("%-10s  %12s  %14s  %14s  %s\n", "----------", "------------", "--------------", "--------------",
              "-------");
}

inline void printBenchRow(const char* sizeLabel, double avg_ms, double alg_bw, double bus_bw, bool correct) {
  std::printf("%-10s  %12.3f  %14.2f  %14.2f  %s\n", sizeLabel, avg_ms, alg_bw, bus_bw, correct ? "PASS" : "FAIL");
}

/* ------------------------------------------------------------------ */
/*  Timing helpers using device events                                 */
/* ------------------------------------------------------------------ */

struct BenchTimer {
  devEvent start, stop;

  void init() {
    devCreateTimingEvent(&start);
    devCreateTimingEvent(&stop);
  }

  void record_start(devStream s) { devRecordEvent(start, s); }
  void record_stop(devStream s) { devRecordEvent(stop, s); }

  float elapsed_ms() {
    float ms = 0.0f;
    devEventElapsedTime(&ms, start, stop);
    return ms;
  }

  void destroy() {
    devDestroyEvent(start);
    devDestroyEvent(stop);
  }
};

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
/*  Per-iteration timing & percentile distribution                     */
/* ------------------------------------------------------------------ */

struct IterTimings {
  std::vector<double> latencies_ms;

  void reserve(int n) { latencies_ms.reserve(n); }
  void clear() { latencies_ms.clear(); }
  void add(double ms) { latencies_ms.push_back(ms); }

  double avg() const {
    double sum = 0;
    for (double v : latencies_ms)
      sum += v;
    return latencies_ms.empty() ? 0.0 : sum / static_cast<double>(latencies_ms.size());
  }

  double percentile(double p) const {
    std::vector<double> sorted = latencies_ms;
    std::sort(sorted.begin(), sorted.end());
    double idx = p / 100.0 * static_cast<double>(sorted.size() - 1);
    int lo = static_cast<int>(idx);
    int hi = lo + 1;
    if (hi >= static_cast<int>(sorted.size()))
      return sorted.back();
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
  }
};

inline void printPercentileTable(const IterTimings& timings, double dataBytes, double busFactor) {
  std::printf("    %6s  %12s  %14s  %14s\n", "%ile", "Lat(ms)", "Alg BW(GB/s)", "Bus BW(GB/s)");
  std::printf("    %6s  %12s  %14s  %14s\n", "------", "------------", "--------------", "--------------");
  for (int p = 0; p <= 100; p += 10) {
    double lat = timings.percentile(static_cast<double>(p));
    double alg_bw = (dataBytes / 1e9) / (lat / 1e3);
    double bus_bw = alg_bw * busFactor;
    std::printf("    P%-5d  %12.3f  %14.2f  %14.2f\n", p, lat, alg_bw, bus_bw);
  }
  std::printf("\n");
}

/* ------------------------------------------------------------------ */
/*  Persistent per-rank thread pool for parallel collective submit     */
/* ------------------------------------------------------------------ */

// Creates nranks persistent worker threads. Each call to run(fn) dispatches
// fn(r) on thread r and blocks until all ranks complete.
//
// Reusing the same threads across iterations keeps Perfetto submit tracks
// stable: one track per rank instead of one track per thread-creation event.
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
  size_t stride;    /* fast-path floats per segment */
  const char* label;
};

static const StrideCfg STRIDE_CONFIGS[] = {
    {32u << 20, 24u << 20, "75% fast-path"},
    {32u << 20, 16u << 20, "50% fast-path"},
    {32u << 20, 8u << 20, "25% fast-path"},
    {32u << 20, 0, "contiguous"},
};
static const int NUM_STRIDE_CONFIGS = sizeof(STRIDE_CONFIGS) / sizeof(STRIDE_CONFIGS[0]);

/* ------------------------------------------------------------------ */
/*  NUMA topology printing                                             */
/* ------------------------------------------------------------------ */

inline void printNumaTopology(int nranks) {
  pccl::LocalTopology topo = pccl::discoverLocalTopology(nranks);
  std::printf("\n--- GPU NUMA Topology (%d GPUs, %d NUMA nodes) ---\n", nranks, topo.num_numa_nodes);
  for (int gpu = 0; gpu < nranks; ++gpu) {
    int logical = topo.getNUMAForGPU(gpu);
    int physical = topo.getPhysicalNUMA(logical);
    char bus_id[32] = {};
    pccl::devGetPciBusId(gpu, bus_id, sizeof(bus_id));
    std::printf("  GPU %d  PCI %s  -> NUMA %d (physical %d)\n", gpu, bus_id, logical, physical);
  }
  std::printf("\n");
}

#endif /* PCCL_TEST_COMMON_HPP */
