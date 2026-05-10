/*
 * Single-process multi-thread HCCL compatibility benchmark.
 *
 * The binary links only against CANN HCCL/ACL. Run it directly for native HCCL,
 * or preload libpccl.so to test PCIe CCL through the same HCCL API surface.
 *
 * Usage:
 *   bench_hccl_compat_sp <nranks> [begin_bytes] [end_bytes] [iters] [warmup_iters]
 *       [--init=root|all] [--rootinfo=zero|hccl] [--devices=0,1,...]
 *
 * Defaults:
 * - --init=all
 * - --rootinfo=zero
 *
 * data_size follows hccl_test convention:
 * - AllReduce: per-rank input/output bytes.
 * - AllGather: per-rank output bytes, send bytes are data_size / nranks.
 * - ReduceScatter: per-rank input bytes, recv bytes are data_size / nranks.
 */

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <hccl/hccl.h>

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
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

enum class Collective {
  AllReduce,
  AllGather,
  ReduceScatter,
};

enum class RootInfoMode {
  Zero,
  Hccl,
};

enum class InitMode {
  Root,
  All,
};

struct Options {
  int nranks = 0;
  size_t begin_bytes = 16ull << 20;
  size_t end_bytes = 512ull << 20;
  int iters = 10;
  int warmup_iters = 2;
  InitMode init = InitMode::All;
  RootInfoMode rootinfo = RootInfoMode::Zero;
  std::vector<int> devices;
};

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
    for (auto& t : threads_)
      t.join();
  }

  RankThreadPool(const RankThreadPool&) = delete;
  RankThreadPool& operator=(const RankThreadPool&) = delete;

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
        if (stop_)
          break;
        seen_gen = generation_;
      }
      fn_(r);
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

[[noreturn]] void die(const char* kind, const char* call, long long code,
                      const char* file, int line) {
  std::fprintf(stderr, "%s error at %s:%d: %s returned %lld\n",
               kind, file, line, call, code);
  std::fflush(stderr);
  std::exit(1);
}

#define CHECK_ACL(call)                                                        \
  do {                                                                         \
    aclError _e = (call);                                                      \
    if (_e != ACL_SUCCESS)                                                     \
      die("ACL", #call, static_cast<long long>(_e), __FILE__, __LINE__);       \
  } while (0)

#define CHECK_HCCL(call)                                                       \
  do {                                                                         \
    HcclResult _r = (call);                                                    \
    if (_r != HCCL_SUCCESS)                                                    \
      die("HCCL", #call, static_cast<long long>(_r), __FILE__, __LINE__);      \
  } while (0)

size_t parseSize(const char* s, size_t fallback) {
  if (!s || !*s)
    return fallback;
  char* end = nullptr;
  unsigned long long value = std::strtoull(s, &end, 10);
  if (end == s || value == 0)
    return fallback;
  size_t scale = 1;
  if (*end) {
    if ((end[0] == 'K' || end[0] == 'k') && end[1] == '\0')
      scale = 1ull << 10;
    else if ((end[0] == 'M' || end[0] == 'm') && end[1] == '\0')
      scale = 1ull << 20;
    else if ((end[0] == 'G' || end[0] == 'g') && end[1] == '\0')
      scale = 1ull << 30;
    else
      return fallback;
  }
  return static_cast<size_t>(value) * scale;
}

bool parseDevices(const char* spec, std::vector<int>* out) {
  if (!spec || !out)
    return false;
  out->clear();
  const char* p = spec;
  while (*p) {
    char* end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0)
      return false;
    out->push_back(static_cast<int>(v));
    if (*end == '\0')
      return true;
    if (*end != ',')
      return false;
    p = end + 1;
  }
  return !out->empty();
}

const char* collectiveName(Collective c) {
  switch (c) {
    case Collective::AllReduce:
      return "all_reduce";
    case Collective::AllGather:
      return "all_gather";
    case Collective::ReduceScatter:
      return "reduce_scatter";
  }
  return "unknown";
}

const char* rootInfoModeName(RootInfoMode mode) {
  return mode == RootInfoMode::Zero ? "zero" : "hccl";
}

const char* initModeName(InitMode mode) {
  return mode == InitMode::Root ? "root" : "all";
}

double busFactor(Collective c, int nranks) {
  double n = static_cast<double>(nranks);
  if (c == Collective::AllReduce)
    return 2.0 * (n - 1.0) / n;
  return (n - 1.0) / n;
}

void printUsage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s <nranks> [begin_bytes] [end_bytes] [iters] [warmup_iters] "
               "[--init=root|all] [--rootinfo=zero|hccl] [--devices=0,1,...]\n"
               "Defaults: --init=all --rootinfo=zero\n",
               prog);
}

bool parseArgs(int argc, char* argv[], Options* opts) {
  if (!opts || argc < 2)
    return false;

  std::vector<const char*> positional;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strncmp(arg, "--init=", 7) == 0) {
      const char* mode = arg + 7;
      if (std::strcmp(mode, "root") == 0) {
        opts->init = InitMode::Root;
      } else if (std::strcmp(mode, "all") == 0) {
        opts->init = InitMode::All;
      } else {
        std::fprintf(stderr, "unknown init mode: %s\n", mode);
        return false;
      }
    } else if (std::strncmp(arg, "--rootinfo=", 11) == 0) {
      const char* mode = arg + 11;
      if (std::strcmp(mode, "zero") == 0) {
        opts->rootinfo = RootInfoMode::Zero;
      } else if (std::strcmp(mode, "hccl") == 0) {
        opts->rootinfo = RootInfoMode::Hccl;
      } else {
        std::fprintf(stderr, "unknown rootinfo mode: %s\n", mode);
        return false;
      }
    } else if (std::strncmp(arg, "--devices=", 10) == 0) {
      if (!parseDevices(arg + 10, &opts->devices)) {
        std::fprintf(stderr, "invalid devices list: %s\n", arg + 10);
        return false;
      }
    } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      return false;
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.empty())
    return false;
  opts->nranks = std::atoi(positional[0]);
  if (opts->nranks <= 0) {
    std::fprintf(stderr, "nranks must be positive\n");
    return false;
  }
  if (positional.size() >= 2)
    opts->begin_bytes = parseSize(positional[1], opts->begin_bytes);
  if (positional.size() >= 3)
    opts->end_bytes = parseSize(positional[2], opts->end_bytes);
  if (positional.size() >= 4)
    opts->iters = std::max(1, std::atoi(positional[3]));
  if (positional.size() >= 5)
    opts->warmup_iters = std::max(0, std::atoi(positional[4]));
  if (positional.size() > 5) {
    std::fprintf(stderr, "too many positional arguments\n");
    return false;
  }

  if (opts->devices.empty()) {
    opts->devices.resize(opts->nranks);
    for (int r = 0; r < opts->nranks; ++r)
      opts->devices[r] = r;
  }
  if (static_cast<int>(opts->devices.size()) != opts->nranks) {
    std::fprintf(stderr, "devices count must match nranks\n");
    return false;
  }
  if (opts->end_bytes < opts->begin_bytes) {
    std::fprintf(stderr, "end_bytes must be >= begin_bytes\n");
    return false;
  }
  return true;
}

void printOptions(const Options& opts) {
  std::printf("HCCL compatibility SP benchmark\n");
  std::printf("nranks=%d begin_bytes=%zu end_bytes=%zu iters=%d warmup_iters=%d init=%s rootinfo=%s devices=",
              opts.nranks, opts.begin_bytes, opts.end_bytes, opts.iters,
              opts.warmup_iters, initModeName(opts.init), rootInfoModeName(opts.rootinfo));
  for (size_t i = 0; i < opts.devices.size(); ++i)
    std::printf("%s%d", i == 0 ? "" : ",", opts.devices[i]);
  std::printf("\n");
}

void fillSendBuffers(Collective c, const Options& opts, size_t data_bytes,
                     const std::vector<void*>& d_send,
                     const std::vector<aclrtStream>& streams) {
  size_t count = data_bytes / sizeof(float);
  if (c == Collective::AllGather)
    count /= static_cast<size_t>(opts.nranks);
  size_t send_bytes = count * sizeof(float);

  for (int r = 0; r < opts.nranks; ++r) {
    CHECK_ACL(aclrtSetDevice(opts.devices[r]));
    std::vector<float> host(count, static_cast<float>(r + 1));
    CHECK_ACL(aclrtMemcpyAsync(d_send[r], send_bytes, host.data(), send_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE, streams[r]));
    CHECK_ACL(aclrtSynchronizeStream(streams[r]));
  }
}

HcclResult submitOne(Collective c, int rank, int nranks, size_t data_bytes,
                     const std::vector<void*>& d_send,
                     const std::vector<void*>& d_recv,
                     const std::vector<HcclComm>& comms,
                     const std::vector<aclrtStream>& streams) {
  uint64_t count = static_cast<uint64_t>(data_bytes / sizeof(float));
  switch (c) {
    case Collective::AllReduce:
      return HcclAllReduce(d_send[rank], d_recv[rank], count,
                           HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                           comms[rank], streams[rank]);
    case Collective::AllGather:
      return HcclAllGather(d_send[rank], d_recv[rank],
                           count / static_cast<uint64_t>(nranks),
                           HCCL_DATA_TYPE_FP32, comms[rank], streams[rank]);
    case Collective::ReduceScatter:
      return HcclReduceScatter(d_send[rank], d_recv[rank],
                               count / static_cast<uint64_t>(nranks),
                               HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                               comms[rank], streams[rank]);
  }
  return HCCL_E_INTERNAL;
}

void syncStreams(const Options& opts, const std::vector<aclrtStream>& streams,
                 RankThreadPool& pool) {
  pool.run([&](int r) {
    CHECK_ACL(aclrtSetDevice(opts.devices[r]));
    CHECK_ACL(aclrtSynchronizeStream(streams[r]));
  });
}

bool verifyBuffer(const float* buf, size_t count, float expected, int rank,
                  const char* tag) {
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(buf[i] - expected) > 1e-5f) {
      std::fprintf(stderr, "FAIL rank %d %s: buf[%zu]=%f expected=%f\n",
                   rank, tag, i, buf[i], expected);
      return false;
    }
  }
  return true;
}

bool verify(Collective c, const Options& opts, size_t data_bytes,
            const std::vector<void*>& d_recv,
            const std::vector<aclrtStream>& streams) {
  size_t count = data_bytes / sizeof(float);
  float reduced = static_cast<float>(opts.nranks) *
                  static_cast<float>(opts.nranks + 1) / 2.0f;

  for (int r = 0; r < opts.nranks; ++r) {
    CHECK_ACL(aclrtSetDevice(opts.devices[r]));
    if (c == Collective::AllGather) {
      size_t send_count = count / static_cast<size_t>(opts.nranks);
      std::vector<float> host(count);
      CHECK_ACL(aclrtMemcpyAsync(host.data(), data_bytes, d_recv[r], data_bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST, streams[r]));
      CHECK_ACL(aclrtSynchronizeStream(streams[r]));
      for (int src = 0; src < opts.nranks; ++src) {
        if (!verifyBuffer(host.data() + static_cast<size_t>(src) * send_count,
                          send_count, static_cast<float>(src + 1), r,
                          "all_gather"))
          return false;
      }
    } else if (c == Collective::AllReduce) {
      std::vector<float> host(count);
      CHECK_ACL(aclrtMemcpyAsync(host.data(), data_bytes, d_recv[r], data_bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST, streams[r]));
      CHECK_ACL(aclrtSynchronizeStream(streams[r]));
      if (!verifyBuffer(host.data(), count, reduced, r, "all_reduce"))
        return false;
    } else {
      size_t recv_count = count / static_cast<size_t>(opts.nranks);
      size_t recv_bytes = recv_count * sizeof(float);
      std::vector<float> host(recv_count);
      CHECK_ACL(aclrtMemcpyAsync(host.data(), recv_bytes, d_recv[r], recv_bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST, streams[r]));
      CHECK_ACL(aclrtSynchronizeStream(streams[r]));
      if (!verifyBuffer(host.data(), recv_count, reduced, r, "reduce_scatter"))
        return false;
    }
  }
  return true;
}

void runCollective(Collective c, const Options& opts,
                   const std::vector<void*>& d_send,
                   const std::vector<void*>& d_recv,
                   const std::vector<HcclComm>& comms,
                   const std::vector<aclrtStream>& streams,
                   RankThreadPool& pool) {
  std::printf("=== %s_sp ===\n", collectiveName(c));
  std::printf("the minbytes is %zu, maxbytes is %zu, iters is %d, warmup_iters is %d\n",
              opts.begin_bytes, opts.end_bytes, opts.iters, opts.warmup_iters);
  std::printf("data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | bus_bandwidth(GB/s): | check_result:\n");

  double factor = busFactor(c, opts.nranks);
  for (size_t bytes = opts.begin_bytes; bytes <= opts.end_bytes; bytes <<= 1) {
    if (bytes % (static_cast<size_t>(opts.nranks) * sizeof(float)) != 0)
      continue;

    fillSendBuffers(c, opts, bytes, d_send, streams);
    for (int i = 0; i < opts.warmup_iters; ++i) {
      pool.run([&](int r) {
        CHECK_ACL(aclrtSetDevice(opts.devices[r]));
        CHECK_HCCL(submitOne(c, r, opts.nranks, bytes, d_send, d_recv, comms, streams));
      });
      syncStreams(opts, streams, pool);
    }

    double total_us = 0.0;
    for (int i = 0; i < opts.iters; ++i) {
      syncStreams(opts, streams, pool);
      auto t0 = std::chrono::steady_clock::now();
      pool.run([&](int r) {
        CHECK_ACL(aclrtSetDevice(opts.devices[r]));
        CHECK_HCCL(submitOne(c, r, opts.nranks, bytes, d_send, d_recv, comms, streams));
      });
      syncStreams(opts, streams, pool);
      auto t1 = std::chrono::steady_clock::now();
      total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double avg_us = total_us / static_cast<double>(opts.iters);
    double alg_bw = static_cast<double>(bytes) / avg_us / 1000.0;
    double bus_bw = alg_bw * factor;
    bool ok = verify(c, opts, bytes, d_recv, streams);
    std::printf("%-18zu | %-14.2f | %-20.5f | %-20.5f | %s\n",
                bytes, avg_us, alg_bw, bus_bw, ok ? "success" : "failed");
    std::fflush(stdout);
    if (!ok)
      std::exit(1);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Options opts;
  if (!parseArgs(argc, argv, &opts)) {
    printUsage(argv[0]);
    return 1;
  }

  CHECK_ACL(aclInit(nullptr));
  printOptions(opts);

  std::vector<HcclComm> comms(opts.nranks, nullptr);
  std::vector<aclrtStream> streams(opts.nranks, nullptr);
  std::vector<void*> d_send(opts.nranks, nullptr);
  std::vector<void*> d_recv(opts.nranks, nullptr);

  {
    RankThreadPool pool(opts.nranks);
    if (opts.init == InitMode::All) {
      for (int r = 0; r < opts.nranks; ++r)
        CHECK_ACL(aclrtSetDevice(opts.devices[r]));
      std::vector<int32_t> devices(opts.devices.begin(), opts.devices.end());
      CHECK_HCCL(HcclCommInitAll(static_cast<uint32_t>(opts.nranks),
                                 devices.data(), comms.data()));
      pool.run([&](int r) {
        CHECK_ACL(aclrtSetDevice(opts.devices[r]));
        CHECK_ACL(aclrtCreateStream(&streams[r]));
        CHECK_ACL(aclrtMalloc(&d_send[r], opts.end_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&d_recv[r], opts.end_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
      });
    } else {
      HcclRootInfo root_info;
      std::memset(&root_info, 0, sizeof(root_info));
      if (opts.rootinfo == RootInfoMode::Hccl)
        CHECK_HCCL(HcclGetRootInfo(&root_info));

      pool.run([&](int r) {
        CHECK_ACL(aclrtSetDevice(opts.devices[r]));
        CHECK_ACL(aclrtCreateStream(&streams[r]));
        CHECK_ACL(aclrtMalloc(&d_send[r], opts.end_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&d_recv[r], opts.end_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_HCCL(HcclCommInitRootInfo(static_cast<uint32_t>(opts.nranks),
                                        &root_info,
                                        static_cast<uint32_t>(r),
                                        &comms[r]));
      });
    }

    runCollective(Collective::AllReduce, opts, d_send, d_recv, comms, streams, pool);
    runCollective(Collective::AllGather, opts, d_send, d_recv, comms, streams, pool);
    runCollective(Collective::ReduceScatter, opts, d_send, d_recv, comms, streams, pool);

    pool.run([&](int r) {
      CHECK_ACL(aclrtSetDevice(opts.devices[r]));
      if (streams[r])
        CHECK_ACL(aclrtSynchronizeStream(streams[r]));
      if (comms[r])
        CHECK_HCCL(HcclCommDestroy(comms[r]));
      if (d_send[r])
        CHECK_ACL(aclrtFree(d_send[r]));
      if (d_recv[r])
        CHECK_ACL(aclrtFree(d_recv[r]));
      if (streams[r])
        CHECK_ACL(aclrtDestroyStream(streams[r]));
    });
  }

  CHECK_ACL(aclFinalize());
  return 0;
}
