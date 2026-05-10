#include <cuda_runtime.h>
#include <nccl.h>

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
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK_CUDA(call)                                                                 \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(_e));                                              \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

#define CHECK_NCCL(call)                                                                 \
  do {                                                                                   \
    ncclResult_t _r = (call);                                                            \
    if (_r != ncclSuccess) {                                                             \
      std::fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   ncclGetErrorString(_r));                                              \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

template <int Unroll>
__global__ void fma_kernel(float* out, int inner_iters, unsigned long long* sink) {
  const unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 1.0f + static_cast<float>(tid & 255) * 0.0001f;
  float y = 1.0001f;
  float z = 0.9999f;

  for (int i = 0; i < inner_iters; ++i) {
#pragma unroll
    for (int j = 0; j < Unroll; ++j) {
      x = fmaf(x, y, z);
      y = fmaf(y, z, x);
      z = fmaf(z, x, y);
      x = fmaf(x, 0.999999f, 0.000001f);
    }
  }

  out[tid] = x + y + z;
  if (tid == 0)
    atomicAdd(sink, static_cast<unsigned long long>(out[0] != 0.0f));
}

struct Args {
  std::string coll = "all";
  bool with_compute = false;
  int nranks = 4;
  size_t total_bytes = 128ull << 20;
  int warmup = 5;
  int iters = 50;
  int compute_batch_repeats = 8;
  int kernel_inner_iters = 1024;
  int blocks = 2160;
  int threads_per_block = 256;
  bool verify = false;
};

void usage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s [--coll ag|rs|ar|all] [--with-compute 0|1] [--nranks N]\n"
               "          [--total-bytes BYTES] [--warmup N] [--iters N]\n"
               "          [--compute-batch-repeats N] [--kernel-inner-iters N]\n"
               "          [--blocks N] [--threads-per-block N] [--verify 0|1]\n",
               prog);
}

size_t parseSize(const char* s) {
  char* end = nullptr;
  double value = std::strtod(s, &end);
  if (end == s || value <= 0.0)
    return 0;
  while (*end == ' ')
    ++end;

  double scale = 1.0;
  if (*end != '\0') {
    if (std::strcmp(end, "K") == 0 || std::strcmp(end, "KB") == 0 || std::strcmp(end, "KiB") == 0) {
      scale = 1024.0;
    } else if (std::strcmp(end, "M") == 0 || std::strcmp(end, "MB") == 0 || std::strcmp(end, "MiB") == 0) {
      scale = 1024.0 * 1024.0;
    } else if (std::strcmp(end, "G") == 0 || std::strcmp(end, "GB") == 0 || std::strcmp(end, "GiB") == 0) {
      scale = 1024.0 * 1024.0 * 1024.0;
    } else {
      return 0;
    }
  }
  return static_cast<size_t>(value * scale);
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto needValue = [&](const char* opt) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", opt);
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--coll") == 0) {
      args.coll = needValue(argv[i]);
    } else if (std::strcmp(argv[i], "--with-compute") == 0) {
      args.with_compute = std::atoi(needValue(argv[i])) != 0;
    } else if (std::strcmp(argv[i], "--nranks") == 0) {
      args.nranks = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--total-bytes") == 0) {
      args.total_bytes = parseSize(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      args.warmup = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      args.iters = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--compute-batch-repeats") == 0) {
      args.compute_batch_repeats = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--kernel-inner-iters") == 0) {
      args.kernel_inner_iters = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--blocks") == 0) {
      args.blocks = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--threads-per-block") == 0) {
      args.threads_per_block = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--verify") == 0) {
      args.verify = std::atoi(needValue(argv[i])) != 0;
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }

  if ((args.coll != "ag" && args.coll != "rs" && args.coll != "ar" && args.coll != "all") ||
      args.nranks <= 0 || args.total_bytes == 0 ||
      args.total_bytes % (static_cast<size_t>(args.nranks) * sizeof(float)) != 0 ||
      args.warmup < 0 || args.iters <= 0 || args.compute_batch_repeats <= 0 ||
      args.kernel_inner_iters <= 0 || args.blocks <= 0 || args.threads_per_block <= 0) {
    usage(argv[0]);
    std::exit(2);
  }
  return args;
}

struct Stats {
  double avg = 0.0;
  double min = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double max = 0.0;
};

Stats summarize(std::vector<double> values) {
  Stats s;
  if (values.empty())
    return s;
  std::sort(values.begin(), values.end());
  s.min = values.front();
  s.max = values.back();
  s.avg = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  auto pct = [&](double p) {
    double idx = p * static_cast<double>(values.size() - 1);
    int lo = static_cast<int>(idx);
    int hi = std::min(lo + 1, static_cast<int>(values.size() - 1));
    double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
  };
  s.p50 = pct(0.50);
  s.p90 = pct(0.90);
  return s;
}

enum class Coll {
  Ag,
  Rs,
  Ar,
};

const char* collName(Coll coll) {
  switch (coll) {
    case Coll::Ag:
      return "AG";
    case Coll::Rs:
      return "RS";
    case Coll::Ar:
      return "AR";
  }
  return "unknown";
}

double busFactor(Coll coll, int nranks) {
  double n = static_cast<double>(nranks);
  switch (coll) {
    case Coll::Ag:
    case Coll::Rs:
      return (n - 1.0) / n;
    case Coll::Ar:
      return 2.0 * (n - 1.0) / n;
  }
  return 1.0;
}

std::vector<Coll> selectedColls(const std::string& name) {
  if (name == "ag")
    return {Coll::Ag};
  if (name == "rs")
    return {Coll::Rs};
  if (name == "ar")
    return {Coll::Ar};
  return {Coll::Ag, Coll::Rs, Coll::Ar};
}

struct Resources {
  int nranks = 0;
  size_t total_bytes = 0;
  size_t total_count = 0;
  size_t per_rank_count = 0;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> comm_streams;
  std::vector<cudaStream_t> compute_streams;
  std::vector<float*> send;
  std::vector<float*> recv;
  std::vector<float*> compute_out;
  std::vector<unsigned long long*> sink;
};

void fillBuffer(float* ptr, size_t count, float value, cudaStream_t stream) {
  std::vector<float> host(count, value);
  CHECK_CUDA(cudaMemcpyAsync(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void allocateResources(const Args& args, Resources* res) {
  res->nranks = args.nranks;
  res->total_bytes = args.total_bytes;
  res->total_count = args.total_bytes / sizeof(float);
  res->per_rank_count = res->total_count / static_cast<size_t>(args.nranks);

  res->comms.assign(args.nranks, nullptr);
  res->comm_streams.resize(args.nranks);
  res->compute_streams.resize(args.nranks);
  res->send.resize(args.nranks);
  res->recv.resize(args.nranks);
  res->compute_out.resize(args.nranks);
  res->sink.resize(args.nranks);

  int devlist[64];
  if (args.nranks > static_cast<int>(sizeof(devlist) / sizeof(devlist[0]))) {
    std::fprintf(stderr, "too many ranks\n");
    std::exit(2);
  }
  for (int r = 0; r < args.nranks; ++r)
    devlist[r] = r;
  CHECK_NCCL(ncclCommInitAll(res->comms.data(), args.nranks, devlist));

  const size_t compute_elems = static_cast<size_t>(args.blocks) * static_cast<size_t>(args.threads_per_block);
  for (int r = 0; r < args.nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamCreateWithFlags(&res->comm_streams[r], cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&res->compute_streams[r], cudaStreamNonBlocking));
    CHECK_CUDA(cudaMalloc(&res->send[r], args.total_bytes));
    CHECK_CUDA(cudaMalloc(&res->recv[r], args.total_bytes));
    CHECK_CUDA(cudaMalloc(&res->compute_out[r], compute_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&res->sink[r], sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemsetAsync(res->recv[r], 0, args.total_bytes, res->comm_streams[r]));
    CHECK_CUDA(cudaMemsetAsync(res->sink[r], 0, sizeof(unsigned long long), res->compute_streams[r]));
    fillBuffer(res->send[r], res->total_count, static_cast<float>(r + 1), res->comm_streams[r]);
    CHECK_CUDA(cudaStreamSynchronize(res->comm_streams[r]));
    CHECK_CUDA(cudaStreamSynchronize(res->compute_streams[r]));
  }
}

void destroyResources(Resources* res) {
  for (int r = 0; r < res->nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  for (int r = 0; r < res->nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaFree(res->sink[r]));
    CHECK_CUDA(cudaFree(res->compute_out[r]));
    CHECK_CUDA(cudaFree(res->recv[r]));
    CHECK_CUDA(cudaFree(res->send[r]));
    CHECK_CUDA(cudaStreamDestroy(res->compute_streams[r]));
    CHECK_CUDA(cudaStreamDestroy(res->comm_streams[r]));
    CHECK_NCCL(ncclCommDestroy(res->comms[r]));
  }
}

void submitCollective(Coll coll, Resources* res) {
  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < res->nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    switch (coll) {
      case Coll::Ag:
        CHECK_NCCL(ncclAllGather(res->send[r], res->recv[r], res->per_rank_count, ncclFloat32, res->comms[r],
                                 res->comm_streams[r]));
        break;
      case Coll::Rs:
        CHECK_NCCL(ncclReduceScatter(res->send[r], res->recv[r], res->per_rank_count, ncclFloat32, ncclSum,
                                     res->comms[r], res->comm_streams[r]));
        break;
      case Coll::Ar:
        CHECK_NCCL(ncclAllReduce(res->send[r], res->recv[r], res->total_count, ncclFloat32, ncclSum,
                                 res->comms[r], res->comm_streams[r]));
        break;
    }
  }
  CHECK_NCCL(ncclGroupEnd());
}

void syncComm(Resources* res) {
  for (int r = 0; r < res->nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(res->comm_streams[r]));
  }
}

void launchComputeBatch(const Args& args, Resources* res) {
  for (int repeat = 0; repeat < args.compute_batch_repeats; ++repeat) {
    for (int r = 0; r < res->nranks; ++r) {
      CHECK_CUDA(cudaSetDevice(r));
      fma_kernel<8><<<args.blocks, args.threads_per_block, 0, res->compute_streams[r]>>>(
          res->compute_out[r], args.kernel_inner_iters, res->sink[r]);
      CHECK_CUDA(cudaGetLastError());
    }
  }
}

void syncCompute(Resources* res) {
  for (int r = 0; r < res->nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(res->compute_streams[r]));
  }
}

class ComputePressure {
 public:
  ComputePressure(const Args& args, Resources* res) : args_(args), res_(res), stop_(false), batches_(0) {}

  void start() {
    stop_.store(false, std::memory_order_release);
    thread_ = std::thread([this] {
      while (!stop_.load(std::memory_order_acquire)) {
        launchComputeBatch(args_, res_);
        syncCompute(res_);
        batches_.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  void stop() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable())
      thread_.join();
    syncCompute(res_);
  }

  int batches() const { return batches_.load(std::memory_order_relaxed); }

 private:
  const Args& args_;
  Resources* res_;
  std::atomic<bool> stop_;
  std::atomic<int> batches_;
  std::thread thread_;
};

bool verifyCollective(Coll coll, Resources* res) {
  const float sum_expected = static_cast<float>(res->nranks) * static_cast<float>(res->nranks + 1) / 2.0f;
  for (int r = 0; r < res->nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    size_t count = (coll == Coll::Rs) ? res->per_rank_count : res->total_count;
    std::vector<float> host(count);
    CHECK_CUDA(cudaMemcpyAsync(host.data(), res->recv[r], count * sizeof(float), cudaMemcpyDeviceToHost,
                               res->comm_streams[r]));
    CHECK_CUDA(cudaStreamSynchronize(res->comm_streams[r]));

    if (coll == Coll::Ag) {
      for (int src = 0; src < res->nranks; ++src) {
        float expected = static_cast<float>(src + 1);
        const float* seg = host.data() + static_cast<size_t>(src) * res->per_rank_count;
        for (size_t i = 0; i < res->per_rank_count; i += std::max<size_t>(1, res->per_rank_count / 16)) {
          if (std::fabs(seg[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "verify failed coll=AG rank=%d src=%d idx=%zu got=%f expected=%f\n",
                         r, src, i, seg[i], expected);
            return false;
          }
        }
      }
    } else {
      for (size_t i = 0; i < count; i += std::max<size_t>(1, count / 16)) {
        if (std::fabs(host[i] - sum_expected) > 1e-5f) {
          std::fprintf(stderr, "verify failed coll=%s rank=%d idx=%zu got=%f expected=%f\n",
                       collName(coll), r, i, host[i], sum_expected);
          return false;
        }
      }
    }
  }
  return true;
}

struct Result {
  Coll coll;
  bool with_compute;
  Stats stats;
  double algbw = 0.0;
  double busbw = 0.0;
  int compute_batches = 0;
};

Result runOne(Coll coll, const Args& args, Resources* res) {
  ComputePressure pressure(args, res);
  if (args.with_compute) {
    pressure.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  for (int i = 0; i < args.warmup; ++i) {
    submitCollective(coll, res);
    syncComm(res);
  }

  std::vector<double> times;
  times.reserve(args.iters);
  for (int i = 0; i < args.iters; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    submitCollective(coll, res);
    syncComm(res);
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  if (args.with_compute)
    pressure.stop();

  if (args.verify && !verifyCollective(coll, res))
    std::exit(1);

  Stats stats = summarize(times);
  double algbw = (static_cast<double>(args.total_bytes) / 1e9) / (stats.avg / 1e3);
  double busbw = algbw * busFactor(coll, args.nranks);
  return {coll, args.with_compute, stats, algbw, busbw, pressure.batches()};
}

void printConfig(const Args& args) {
  int version = 0;
  CHECK_NCCL(ncclGetVersion(&version));
  std::printf("nccl_version=%d\n", version);
  std::printf("coll=%s\n", args.coll.c_str());
  std::printf("with_compute=%d\n", args.with_compute ? 1 : 0);
  std::printf("nranks=%d\n", args.nranks);
  std::printf("total_bytes=%zu\n", args.total_bytes);
  std::printf("warmup=%d\n", args.warmup);
  std::printf("iters=%d\n", args.iters);
  std::printf("compute_batch_repeats=%d\n", args.compute_batch_repeats);
  std::printf("kernel_inner_iters=%d\n", args.kernel_inner_iters);
  std::printf("blocks=%d\n", args.blocks);
  std::printf("threads_per_block=%d\n", args.threads_per_block);
}

void printResult(const Result& result) {
  std::printf("RESULT coll=%s with_compute=%d avg_ms=%.6f p50_ms=%.6f p90_ms=%.6f min_ms=%.6f max_ms=%.6f "
              "algbw_GBs=%.6f busbw_GBs=%.6f compute_batches=%d\n",
              collName(result.coll),
              result.with_compute ? 1 : 0,
              result.stats.avg,
              result.stats.p50,
              result.stats.p90,
              result.stats.min,
              result.stats.max,
              result.algbw,
              result.busbw,
              result.compute_batches);
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parseArgs(argc, argv);

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < args.nranks) {
    std::fprintf(stderr, "need %d CUDA devices, found %d\n", args.nranks, device_count);
    return 1;
  }

  printConfig(args);
  Resources res;
  allocateResources(args, &res);
  for (Coll coll : selectedColls(args.coll)) {
    Result result = runOne(coll, args, &res);
    printResult(result);
  }
  destroyResources(&res);
  return 0;
}
