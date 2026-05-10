#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK_CUDA(call)                                                                 \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error at %s:%d rank=%d: %s\n", __FILE__, __LINE__,      \
                   g_rank, cudaGetErrorString(_e));                                      \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
    }                                                                                    \
  } while (0)

#define CHECK_NCCL(call)                                                                 \
  do {                                                                                   \
    ncclResult_t _r = (call);                                                            \
    if (_r != ncclSuccess) {                                                             \
      std::fprintf(stderr, "NCCL error at %s:%d rank=%d: %s\n", __FILE__, __LINE__,      \
                   g_rank, ncclGetErrorString(_r));                                      \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
    }                                                                                    \
  } while (0)

#define CHECK_MPI(call)                                                                  \
  do {                                                                                   \
    int _r = (call);                                                                     \
    if (_r != MPI_SUCCESS) {                                                             \
      std::fprintf(stderr, "MPI error at %s:%d rank=%d code=%d\n", __FILE__, __LINE__,   \
                   g_rank, _r);                                                          \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
    }                                                                                    \
  } while (0)

int g_rank = -1;

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
  size_t total_bytes = 128ull << 20;
  int warmup = 5;
  int iters = 30;
  int compute_batch_repeats = 8;
  int kernel_inner_iters = 1024;
  int blocks = 2160;
  int threads_per_block = 256;
  bool verify = false;
};

void usage(const char* prog) {
  if (g_rank == 0) {
    std::fprintf(stderr,
                 "Usage: %s [--coll ag|rs|ar|all] [--with-compute 0|1]\n"
                 "          [--total-bytes BYTES] [--warmup N] [--iters N]\n"
                 "          [--compute-batch-repeats N] [--kernel-inner-iters N]\n"
                 "          [--blocks N] [--threads-per-block N] [--verify 0|1]\n",
                 prog);
  }
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
        if (g_rank == 0)
          std::fprintf(stderr, "%s needs a value\n", opt);
        usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--coll") == 0) {
      args.coll = needValue(argv[i]);
    } else if (std::strcmp(argv[i], "--with-compute") == 0) {
      args.with_compute = std::atoi(needValue(argv[i])) != 0;
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
      MPI_Finalize();
      std::exit(0);
    } else {
      if (g_rank == 0)
        std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
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

class ComputePressure {
 public:
  ComputePressure(const Args& args, int device, cudaStream_t stream, float* out, unsigned long long* sink)
      : args_(args), device_(device), stream_(stream), out_(out), sink_(sink), stop_(false), batches_(0) {}

  void start() {
    stop_.store(false, std::memory_order_release);
    thread_ = std::thread([this] {
      CHECK_CUDA(cudaSetDevice(device_));
      while (!stop_.load(std::memory_order_acquire)) {
        for (int repeat = 0; repeat < args_.compute_batch_repeats; ++repeat) {
          fma_kernel<8><<<args_.blocks, args_.threads_per_block, 0, stream_>>>(
              out_, args_.kernel_inner_iters, sink_);
          CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        batches_.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  void stop() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable())
      thread_.join();
    CHECK_CUDA(cudaStreamSynchronize(stream_));
  }

  int batches() const { return batches_.load(std::memory_order_relaxed); }

 private:
  const Args& args_;
  int device_;
  cudaStream_t stream_;
  float* out_;
  unsigned long long* sink_;
  std::atomic<bool> stop_;
  std::atomic<int> batches_;
  std::thread thread_;
};

void fillBuffer(float* ptr, size_t count, float value, cudaStream_t stream) {
  std::vector<float> host(count, value);
  CHECK_CUDA(cudaMemcpyAsync(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void submitCollective(Coll coll, ncclComm_t comm, cudaStream_t stream, float* send, float* recv,
                      size_t total_count, size_t per_rank_count) {
  switch (coll) {
    case Coll::Ag:
      CHECK_NCCL(ncclAllGather(send, recv, per_rank_count, ncclFloat32, comm, stream));
      break;
    case Coll::Rs:
      CHECK_NCCL(ncclReduceScatter(send, recv, per_rank_count, ncclFloat32, ncclSum, comm, stream));
      break;
    case Coll::Ar:
      CHECK_NCCL(ncclAllReduce(send, recv, total_count, ncclFloat32, ncclSum, comm, stream));
      break;
  }
}

bool verifyCollective(Coll coll, int rank, int nranks, cudaStream_t stream, float* recv,
                      size_t total_count, size_t per_rank_count) {
  const float sum_expected = static_cast<float>(nranks) * static_cast<float>(nranks + 1) / 2.0f;
  size_t count = (coll == Coll::Rs) ? per_rank_count : total_count;
  std::vector<float> host(count);
  CHECK_CUDA(cudaMemcpyAsync(host.data(), recv, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (coll == Coll::Ag) {
    for (int src = 0; src < nranks; ++src) {
      float expected = static_cast<float>(src + 1);
      const float* seg = host.data() + static_cast<size_t>(src) * per_rank_count;
      for (size_t i = 0; i < per_rank_count; i += std::max<size_t>(1, per_rank_count / 16)) {
        if (std::fabs(seg[i] - expected) > 1e-5f) {
          std::fprintf(stderr, "verify failed coll=AG rank=%d src=%d idx=%zu got=%f expected=%f\n",
                       rank, src, i, seg[i], expected);
          return false;
        }
      }
    }
  } else {
    for (size_t i = 0; i < count; i += std::max<size_t>(1, count / 16)) {
      if (std::fabs(host[i] - sum_expected) > 1e-5f) {
        std::fprintf(stderr, "verify failed coll=%s rank=%d idx=%zu got=%f expected=%f\n",
                     collName(coll), rank, i, host[i], sum_expected);
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  int rank = 0;
  int nranks = 0;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  g_rank = rank;

  Args args = parseArgs(argc, argv);
  if ((args.coll != "ag" && args.coll != "rs" && args.coll != "ar" && args.coll != "all") ||
      args.total_bytes == 0 ||
      args.total_bytes % (static_cast<size_t>(nranks) * sizeof(float)) != 0 ||
      args.warmup < 0 || args.iters <= 0 || args.compute_batch_repeats <= 0 ||
      args.kernel_inner_iters <= 0 || args.blocks <= 0 || args.threads_per_block <= 0) {
    usage(argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < nranks) {
    if (rank == 0)
      std::fprintf(stderr, "need %d visible CUDA devices, found %d\n", nranks, device_count);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CHECK_CUDA(cudaSetDevice(rank));

  int version = 0;
  CHECK_NCCL(ncclGetVersion(&version));
  if (rank == 0) {
    std::printf("nccl_version=%d\n", version);
    std::printf("coll=%s\n", args.coll.c_str());
    std::printf("with_compute=%d\n", args.with_compute ? 1 : 0);
    std::printf("nranks=%d\n", nranks);
    std::printf("total_bytes=%zu\n", args.total_bytes);
    std::printf("warmup=%d\n", args.warmup);
    std::printf("iters=%d\n", args.iters);
    std::printf("compute_batch_repeats=%d\n", args.compute_batch_repeats);
    std::printf("kernel_inner_iters=%d\n", args.kernel_inner_iters);
    std::printf("blocks=%d\n", args.blocks);
    std::printf("threads_per_block=%d\n", args.threads_per_block);
  }

  ncclUniqueId id;
  if (rank == 0)
    CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm = nullptr;
  CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));

  cudaStream_t comm_stream = nullptr;
  cudaStream_t compute_stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

  size_t total_count = args.total_bytes / sizeof(float);
  size_t per_rank_count = total_count / static_cast<size_t>(nranks);
  float* send = nullptr;
  float* recv = nullptr;
  float* compute_out = nullptr;
  unsigned long long* sink = nullptr;
  size_t compute_elems = static_cast<size_t>(args.blocks) * static_cast<size_t>(args.threads_per_block);
  CHECK_CUDA(cudaMalloc(&send, args.total_bytes));
  CHECK_CUDA(cudaMalloc(&recv, args.total_bytes));
  CHECK_CUDA(cudaMalloc(&compute_out, compute_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&sink, sizeof(unsigned long long)));
  CHECK_CUDA(cudaMemsetAsync(recv, 0, args.total_bytes, comm_stream));
  CHECK_CUDA(cudaMemsetAsync(sink, 0, sizeof(unsigned long long), compute_stream));
  fillBuffer(send, total_count, static_cast<float>(rank + 1), comm_stream);
  CHECK_CUDA(cudaStreamSynchronize(comm_stream));
  CHECK_CUDA(cudaStreamSynchronize(compute_stream));

  for (Coll coll : selectedColls(args.coll)) {
    ComputePressure pressure(args, rank, compute_stream, compute_out, sink);
    if (args.with_compute) {
      pressure.start();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    for (int i = 0; i < args.warmup; ++i) {
      CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
      submitCollective(coll, comm, comm_stream, send, recv, total_count, per_rank_count);
      CHECK_CUDA(cudaStreamSynchronize(comm_stream));
      CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    }

    std::vector<double> rank0_times;
    if (rank == 0)
      rank0_times.reserve(args.iters);
    for (int i = 0; i < args.iters; ++i) {
      CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
      double t0 = MPI_Wtime();
      submitCollective(coll, comm, comm_stream, send, recv, total_count, per_rank_count);
      CHECK_CUDA(cudaStreamSynchronize(comm_stream));
      double local_ms = (MPI_Wtime() - t0) * 1000.0;
      double max_ms = 0.0;
      CHECK_MPI(MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
      CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
      if (rank == 0)
        rank0_times.push_back(max_ms);
    }

    if (args.with_compute)
      pressure.stop();

    int local_batches = args.with_compute ? pressure.batches() : 0;
    int max_batches = 0;
    CHECK_MPI(MPI_Reduce(&local_batches, &max_batches, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD));

    int local_ok = 1;
    if (args.verify)
      local_ok = verifyCollective(coll, rank, nranks, comm_stream, recv, total_count, per_rank_count) ? 1 : 0;
    int global_ok = 0;
    CHECK_MPI(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (global_ok != 1)
      MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank == 0) {
      Stats stats = summarize(rank0_times);
      double algbw = (static_cast<double>(args.total_bytes) / 1e9) / (stats.avg / 1e3);
      double busbw = algbw * busFactor(coll, nranks);
      std::printf("RESULT coll=%s with_compute=%d avg_ms=%.6f p50_ms=%.6f p90_ms=%.6f min_ms=%.6f max_ms=%.6f "
                  "algbw_GBs=%.6f busbw_GBs=%.6f compute_batches=%d\n",
                  collName(coll),
                  args.with_compute ? 1 : 0,
                  stats.avg,
                  stats.p50,
                  stats.p90,
                  stats.min,
                  stats.max,
                  algbw,
                  busbw,
                  max_batches);
      std::fflush(stdout);
    }
  }

  CHECK_CUDA(cudaFree(sink));
  CHECK_CUDA(cudaFree(compute_out));
  CHECK_CUDA(cudaFree(recv));
  CHECK_CUDA(cudaFree(send));
  CHECK_CUDA(cudaStreamDestroy(compute_stream));
  CHECK_CUDA(cudaStreamDestroy(comm_stream));
  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_MPI(MPI_Finalize());
  return 0;
}
