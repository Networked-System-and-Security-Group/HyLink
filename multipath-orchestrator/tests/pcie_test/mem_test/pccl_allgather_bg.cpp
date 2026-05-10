#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <pccl/pccl.h>

#define CUDACHECK(cmd)                                                        \
  do {                                                                        \
    cudaError_t e = (cmd);                                                    \
    if (e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__,          \
              cudaGetErrorString(e));                                         \
      return -1;                                                              \
    }                                                                         \
  } while (0)

#define PCCLCHECK(cmd)                                                        \
  do {                                                                        \
    pcclResult_t r = (cmd);                                                   \
    if (r != pcclSuccess) {                                                   \
      fprintf(stderr, "PCCL error %s:%d '%s'\n", __FILE__, __LINE__,          \
              pcclGetErrorString(r));                                         \
      return -1;                                                              \
    }                                                                         \
  } while (0)

struct Options {
  int devices = 4;
  size_t size = 255ULL * 1024ULL * 1024ULL;
  unsigned long long iters = 0;
  unsigned long long warmup = 2;
  unsigned report_sec = 5;
};

static pcclUniqueId singleProcessId() {
  pcclUniqueId id;
  std::memset(&id, 0, sizeof(id));
  return id;
}

static size_t parseSize(const char* str) {
  char* end = nullptr;
  double val = strtod(str, &end);
  if (end == str) return 0;
  double mult = 1.0;
  if (*end == 'K' || *end == 'k') mult = 1024.0;
  else if (*end == 'M' || *end == 'm') mult = 1024.0 * 1024.0;
  else if (*end == 'G' || *end == 'g') mult = 1024.0 * 1024.0 * 1024.0;
  return static_cast<size_t>(val * mult);
}

static void usage(const char* prog) {
  printf("Usage: %s [--devices=N] [--size=255M] [--iters=N] [--warmup=N] [--report-sec=N]\n", prog);
  printf("  --iters=0 means continuous background mode.\n");
}

static int parseArgs(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--devices=", 10) == 0) opt->devices = atoi(argv[i] + 10);
    else if (strncmp(argv[i], "--size=", 7) == 0) opt->size = parseSize(argv[i] + 7);
    else if (strncmp(argv[i], "--iters=", 8) == 0) opt->iters = strtoull(argv[i] + 8, nullptr, 10);
    else if (strncmp(argv[i], "--warmup=", 9) == 0) opt->warmup = strtoull(argv[i] + 9, nullptr, 10);
    else if (strncmp(argv[i], "--report-sec=", 13) == 0) opt->report_sec = static_cast<unsigned>(atoi(argv[i] + 13));
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(argv[0]); exit(0); }
    else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return -1; }
  }
  if (opt->devices <= 1 || opt->size == 0 || opt->report_sec == 0) return -1;
  return 0;
}

static int syncAll(const std::vector<cudaStream_t>& streams) {
  for (size_t r = 0; r < streams.size(); ++r) {
    CUDACHECK(cudaSetDevice(static_cast<int>(r)));
    CUDACHECK(cudaStreamSynchronize(streams[r]));
  }
  return 0;
}

static int issueAllGather(int devices, size_t sendcount,
                          const std::vector<void*>& send,
                          const std::vector<void*>& recv,
                          const std::vector<pcclComm_t>& comms,
                          const std::vector<cudaStream_t>& streams) {
  std::vector<std::thread> threads;
  threads.reserve(devices);
  std::atomic<int> failed{0};
  for (int r = 0; r < devices; ++r) {
    threads.emplace_back([&, r]() {
      cudaSetDevice(r);
      pcclResult_t pr = pcclAllGather(send[r], recv[r], sendcount, comms[r], streams[r]);
      if (pr != pcclSuccess) {
        fprintf(stderr, "rank %d pcclAllGather failed: %s\n", r, pcclGetErrorString(pr));
        failed.store(1);
      }
    });
  }
  for (auto& t : threads) t.join();
  return failed.load() ? -1 : 0;
}

int main(int argc, char** argv) {
  Options opt;
  if (parseArgs(argc, argv, &opt) != 0) return 1;
  if (opt.size % sizeof(float) != 0) {
    fprintf(stderr, "size must be divisible by sizeof(float)\n");
    return 1;
  }
  size_t count = opt.size / sizeof(float);
  if (count % static_cast<size_t>(opt.devices) != 0) {
    fprintf(stderr, "float count must be divisible by devices\n");
    return 1;
  }
  size_t sendcount = count / opt.devices;
  double bytes_per_iter = static_cast<double>(opt.size);

  int available = 0;
  CUDACHECK(cudaGetDeviceCount(&available));
  if (opt.devices > available) return 1;

  printf("PCCL AllGather bg: devices=%d size=%zu bytes (%.2f MiB) sendcount=%zu floats iters=%llu warmup=%llu\n",
         opt.devices, opt.size, opt.size / (1024.0 * 1024.0), sendcount, opt.iters, opt.warmup);
  fflush(stdout);

  std::vector<pcclComm_t> comms(opt.devices);
  std::vector<cudaStream_t> streams(opt.devices);
  pcclUniqueId uid = singleProcessId();
  for (int r = 0; r < opt.devices; ++r) {
    CUDACHECK(cudaSetDevice(r));
    PCCLCHECK(pcclCommInitRank(&comms[r], opt.devices, uid, r));
    CUDACHECK(cudaStreamCreate(&streams[r]));
  }

  std::vector<void*> send(opt.devices, nullptr), recv(opt.devices, nullptr);
  for (int r = 0; r < opt.devices; ++r) {
    CUDACHECK(cudaSetDevice(r));
    CUDACHECK(cudaMalloc(&send[r], sendcount * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[r], opt.size));
    CUDACHECK(cudaMemset(send[r], r + 1, sendcount * sizeof(float)));
    CUDACHECK(cudaMemset(recv[r], 0, opt.size));
  }

  for (unsigned long long i = 0; i < opt.warmup; ++i) {
    if (issueAllGather(opt.devices, sendcount, send, recv, comms, streams) != 0) return 1;
    if (syncAll(streams) != 0) return 1;
  }

  using clock = std::chrono::steady_clock;
  if (opt.iters == 0) {
    unsigned long long total = 0, window = 0;
    auto start = clock::now();
    while (true) {
      if (issueAllGather(opt.devices, sendcount, send, recv, comms, streams) != 0) return 1;
      if (syncAll(streams) != 0) return 1;
      ++total;
      ++window;
      auto now = clock::now();
      double sec = std::chrono::duration<double>(now - start).count();
      if (sec >= opt.report_sec) {
        double algbw = bytes_per_iter * window / sec / 1.0e9;
        double busbw = algbw * (static_cast<double>(opt.devices - 1) / opt.devices);
        printf("iters=%llu window_iters=%llu time=%.3f s algbw=%.2f GB/s busbw=%.2f GB/s\n",
               total, window, sec, algbw, busbw);
        fflush(stdout);
        start = now;
        window = 0;
      }
    }
  }

  auto t0 = clock::now();
  for (unsigned long long i = 0; i < opt.iters; ++i) {
    if (issueAllGather(opt.devices, sendcount, send, recv, comms, streams) != 0) return 1;
    if (syncAll(streams) != 0) return 1;
  }
  auto t1 = clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double algbw = bytes_per_iter * opt.iters / sec / 1.0e9;
  double busbw = algbw * (static_cast<double>(opt.devices - 1) / opt.devices);
  printf("Result: iters=%llu time=%.6f s algbw=%.2f GB/s busbw=%.2f GB/s\n",
         opt.iters, sec, algbw, busbw);

  for (int r = 0; r < opt.devices; ++r) {
    cudaSetDevice(r);
    cudaFree(send[r]);
    cudaFree(recv[r]);
    cudaStreamDestroy(streams[r]);
    pcclCommDestroy(comms[r]);
  }
  pcclFinalize();
  return 0;
}
