#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                                        \
  do {                                                                        \
    cudaError_t e = (cmd);                                                    \
    if (e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__,          \
              cudaGetErrorString(e));                                         \
      return -1;                                                              \
    }                                                                         \
  } while (0)

struct Options {
  int devices = 4;
  size_t size = 255ULL * 1024ULL * 1024ULL;
  unsigned long long iters = 0;
  unsigned long long warmup = 5;
  unsigned report_sec = 5;
  std::string direction = "d2h";
};

static size_t parseSize(const char* str) {
  char* end = nullptr;
  double val = strtod(str, &end);
  if (end == str) return 0;

  double mult = 1.0;
  if (*end == 'K' || *end == 'k')
    mult = 1024.0;
  else if (*end == 'M' || *end == 'm')
    mult = 1024.0 * 1024.0;
  else if (*end == 'G' || *end == 'g')
    mult = 1024.0 * 1024.0 * 1024.0;

  return static_cast<size_t>(val * mult);
}

static void printUsage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --devices=N       GPU count (default 4)\n");
  printf("  --size=S          bytes per GPU copy, K/M/G suffix supported (default 255M)\n");
  printf("  --iters=N         measured iterations; 0 means run forever (default 0)\n");
  printf("  --warmup=N        warmup iterations before timing (default 5)\n");
  printf("  --report-sec=N    report interval for --iters=0 (default 5)\n");
  printf("  --direction=d2h   device to host copy (default)\n");
  printf("  --direction=h2d   host to device copy\n");
  printf("\nExamples:\n");
  printf("  %s --devices=4 --size=255M --direction=d2h --iters=0\n", prog);
  printf("  %s --devices=4 --size=255M --direction=h2d --iters=100\n", prog);
}

static int parseArgs(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--devices=", 10) == 0) {
      opt->devices = atoi(argv[i] + 10);
    } else if (strncmp(argv[i], "--size=", 7) == 0) {
      opt->size = parseSize(argv[i] + 7);
    } else if (strncmp(argv[i], "--iters=", 8) == 0) {
      opt->iters = strtoull(argv[i] + 8, nullptr, 10);
    } else if (strncmp(argv[i], "--warmup=", 9) == 0) {
      opt->warmup = strtoull(argv[i] + 9, nullptr, 10);
    } else if (strncmp(argv[i], "--report-sec=", 13) == 0) {
      opt->report_sec = static_cast<unsigned>(atoi(argv[i] + 13));
    } else if (strncmp(argv[i], "--direction=", 12) == 0) {
      opt->direction = argv[i] + 12;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      exit(0);
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      printUsage(argv[0]);
      return -1;
    }
  }

  if (opt->devices <= 0 || opt->size == 0 || opt->report_sec == 0) {
    fprintf(stderr, "Invalid devices=%d size=%zu report_sec=%u\n",
            opt->devices, opt->size, opt->report_sec);
    return -1;
  }
  if (opt->direction != "d2h" && opt->direction != "h2d") {
    fprintf(stderr, "Invalid direction '%s'; expected d2h or h2d\n",
            opt->direction.c_str());
    return -1;
  }
  return 0;
}

struct WorkerState {
  std::atomic<unsigned long long> bytes{0};
  std::atomic<int> failed{0};
};

static int runWorker(int device, const Options& opt, WorkerState* state,
                     std::atomic<int>* ready, std::atomic<int>* go) {
  CUDACHECK(cudaSetDevice(device));

  void* h_buf = nullptr;
  void* d_buf = nullptr;
  cudaStream_t stream = nullptr;

  CUDACHECK(cudaMallocHost(&h_buf, opt.size));
  CUDACHECK(cudaMalloc(&d_buf, opt.size));
  CUDACHECK(cudaStreamCreate(&stream));
  CUDACHECK(cudaMemset(d_buf, device & 0xff, opt.size));
  memset(h_buf, device & 0xff, opt.size);

  cudaMemcpyKind kind = opt.direction == "d2h" ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
  void* dst = opt.direction == "d2h" ? h_buf : d_buf;
  void* src = opt.direction == "d2h" ? d_buf : h_buf;

  for (unsigned long long i = 0; i < opt.warmup; ++i) {
    CUDACHECK(cudaMemcpyAsync(dst, src, opt.size, kind, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  ready->fetch_add(1);
  while (!go->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  if (opt.iters == 0) {
    while (true) {
      CUDACHECK(cudaMemcpyAsync(dst, src, opt.size, kind, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      state->bytes.fetch_add(opt.size, std::memory_order_relaxed);
    }
  } else {
    for (unsigned long long i = 0; i < opt.iters; ++i) {
      CUDACHECK(cudaMemcpyAsync(dst, src, opt.size, kind, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      state->bytes.fetch_add(opt.size, std::memory_order_relaxed);
    }
  }

  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(d_buf));
  CUDACHECK(cudaFreeHost(h_buf));
  return 0;
}

int main(int argc, char** argv) {
  Options opt;
  if (parseArgs(argc, argv, &opt) != 0) return 1;

  int available = 0;
  CUDACHECK(cudaGetDeviceCount(&available));
  if (opt.devices > available) {
    fprintf(stderr, "Requested %d GPUs but only %d available\n", opt.devices, available);
    return 1;
  }

  printf("PCIe CUDA memcpy test: devices=%d direction=%s size=%zu bytes (%.2f MiB)\n",
         opt.devices, opt.direction.c_str(), opt.size, opt.size / (1024.0 * 1024.0));
  printf("Iters=%llu warmup=%llu%s\n",
         opt.iters, opt.warmup, opt.iters == 0 ? " (continuous)" : "");

  WorkerState state;
  std::atomic<int> ready{0};
  std::atomic<int> go{0};
  std::vector<std::thread> threads;
  threads.reserve(opt.devices);

  for (int d = 0; d < opt.devices; ++d) {
    threads.emplace_back([d, &opt, &state, &ready, &go]() {
      int rc = runWorker(d, opt, &state, &ready, &go);
      if (rc != 0) state.failed.store(1);
    });
  }

  while (ready.load(std::memory_order_acquire) < opt.devices) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  using clock = std::chrono::steady_clock;
  unsigned long long start_bytes = state.bytes.load(std::memory_order_relaxed);
  auto start = clock::now();
  go.store(1, std::memory_order_release);

  if (opt.iters == 0) {
    unsigned long long last_bytes = start_bytes;
    auto last = start;
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(opt.report_sec));
      auto now = clock::now();
      unsigned long long cur = state.bytes.load(std::memory_order_relaxed);
      double sec = std::chrono::duration<double>(now - last).count();
      double gbps = static_cast<double>(cur - last_bytes) / sec / 1.0e9;
      printf("bytes=%llu window_bytes=%llu time=%.3f s aggregate_bw=%.2f GB/s\n",
             cur, cur - last_bytes, sec, gbps);
      fflush(stdout);
      last = now;
      last_bytes = cur;
      if (state.failed.load()) return 1;
    }
  }

  for (auto& t : threads) t.join();
  auto end = clock::now();
  unsigned long long total_bytes = state.bytes.load(std::memory_order_relaxed) - start_bytes;
  double sec = std::chrono::duration<double>(end - start).count();
  double gbps = static_cast<double>(total_bytes) / sec / 1.0e9;
  printf("Result: bytes=%llu time=%.6f s aggregate_bw=%.2f GB/s per_gpu_avg=%.2f GB/s\n",
         total_bytes, sec, gbps, gbps / opt.devices);

  return state.failed.load() ? 1 : 0;
}
