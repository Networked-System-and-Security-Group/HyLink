#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  size_t chunk = 16ULL * 1024ULL * 1024ULL;
  int sleep_us = 0;
  unsigned report_sec = 5;
};

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
  printf("Usage: %s [--devices=4] [--size=255M] [--chunk=16M] [--sleep-us=0] [--report-sec=5]\n", prog);
}

static int parseArgs(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--devices=", 10) == 0) opt->devices = atoi(argv[i] + 10);
    else if (strncmp(argv[i], "--size=", 7) == 0) opt->size = parseSize(argv[i] + 7);
    else if (strncmp(argv[i], "--chunk=", 8) == 0) opt->chunk = parseSize(argv[i] + 8);
    else if (strncmp(argv[i], "--sleep-us=", 11) == 0) opt->sleep_us = atoi(argv[i] + 11);
    else if (strncmp(argv[i], "--report-sec=", 13) == 0) opt->report_sec = static_cast<unsigned>(atoi(argv[i] + 13));
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(argv[0]); exit(0); }
    else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return -1; }
  }
  if (opt->devices <= 0 || opt->size == 0 || opt->chunk == 0 || opt->report_sec == 0) return -1;
  if (opt->chunk > opt->size) opt->chunk = opt->size;
  return 0;
}

struct Shared {
  std::atomic<unsigned long long> bytes{0};
  std::atomic<int> failed{0};
};

static int worker(int dev, Options opt, Shared* shared) {
  CUDACHECK(cudaSetDevice(dev));
  void* d = nullptr;
  void* h = nullptr;
  cudaStream_t stream = nullptr;
  CUDACHECK(cudaMalloc(&d, opt.size));
  CUDACHECK(cudaMallocHost(&h, opt.size));
  CUDACHECK(cudaStreamCreate(&stream));
  CUDACHECK(cudaMemset(d, dev & 0xff, opt.size));

  while (true) {
    size_t off = 0;
    while (off < opt.size) {
      size_t n = opt.chunk;
      if (off + n > opt.size) n = opt.size - off;
      char* hp = static_cast<char*>(h) + off;
      char* dp = static_cast<char*>(d) + off;
      CUDACHECK(cudaMemcpyAsync(hp, dp, n, cudaMemcpyDeviceToHost, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      shared->bytes.fetch_add(n, std::memory_order_relaxed);
      if (opt.sleep_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(opt.sleep_us));
      off += n;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  Options opt;
  if (parseArgs(argc, argv, &opt) != 0) return 1;
  int available = 0;
  CUDACHECK(cudaGetDeviceCount(&available));
  if (opt.devices > available) return 1;

  printf("D2H paced test: devices=%d size=%zu chunk=%zu sleep_us=%d\n", opt.devices, opt.size, opt.chunk, opt.sleep_us);
  fflush(stdout);

  Shared shared;
  std::vector<std::thread> threads;
  for (int d = 0; d < opt.devices; ++d) {
    threads.emplace_back([d, opt, &shared]() {
      if (worker(d, opt, &shared) != 0) shared.failed.store(1);
    });
  }

  using clock = std::chrono::steady_clock;
  unsigned long long last = 0;
  auto last_t = clock::now();
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(opt.report_sec));
    auto now = clock::now();
    unsigned long long cur = shared.bytes.load(std::memory_order_relaxed);
    double sec = std::chrono::duration<double>(now - last_t).count();
    printf("bytes=%llu window_bytes=%llu time=%.3f s aggregate_bw=%.2f GB/s\n", cur, cur - last, sec, (cur - last) / sec / 1.0e9);
    fflush(stdout);
    last = cur;
    last_t = now;
    if (shared.failed.load()) return 1;
  }
}
