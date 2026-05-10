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
  unsigned long long iters = 100;
  unsigned long long warmup = 10;
  unsigned report_sec = 5;
  std::string pattern = "allpairs";
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
  printf("  --devices=N        GPU count (default 4)\n");
  printf("  --size=S           bytes per directed copy, K/M/G suffix supported (default 255M)\n");
  printf("  --iters=N          measured iterations; 0 means run forever (default 100)\n");
  printf("  --warmup=N         warmup iterations before timing (default 10)\n");
  printf("  --report-sec=N     report interval for --iters=0 (default 5)\n");
  printf("  --pattern=ring     copy i -> i+1 only\n");
  printf("  --pattern=allpairs copy every ordered GPU pair i -> j, i != j (default)\n");
  printf("\nExamples:\n");
  printf("  %s --devices=4 --size=255M --iters=100\n", prog);
  printf("  %s --devices=4 --size=255M --iters=0 --pattern=ring\n", prog);
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
    } else if (strncmp(argv[i], "--pattern=", 10) == 0) {
      opt->pattern = argv[i] + 10;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      exit(0);
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      printUsage(argv[0]);
      return -1;
    }
  }

  if (opt->devices <= 1 || opt->size == 0 || opt->report_sec == 0) {
    fprintf(stderr, "Invalid devices=%d size=%zu report_sec=%u\n",
            opt->devices, opt->size, opt->report_sec);
    return -1;
  }
  if (opt->pattern != "ring" && opt->pattern != "allpairs") {
    fprintf(stderr, "Invalid pattern '%s'; expected ring or allpairs\n",
            opt->pattern.c_str());
    return -1;
  }
  return 0;
}

static int enablePeerAccess(int devices) {
  for (int d = 0; d < devices; ++d) {
    CUDACHECK(cudaSetDevice(d));
    for (int p = 0; p < devices; ++p) {
      if (d == p) continue;
      int can_access = 0;
      CUDACHECK(cudaDeviceCanAccessPeer(&can_access, d, p));
      if (!can_access) {
        fprintf(stderr, "GPU %d cannot access GPU %d by peer access\n", d, p);
        return -1;
      }
      cudaError_t err = cudaDeviceEnablePeerAccess(p, 0);
      if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "GPU %d enable peer %d failed: %s\n",
                d, p, cudaGetErrorString(err));
        return -1;
      }
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      }
    }
  }
  return 0;
}

static int syncAll(int devices) {
  for (int d = 0; d < devices; ++d) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return 0;
}

static int issueCopies(const Options& opt,
                       const std::vector<void*>& src,
                       const std::vector<std::vector<void*> >& dst,
                       const std::vector<cudaStream_t>& streams) {
  for (int s = 0; s < opt.devices; ++s) {
    CUDACHECK(cudaSetDevice(s));
    if (opt.pattern == "ring") {
      int d = (s + 1) % opt.devices;
      CUDACHECK(cudaMemcpyPeerAsync(dst[d][s], d, src[s], s, opt.size, streams[s]));
    } else {
      for (int d = 0; d < opt.devices; ++d) {
        if (d == s) continue;
        CUDACHECK(cudaMemcpyPeerAsync(dst[d][s], d, src[s], s, opt.size, streams[s]));
      }
    }
  }
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

  int copies_per_iter = opt.pattern == "ring" ? opt.devices : opt.devices * (opt.devices - 1);
  double bytes_per_iter = static_cast<double>(opt.size) * copies_per_iter;

  printf("CUDA peer D2D test: devices=%d size=%zu bytes (%.2f MiB) pattern=%s\n",
         opt.devices, opt.size, opt.size / (1024.0 * 1024.0), opt.pattern.c_str());
  printf("Per iteration: %d directed peer copies, %.2f GiB transferred\n",
         copies_per_iter, bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
  printf("Iters=%llu warmup=%llu%s\n",
         opt.iters, opt.warmup, opt.iters == 0 ? " (continuous)" : "");

  if (enablePeerAccess(opt.devices) != 0) return 1;

  std::vector<void*> src(opt.devices, nullptr);
  std::vector<std::vector<void*> > dst(opt.devices, std::vector<void*>(opt.devices, nullptr));
  std::vector<cudaStream_t> streams(opt.devices);

  for (int d = 0; d < opt.devices; ++d) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaMalloc(&src[d], opt.size));
    CUDACHECK(cudaMemset(src[d], d & 0xff, opt.size));
    for (int s = 0; s < opt.devices; ++s) {
      if (s == d) continue;
      CUDACHECK(cudaMalloc(&dst[d][s], opt.size));
    }
    CUDACHECK(cudaStreamCreate(&streams[d]));
  }

  for (unsigned long long i = 0; i < opt.warmup; ++i) {
    if (issueCopies(opt, src, dst, streams) != 0) return 1;
    if (syncAll(opt.devices) != 0) return 1;
  }

  using clock = std::chrono::steady_clock;
  if (opt.iters == 0) {
    unsigned long long total_iters = 0;
    auto window_start = clock::now();
    unsigned long long window_iters = 0;

    while (true) {
      if (issueCopies(opt, src, dst, streams) != 0) return 1;
      if (syncAll(opt.devices) != 0) return 1;
      ++total_iters;
      ++window_iters;

      auto now = clock::now();
      double sec = std::chrono::duration<double>(now - window_start).count();
      if (sec >= opt.report_sec) {
        double gbps = (bytes_per_iter * window_iters) / sec / 1.0e9;
        printf("iters=%llu window_iters=%llu time=%.3f s aggregate_bw=%.2f GB/s\n",
               total_iters, window_iters, sec, gbps);
        fflush(stdout);
        window_start = now;
        window_iters = 0;
      }
    }
  } else {
    auto start = clock::now();
    for (unsigned long long i = 0; i < opt.iters; ++i) {
      if (issueCopies(opt, src, dst, streams) != 0) return 1;
      if (syncAll(opt.devices) != 0) return 1;
    }
    auto end = clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    double gbps = (bytes_per_iter * opt.iters) / sec / 1.0e9;
    printf("Result: iters=%llu time=%.6f s aggregate_bw=%.2f GB/s per_copy_avg=%.2f GB/s\n",
           opt.iters, sec, gbps, gbps / copies_per_iter);
  }

  for (int d = 0; d < opt.devices; ++d) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaStreamDestroy(streams[d]));
    CUDACHECK(cudaFree(src[d]));
    for (int s = 0; s < opt.devices; ++s) {
      if (dst[d][s]) CUDACHECK(cudaFree(dst[d][s]));
    }
  }
  return 0;
}
