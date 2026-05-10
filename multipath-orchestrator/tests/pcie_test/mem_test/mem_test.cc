/**
 * PCIe pure memcpy test
 *
 * Features:
 *   - Multi-card (default 4), each card runs memcpy independently
 *   - Per card: D2H (Device to Host), D2D (same-device copy, no cross-device)
 *   - Configurable: device count, copy size, iteration count
 *   - No bandwidth measurement, just runs specified iter loops
 *
 * Usage:
 *   ./mem_test
 *   ./mem_test --devices=4 --size=64M --iters=10
 *   ./mem_test --devices=2 --size=1024 --iters=5
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                                                 \
  do {                                                                                 \
    cudaError_t e = (cmd);                                                             \
    if (e != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__,                   \
              cudaGetErrorString(e));                                                  \
      return -1;                                                                       \
    }                                                                                  \
  } while (0)

static size_t parseSize(const char* str) {
  char* end;
  size_t val = strtoull(str, &end, 10);
  if (*end == 'K' || *end == 'k')
    val *= 1024;
  else if (*end == 'M' || *end == 'm')
    val *= 1024 * 1024;
  else if (*end == 'G' || *end == 'g')
    val *= 1024 * 1024 * 1024;
  return val;
}

static void printUsage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --devices=N    GPU count (default 4)\n");
  printf("  --size=S       bytes per card copy, supports K/M/G suffix (default 1M)\n");
  printf("  --iters=I      loops per card (default 1)\n");
  printf("\nExample: %s --devices=4 --size=64M --iters=10\n", prog);
}

static int runDevice(int device, int devCount, size_t copySize, uint32_t iters) {
  (void)devCount;

  CUDACHECK(cudaSetDevice(device));

  void* d_src = nullptr;
  void* d_dst = nullptr;
  void* h_buf = nullptr;

  CUDACHECK(cudaMalloc(&d_src, copySize));
  CUDACHECK(cudaMalloc(&d_dst, copySize));
  CUDACHECK(cudaMallocHost(&h_buf, copySize));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Initialize device memory
  CUDACHECK(cudaMemset(d_src, static_cast<int>(device & 0xFF), copySize));

  for (uint32_t i = 0; i < iters; ++i) {
    // D2H: Device to Host
    CUDACHECK(cudaMemcpyAsync(h_buf, d_src, copySize, cudaMemcpyDeviceToHost, stream));

    // D2D: same-device copy (src -> dst, no cross-device)
    CUDACHECK(cudaMemcpyAsync(d_dst, d_src, copySize, cudaMemcpyDeviceToDevice, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
  }

  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(d_src));
  CUDACHECK(cudaFree(d_dst));
  CUDACHECK(cudaFreeHost(h_buf));

  return 0;
}

int main(int argc, char* argv[]) {
  int devCount = 4;
  size_t copySize = 1024 * 1024;  // 1M
  uint32_t iters = 1;

  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--devices=", 10) == 0) {
      devCount = atoi(argv[i] + 10);
    } else if (strncmp(argv[i], "--size=", 7) == 0) {
      copySize = parseSize(argv[i] + 7);
    } else if (strncmp(argv[i], "--iters=", 8) == 0) {
      iters = static_cast<uint32_t>(atoi(argv[i] + 8));
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      return 0;
    }
  }

  if (devCount <= 0 || copySize == 0 || iters == 0) {
    fprintf(stderr, "Invalid parameters: devices=%d size=%zu iters=%u\n",
            devCount, copySize, iters);
    printUsage(argv[0]);
    return -1;
  }

  int availableDevs = 0;
  CUDACHECK(cudaGetDeviceCount(&availableDevs));
  if (devCount > availableDevs) {
    fprintf(stderr, "Requested %d devices but only %d available\n", devCount, availableDevs);
    return -1;
  }

  printf("PCIe memcpy test: devices=%d size=%zu bytes (%.2f MB) iters=%u\n",
         devCount, copySize, copySize / (1024.0 * 1024.0), iters);
  printf("Per device: D2H + D2D (same device) x %u iters\n", iters);

  std::vector<std::thread> threads(devCount);
  std::atomic<int> anyFailed{0};

  for (int d = 0; d < devCount; ++d) {
    threads[d] = std::thread([d, devCount, copySize, iters, &anyFailed]() {
      int r = runDevice(d, devCount, copySize, iters);
      if (r != 0)
        anyFailed.store(1);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  if (anyFailed.load()) {
    fprintf(stderr, "Test FAILED\n");
    return 1;
  }

  printf("Test PASSED\n");
  return 0;
}
