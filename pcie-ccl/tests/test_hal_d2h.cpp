#include <sys/mman.h>
#include <unistd.h>

#include "test_utils.hpp"

using namespace pccl;

static bool use_malloc = false;

static void allocHostMem(void **ptr, size_t size) {
  if (use_malloc) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (posix_memalign(ptr, page_size, size) != 0) {
      *ptr = nullptr;
      return;
    }
    DEV_CHECK(devHostRegister(*ptr, size));
  } else {
    DEV_CHECK(devMallocHost(ptr, size));
  }
}

static void freeHostMem(void *ptr, size_t size) {
  if (use_malloc) {
    DEV_CHECK(devHostUnregister(ptr));
    free(ptr);
  } else {
    DEV_CHECK(devFreeHost(ptr));
  }
}

void testD2H(int dev_id, size_t size) {
  printTestHeader("D2H Bandwidth", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Device: %d\n\n", dev_id);

  DEV_CHECK(devSetDevice(dev_id));

  void *d_buf, *h_buf;
  DEV_CHECK(devMalloc(&d_buf, size));
  allocHostMem(&h_buf, size);

  devStream stream;
  DEV_CHECK(devCreateStream(&stream));

  DeviceTimer timer;
  std::vector<float> times;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer.recordStart(stream);
    DEV_CHECK(devMemcpyAsync(h_buf, d_buf, size, MemcpyDirection::D2H, stream));
    timer.recordEnd(stream);
    DEV_CHECK(devSynchronizeStream(stream));

    float ms = timer.getElapsedMs();
    double gbps = (size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

    bool verified = false;
    if (i < VERIFY_ITERS) {
      verified = verifyData((char *)h_buf, size, (char)0, "h_buf");
    }

    printIterationResult(i, ms, gbps, -1.0, verified);

    if (i >= WARMUP_ITERS)
      times.push_back(ms);
  }

  auto stats = calculateStats(times, size);
  printStatistics(stats);

  DEV_CHECK(devDestroyStream(stream));
  DEV_CHECK(devFree(d_buf));
  freeHostMem(h_buf, size);
}

void testH2D(int dev_id, size_t size) {
  printTestHeader("H2D Bandwidth", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Device: %d\n\n", dev_id);

  DEV_CHECK(devSetDevice(dev_id));

  void *d_buf, *h_buf, *h_verify;
  DEV_CHECK(devMalloc(&d_buf, size));
  allocHostMem(&h_buf, size);
  allocHostMem(&h_verify, size);

  memset(h_buf, 0x42, size);

  devStream stream;
  DEV_CHECK(devCreateStream(&stream));

  DeviceTimer timer;
  std::vector<float> times;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer.recordStart(stream);
    DEV_CHECK(devMemcpyAsync(d_buf, h_buf, size, MemcpyDirection::H2D, stream));
    timer.recordEnd(stream);
    DEV_CHECK(devSynchronizeStream(stream));

    float ms = timer.getElapsedMs();
    double gbps = (size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

    bool verified = false;
    if (i < VERIFY_ITERS) {
      DEV_CHECK(devMemcpyAsync(h_verify, d_buf, size, MemcpyDirection::D2H, stream));
      DEV_CHECK(devSynchronizeStream(stream));
      verified = verifyData((char *)h_verify, size, (char)0x42, "h_verify");
    }

    printIterationResult(i, ms, gbps, -1.0, verified);

    if (i >= WARMUP_ITERS)
      times.push_back(ms);
  }

  auto stats = calculateStats(times, size);
  printStatistics(stats);

  DEV_CHECK(devDestroyStream(stream));
  DEV_CHECK(devFree(d_buf));
  freeHostMem(h_buf, size);
  freeHostMem(h_verify, size);
}

void testBidirectional(int dev_id, size_t size) {
  printTestHeader("Bidirectional (D2H + H2D)", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Device: %d\n\n", dev_id);

  DEV_CHECK(devSetDevice(dev_id));

  void *d_buf1, *d_buf2, *h_buf1, *h_buf2;
  DEV_CHECK(devMalloc(&d_buf1, size));
  DEV_CHECK(devMalloc(&d_buf2, size));
  allocHostMem(&h_buf1, size);
  allocHostMem(&h_buf2, size);

  devStream stream1, stream2;
  DEV_CHECK(devCreateStream(&stream1));
  DEV_CHECK(devCreateStream(&stream2));

  DeviceTimer timer1, timer2;
  std::vector<float> times_d2h, times_h2d;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer1.recordStart(stream1);
    DEV_CHECK(devMemcpyAsync(h_buf1, d_buf1, size, MemcpyDirection::D2H, stream1));
    timer1.recordEnd(stream1);

    timer2.recordStart(stream2);
    DEV_CHECK(devMemcpyAsync(d_buf2, h_buf2, size, MemcpyDirection::H2D, stream2));
    timer2.recordEnd(stream2);

    DEV_CHECK(devSynchronizeStream(stream1));
    DEV_CHECK(devSynchronizeStream(stream2));

    float ms1 = timer1.getElapsedMs();
    float ms2 = timer2.getElapsedMs();
    double gbps1 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1 / 1000.0);
    double gbps2 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms2 / 1000.0);

    printf("Iteration %d%s: D2H: %.2f ms, %.2f GB/s | H2D: %.2f ms, %.2f GB/s | Total: %.2f GB/s\n", i,
           i < WARMUP_ITERS ? " (warmup)" : "", ms1, gbps1, ms2, gbps2, gbps1 + gbps2);

    if (i >= WARMUP_ITERS) {
      times_d2h.push_back(ms1);
      times_h2d.push_back(ms2);
    }
  }

  auto stats_d2h = calculateStats(times_d2h, size);
  auto stats_h2d = calculateStats(times_h2d, size);

  printf("\nD2H Statistics:\n");
  printf("  Bandwidth: Avg: %.2f GB/s\n", stats_d2h.avg_bw_gbps);
  printf("H2D Statistics:\n");
  printf("  Bandwidth: Avg: %.2f GB/s\n", stats_h2d.avg_bw_gbps);
  printf("Total Bidirectional: Avg: %.2f GB/s\n", stats_d2h.avg_bw_gbps + stats_h2d.avg_bw_gbps);

  DEV_CHECK(devDestroyStream(stream1));
  DEV_CHECK(devDestroyStream(stream2));
  DEV_CHECK(devFree(d_buf1));
  DEV_CHECK(devFree(d_buf2));
  freeHostMem(h_buf1, size);
  freeHostMem(h_buf2, size);
}

void testDualDevice_D2H_D2H(int dev0, int dev1, size_t size) {
  printTestHeader("Dual-Device D2H+D2H", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Devices: %d, %d\n\n", dev0, dev1);

  DEV_CHECK(devSetDevice(dev0));
  void *d_buf0, *h_buf0;
  DEV_CHECK(devMalloc(&d_buf0, size));
  allocHostMem(&h_buf0, size);
  devStream stream0;
  DEV_CHECK(devCreateStream(&stream0));
  DeviceTimer timer0;

  DEV_CHECK(devSetDevice(dev1));
  void *d_buf1, *h_buf1;
  DEV_CHECK(devMalloc(&d_buf1, size));
  allocHostMem(&h_buf1, size);
  devStream stream1;
  DEV_CHECK(devCreateStream(&stream1));
  DeviceTimer timer1;

  std::vector<float> times0, times1;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    DEV_CHECK(devSetDevice(dev0));
    timer0.recordStart(stream0);
    DEV_CHECK(devMemcpyAsync(h_buf0, d_buf0, size, MemcpyDirection::D2H, stream0));
    timer0.recordEnd(stream0);

    DEV_CHECK(devSetDevice(dev1));
    timer1.recordStart(stream1);
    DEV_CHECK(devMemcpyAsync(h_buf1, d_buf1, size, MemcpyDirection::D2H, stream1));
    timer1.recordEnd(stream1);

    DEV_CHECK(devSetDevice(dev0));
    DEV_CHECK(devSynchronizeStream(stream0));
    DEV_CHECK(devSetDevice(dev1));
    DEV_CHECK(devSynchronizeStream(stream1));

    float ms0 = timer0.getElapsedMs();
    float ms1 = timer1.getElapsedMs();
    double gbps0 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms0 / 1000.0);
    double gbps1 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1 / 1000.0);

    printf("Iteration %d%s: Dev0: %.2f GB/s, Dev1: %.2f GB/s, Total: %.2f GB/s\n", i,
           i < WARMUP_ITERS ? " (warmup)" : "", gbps0, gbps1, gbps0 + gbps1);

    if (i >= WARMUP_ITERS) {
      times0.push_back(ms0);
      times1.push_back(ms1);
    }
  }

  auto stats0 = calculateStats(times0, size);
  auto stats1 = calculateStats(times1, size);
  printDualDeviceStats("Dual-Device D2H+D2H", stats0, stats1);

  DEV_CHECK(devSetDevice(dev0));
  DEV_CHECK(devDestroyStream(stream0));
  DEV_CHECK(devFree(d_buf0));
  freeHostMem(h_buf0, size);
  DEV_CHECK(devSetDevice(dev1));
  DEV_CHECK(devDestroyStream(stream1));
  DEV_CHECK(devFree(d_buf1));
  freeHostMem(h_buf1, size);
}

void testDualDevice_D2H_H2D(int dev0, int dev1, size_t size) {
  printTestHeader("Dual-Device D2H+H2D", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Devices: %d (D2H), %d (H2D)\n\n", dev0, dev1);

  DEV_CHECK(devSetDevice(dev0));
  void *d_buf0, *h_buf0;
  DEV_CHECK(devMalloc(&d_buf0, size));
  allocHostMem(&h_buf0, size);
  devStream stream0;
  DEV_CHECK(devCreateStream(&stream0));
  DeviceTimer timer0;

  DEV_CHECK(devSetDevice(dev1));
  void *d_buf1, *h_buf1;
  DEV_CHECK(devMalloc(&d_buf1, size));
  allocHostMem(&h_buf1, size);
  devStream stream1;
  DEV_CHECK(devCreateStream(&stream1));
  DeviceTimer timer1;

  std::vector<float> times0, times1;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    DEV_CHECK(devSetDevice(dev0));
    timer0.recordStart(stream0);
    DEV_CHECK(devMemcpyAsync(h_buf0, d_buf0, size, MemcpyDirection::D2H, stream0));
    timer0.recordEnd(stream0);

    DEV_CHECK(devSetDevice(dev1));
    timer1.recordStart(stream1);
    DEV_CHECK(devMemcpyAsync(d_buf1, h_buf1, size, MemcpyDirection::H2D, stream1));
    timer1.recordEnd(stream1);

    DEV_CHECK(devSetDevice(dev0));
    DEV_CHECK(devSynchronizeStream(stream0));
    DEV_CHECK(devSetDevice(dev1));
    DEV_CHECK(devSynchronizeStream(stream1));

    float ms0 = timer0.getElapsedMs();
    float ms1 = timer1.getElapsedMs();
    double gbps0 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms0 / 1000.0);
    double gbps1 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1 / 1000.0);

    printf("Iteration %d%s: Dev0: %.2f GB/s, Dev1: %.2f GB/s, Total: %.2f GB/s\n", i,
           i < WARMUP_ITERS ? " (warmup)" : "", gbps0, gbps1, gbps0 + gbps1);

    if (i >= WARMUP_ITERS) {
      times0.push_back(ms0);
      times1.push_back(ms1);
    }
  }

  auto stats0 = calculateStats(times0, size);
  auto stats1 = calculateStats(times1, size);
  printDualDeviceStats("Dual-Device D2H+H2D", stats0, stats1);

  DEV_CHECK(devSetDevice(dev0));
  DEV_CHECK(devDestroyStream(stream0));
  DEV_CHECK(devFree(d_buf0));
  freeHostMem(h_buf0, size);
  DEV_CHECK(devSetDevice(dev1));
  DEV_CHECK(devDestroyStream(stream1));
  DEV_CHECK(devFree(d_buf1));
  freeHostMem(h_buf1, size);
}

void testDualDevice_H2D_D2H(int dev0, int dev1, size_t size) {
  printTestHeader("Dual-Device H2D+D2H", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Devices: %d (H2D), %d (D2H)\n\n", dev0, dev1);

  DEV_CHECK(devSetDevice(dev0));
  void *d_buf0, *h_buf0;
  DEV_CHECK(devMalloc(&d_buf0, size));
  allocHostMem(&h_buf0, size);
  devStream stream0;
  DEV_CHECK(devCreateStream(&stream0));
  DeviceTimer timer0;

  DEV_CHECK(devSetDevice(dev1));
  void *d_buf1, *h_buf1;
  DEV_CHECK(devMalloc(&d_buf1, size));
  allocHostMem(&h_buf1, size);
  devStream stream1;
  DEV_CHECK(devCreateStream(&stream1));
  DeviceTimer timer1;

  std::vector<float> times0, times1;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    DEV_CHECK(devSetDevice(dev0));
    timer0.recordStart(stream0);
    DEV_CHECK(devMemcpyAsync(d_buf0, h_buf0, size, MemcpyDirection::H2D, stream0));
    timer0.recordEnd(stream0);

    DEV_CHECK(devSetDevice(dev1));
    timer1.recordStart(stream1);
    DEV_CHECK(devMemcpyAsync(h_buf1, d_buf1, size, MemcpyDirection::D2H, stream1));
    timer1.recordEnd(stream1);

    DEV_CHECK(devSetDevice(dev0));
    DEV_CHECK(devSynchronizeStream(stream0));
    DEV_CHECK(devSetDevice(dev1));
    DEV_CHECK(devSynchronizeStream(stream1));

    float ms0 = timer0.getElapsedMs();
    float ms1 = timer1.getElapsedMs();
    double gbps0 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms0 / 1000.0);
    double gbps1 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1 / 1000.0);

    printf("Iteration %d%s: Dev0: %.2f GB/s, Dev1: %.2f GB/s, Total: %.2f GB/s\n", i,
           i < WARMUP_ITERS ? " (warmup)" : "", gbps0, gbps1, gbps0 + gbps1);

    if (i >= WARMUP_ITERS) {
      times0.push_back(ms0);
      times1.push_back(ms1);
    }
  }

  auto stats0 = calculateStats(times0, size);
  auto stats1 = calculateStats(times1, size);
  printDualDeviceStats("Dual-Device H2D+D2H", stats0, stats1);

  DEV_CHECK(devSetDevice(dev0));
  DEV_CHECK(devDestroyStream(stream0));
  DEV_CHECK(devFree(d_buf0));
  freeHostMem(h_buf0, size);
  DEV_CHECK(devSetDevice(dev1));
  DEV_CHECK(devDestroyStream(stream1));
  DEV_CHECK(devFree(d_buf1));
  freeHostMem(h_buf1, size);
}

void testDualDevice_H2D_H2D(int dev0, int dev1, size_t size) {
  printTestHeader("Dual-Device H2D+H2D", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Devices: %d, %d\n\n", dev0, dev1);

  DEV_CHECK(devSetDevice(dev0));
  void *d_buf0, *h_buf0;
  DEV_CHECK(devMalloc(&d_buf0, size));
  allocHostMem(&h_buf0, size);
  devStream stream0;
  DEV_CHECK(devCreateStream(&stream0));
  DeviceTimer timer0;

  DEV_CHECK(devSetDevice(dev1));
  void *d_buf1, *h_buf1;
  DEV_CHECK(devMalloc(&d_buf1, size));
  allocHostMem(&h_buf1, size);
  devStream stream1;
  DEV_CHECK(devCreateStream(&stream1));
  DeviceTimer timer1;

  std::vector<float> times0, times1;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    DEV_CHECK(devSetDevice(dev0));
    timer0.recordStart(stream0);
    DEV_CHECK(devMemcpyAsync(d_buf0, h_buf0, size, MemcpyDirection::H2D, stream0));
    timer0.recordEnd(stream0);

    DEV_CHECK(devSetDevice(dev1));
    timer1.recordStart(stream1);
    DEV_CHECK(devMemcpyAsync(d_buf1, h_buf1, size, MemcpyDirection::H2D, stream1));
    timer1.recordEnd(stream1);

    DEV_CHECK(devSetDevice(dev0));
    DEV_CHECK(devSynchronizeStream(stream0));
    DEV_CHECK(devSetDevice(dev1));
    DEV_CHECK(devSynchronizeStream(stream1));

    float ms0 = timer0.getElapsedMs();
    float ms1 = timer1.getElapsedMs();
    double gbps0 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms0 / 1000.0);
    double gbps1 = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1 / 1000.0);

    printf("Iteration %d%s: Dev0: %.2f GB/s, Dev1: %.2f GB/s, Total: %.2f GB/s\n", i,
           i < WARMUP_ITERS ? " (warmup)" : "", gbps0, gbps1, gbps0 + gbps1);

    if (i >= WARMUP_ITERS) {
      times0.push_back(ms0);
      times1.push_back(ms1);
    }
  }

  auto stats0 = calculateStats(times0, size);
  auto stats1 = calculateStats(times1, size);
  printDualDeviceStats("Dual-Device H2D+H2D", stats0, stats1);

  DEV_CHECK(devSetDevice(dev0));
  DEV_CHECK(devDestroyStream(stream0));
  DEV_CHECK(devFree(d_buf0));
  freeHostMem(h_buf0, size);
  DEV_CHECK(devSetDevice(dev1));
  DEV_CHECK(devDestroyStream(stream1));
  DEV_CHECK(devFree(d_buf1));
  freeHostMem(h_buf1, size);
}

void testDualDevice_Bidirectional(int dev0, int dev1, size_t size) {
  printTestHeader("Dual-Device Bidirectional (4 streams)", size, WARMUP_ITERS, MEASURE_ITERS);
  printf("Devices: %d, %d\n\n", dev0, dev1);

  DEV_CHECK(devSetDevice(dev0));
  void *d_buf0_1, *d_buf0_2, *h_buf0_1, *h_buf0_2;
  DEV_CHECK(devMalloc(&d_buf0_1, size));
  DEV_CHECK(devMalloc(&d_buf0_2, size));
  allocHostMem(&h_buf0_1, size);
  allocHostMem(&h_buf0_2, size);
  devStream stream0_d2h, stream0_h2d;
  DEV_CHECK(devCreateStream(&stream0_d2h));
  DEV_CHECK(devCreateStream(&stream0_h2d));
  DeviceTimer timer0_d2h, timer0_h2d;

  DEV_CHECK(devSetDevice(dev1));
  void *d_buf1_1, *d_buf1_2, *h_buf1_1, *h_buf1_2;
  DEV_CHECK(devMalloc(&d_buf1_1, size));
  DEV_CHECK(devMalloc(&d_buf1_2, size));
  allocHostMem(&h_buf1_1, size);
  allocHostMem(&h_buf1_2, size);
  devStream stream1_d2h, stream1_h2d;
  DEV_CHECK(devCreateStream(&stream1_d2h));
  DEV_CHECK(devCreateStream(&stream1_h2d));
  DeviceTimer timer1_d2h, timer1_h2d;

  std::vector<float> times0_d2h, times0_h2d, times1_d2h, times1_h2d;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    DEV_CHECK(devSetDevice(dev0));
    timer0_d2h.recordStart(stream0_d2h);
    DEV_CHECK(devMemcpyAsync(h_buf0_1, d_buf0_1, size, MemcpyDirection::D2H, stream0_d2h));
    timer0_d2h.recordEnd(stream0_d2h);
    timer0_h2d.recordStart(stream0_h2d);
    DEV_CHECK(devMemcpyAsync(d_buf0_2, h_buf0_2, size, MemcpyDirection::H2D, stream0_h2d));
    timer0_h2d.recordEnd(stream0_h2d);

    DEV_CHECK(devSetDevice(dev1));
    timer1_d2h.recordStart(stream1_d2h);
    DEV_CHECK(devMemcpyAsync(h_buf1_1, d_buf1_1, size, MemcpyDirection::D2H, stream1_d2h));
    timer1_d2h.recordEnd(stream1_d2h);
    timer1_h2d.recordStart(stream1_h2d);
    DEV_CHECK(devMemcpyAsync(d_buf1_2, h_buf1_2, size, MemcpyDirection::H2D, stream1_h2d));
    timer1_h2d.recordEnd(stream1_h2d);

    DEV_CHECK(devSetDevice(dev0));
    DEV_CHECK(devSynchronizeStream(stream0_d2h));
    DEV_CHECK(devSynchronizeStream(stream0_h2d));
    DEV_CHECK(devSetDevice(dev1));
    DEV_CHECK(devSynchronizeStream(stream1_d2h));
    DEV_CHECK(devSynchronizeStream(stream1_h2d));

    float ms0_d2h = timer0_d2h.getElapsedMs();
    float ms0_h2d = timer0_h2d.getElapsedMs();
    float ms1_d2h = timer1_d2h.getElapsedMs();
    float ms1_h2d = timer1_h2d.getElapsedMs();
    double gbps0_d2h = (size / (1024.0 * 1024.0 * 1024.0)) / (ms0_d2h / 1000.0);
    double gbps0_h2d = (size / (1024.0 * 1024.0 * 1024.0)) / (ms0_h2d / 1000.0);
    double gbps1_d2h = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1_d2h / 1000.0);
    double gbps1_h2d = (size / (1024.0 * 1024.0 * 1024.0)) / (ms1_h2d / 1000.0);

    printf(
        "Iteration %d%s: Dev0 D2H: %.2f GB/s, Dev0 H2D: %.2f GB/s, Dev1 D2H: %.2f GB/s, Dev1 H2D: %.2f GB/s, Total: "
        "%.2f GB/s\n",
        i, i < WARMUP_ITERS ? " (warmup)" : "", gbps0_d2h, gbps0_h2d, gbps1_d2h, gbps1_h2d,
        gbps0_d2h + gbps0_h2d + gbps1_d2h + gbps1_h2d);

    if (i >= WARMUP_ITERS) {
      times0_d2h.push_back(ms0_d2h);
      times0_h2d.push_back(ms0_h2d);
      times1_d2h.push_back(ms1_d2h);
      times1_h2d.push_back(ms1_h2d);
    }
  }

  auto stats0_d2h = calculateStats(times0_d2h, size);
  auto stats0_h2d = calculateStats(times0_h2d, size);
  auto stats1_d2h = calculateStats(times1_d2h, size);
  auto stats1_h2d = calculateStats(times1_h2d, size);
  printDualBidirectionalStats(stats0_d2h, stats0_h2d, stats1_d2h, stats1_h2d);

  DEV_CHECK(devSetDevice(dev0));
  DEV_CHECK(devDestroyStream(stream0_d2h));
  DEV_CHECK(devDestroyStream(stream0_h2d));
  DEV_CHECK(devFree(d_buf0_1));
  DEV_CHECK(devFree(d_buf0_2));
  freeHostMem(h_buf0_1, size);
  freeHostMem(h_buf0_2, size);
  DEV_CHECK(devSetDevice(dev1));
  DEV_CHECK(devDestroyStream(stream1_d2h));
  DEV_CHECK(devDestroyStream(stream1_h2d));
  DEV_CHECK(devFree(d_buf1_1));
  DEV_CHECK(devFree(d_buf1_2));
  freeHostMem(h_buf1_1, size);
  freeHostMem(h_buf1_2, size);
}

int main(int argc, char **argv) {
  int dev0 = 0, dev1 = 1;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--use-malloc") == 0) {
      use_malloc = true;
      printf("Using regular malloc (non-pinned memory)\n");
    } else if (i == 1) {
      dev0 = atoi(argv[i]);
    } else if (i == 2) {
      dev1 = atoi(argv[i]);
    }
  }

  DEV_CHECK(devInit());

  size_t sizes[] = {TEST_SIZE_1M, TEST_SIZE_512M};

  for (size_t size : sizes) {
    testD2H(dev0, size);
    testH2D(dev0, size);
    testBidirectional(dev0, size);
    testDualDevice_D2H_D2H(dev0, dev1, size);
    testDualDevice_D2H_H2D(dev0, dev1, size);
    testDualDevice_H2D_D2H(dev0, dev1, size);
    testDualDevice_H2D_H2D(dev0, dev1, size);
    testDualDevice_Bidirectional(dev0, dev1, size);
  }

  return 0;
}
