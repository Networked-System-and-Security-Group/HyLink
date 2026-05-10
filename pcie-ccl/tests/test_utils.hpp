#pragma once

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/hal/device.hpp"

namespace pccl {

// Configuration Macros
#define WARMUP_ITERS 2
#define MEASURE_ITERS 10
#define VERIFY_ITERS 1
#define TEST_SIZE_1M (1ULL * 1024 * 1024)
#define TEST_SIZE_512M (512ULL * 1024 * 1024)
#define DEFAULT_SIZE (256ULL * 1024 * 1024)

// Error Checking Macros
#define DEV_CHECK(cmd)                                                              \
  do {                                                                              \
    auto err = (cmd);                                                               \
    if (err != devSuccess) {                                                        \
      fprintf(stderr, "Device error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

#define HOST_CHECK(cmd)                                                           \
  do {                                                                            \
    auto err = (cmd);                                                             \
    if (err != hostStatus::SUCCESS) {                                             \
      fprintf(stderr, "Host error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

#define PCCL_CHECK(cmd)                                                           \
  do {                                                                            \
    auto err = (cmd);                                                             \
    if (err != pcclSuccess) {                                                     \
      fprintf(stderr, "PCCL error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

// Statistics Structure
struct BandwidthStats {
  double min_time_ms, max_time_ms, avg_time_ms;
  double min_bw_gbps, max_bw_gbps, avg_bw_gbps;
  double min_bus_bw_gbps, max_bus_bw_gbps, avg_bus_bw_gbps;
};

// Device Timer Class
class DeviceTimer {
 private:
  devEvent start_event, end_event;

 public:
  DeviceTimer() {
    DEV_CHECK(devCreateTimingEvent(&start_event));
    DEV_CHECK(devCreateTimingEvent(&end_event));
  }

  ~DeviceTimer() {
    devDestroyEvent(start_event);
    devDestroyEvent(end_event);
  }

  void recordStart(devStream stream) { DEV_CHECK(devRecordEvent(start_event, stream)); }

  void recordEnd(devStream stream) { DEV_CHECK(devRecordEvent(end_event, stream)); }

  float getElapsedMs() {
    float ms;
    DEV_CHECK(devEventElapsedTime(&ms, start_event, end_event));
    return ms;
  }
};

// Host Timer Class
class HostTimer {
 private:
  std::chrono::high_resolution_clock::time_point start_time, end_time;

 public:
  void start() { start_time = std::chrono::high_resolution_clock::now(); }

  void end() { end_time = std::chrono::high_resolution_clock::now(); }

  float getElapsedMs() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0f;
  }
};

// Statistics Calculator
inline BandwidthStats calculateStats(const std::vector<float>& times_ms, size_t data_size,
                                     double bus_bw_factor = 1.0) {
  BandwidthStats stats;
  stats.min_time_ms = times_ms[0];
  stats.max_time_ms = times_ms[0];
  stats.avg_time_ms = 0;

  for (float t : times_ms) {
    if (t < stats.min_time_ms)
      stats.min_time_ms = t;
    if (t > stats.max_time_ms)
      stats.max_time_ms = t;
    stats.avg_time_ms += t;
  }
  stats.avg_time_ms /= times_ms.size();

  double size_gb = data_size / 1e9;
  stats.min_bw_gbps = size_gb / (stats.max_time_ms / 1000.0);
  stats.max_bw_gbps = size_gb / (stats.min_time_ms / 1000.0);
  stats.avg_bw_gbps = size_gb / (stats.avg_time_ms / 1000.0);

  stats.min_bus_bw_gbps = stats.min_bw_gbps * bus_bw_factor;
  stats.max_bus_bw_gbps = stats.max_bw_gbps * bus_bw_factor;
  stats.avg_bus_bw_gbps = stats.avg_bw_gbps * bus_bw_factor;

  return stats;
}

// Output Formatters
inline void printTestHeader(const char* test_name, size_t size, int warmup, int measure) {
  printf("\n=== Test: %s ===\n", test_name);
  printf("Size: %.0f MB\n", size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", warmup, measure);
}

inline void printIterationResult(int iter, float time_ms, double alg_bw_gbps, double bus_bw_gbps = -1.0,
                                 bool verified = false) {
  if (iter < WARMUP_ITERS) {
    if (bus_bw_gbps >= 0.0) {
      printf("Iteration %d (warmup): %.2f ms, alg %.2f GB/s, bus %.2f GB/s", iter, time_ms, alg_bw_gbps, bus_bw_gbps);
    } else {
      printf("Iteration %d (warmup): %.2f ms, %.2f GB/s", iter, time_ms, alg_bw_gbps);
    }
    if (verified)
      printf(" [VERIFIED]");
    printf("\n");
  } else {
    if (bus_bw_gbps >= 0.0) {
      printf("Iteration %d: %.2f ms, alg %.2f GB/s, bus %.2f GB/s\n", iter, time_ms, alg_bw_gbps, bus_bw_gbps);
    } else {
      printf("Iteration %d: %.2f ms, %.2f GB/s\n", iter, time_ms, alg_bw_gbps);
    }
  }
}

inline void printStatistics(const BandwidthStats& stats) {
  printf("\nStatistics (%d iterations):\n", MEASURE_ITERS);
  printf("  Time:      Min: %.2f ms, Max: %.2f ms, Avg: %.2f ms\n", stats.min_time_ms, stats.max_time_ms,
         stats.avg_time_ms);
  printf("  Alg BW:    Min: %.2f GB/s, Max: %.2f GB/s, Avg: %.2f GB/s\n", stats.min_bw_gbps, stats.max_bw_gbps,
         stats.avg_bw_gbps);
  printf("  Bus BW:    Min: %.2f GB/s, Max: %.2f GB/s, Avg: %.2f GB/s\n", stats.min_bus_bw_gbps, stats.max_bus_bw_gbps,
         stats.avg_bus_bw_gbps);
}

inline void printDualDeviceStats(const char* label, const BandwidthStats& dev0, const BandwidthStats& dev1) {
  printf("\n%s:\n", label);
  printf("  Device 0: Avg: %.2f GB/s\n", dev0.avg_bw_gbps);
  printf("  Device 1: Avg: %.2f GB/s\n", dev1.avg_bw_gbps);
  printf("  Total Aggregate: Avg: %.2f GB/s\n", dev0.avg_bw_gbps + dev1.avg_bw_gbps);
}

inline void printDualBidirectionalStats(const BandwidthStats& dev0_d2h, const BandwidthStats& dev0_h2d,
                                        const BandwidthStats& dev1_d2h, const BandwidthStats& dev1_h2d) {
  printf("\nDual-Device Bidirectional Statistics:\n");
  printf("  Device 0 D2H: Avg: %.2f GB/s\n", dev0_d2h.avg_bw_gbps);
  printf("  Device 0 H2D: Avg: %.2f GB/s\n", dev0_h2d.avg_bw_gbps);
  printf("  Device 1 D2H: Avg: %.2f GB/s\n", dev1_d2h.avg_bw_gbps);
  printf("  Device 1 H2D: Avg: %.2f GB/s\n", dev1_h2d.avg_bw_gbps);
  printf("  Total Aggregate: Avg: %.2f GB/s\n",
         dev0_d2h.avg_bw_gbps + dev0_h2d.avg_bw_gbps + dev1_d2h.avg_bw_gbps + dev1_h2d.avg_bw_gbps);
}

inline double calculateAlgBandwidth(size_t data_size, float time_ms) {
  double size_gb = data_size / 1e9;
  return size_gb / (time_ms / 1000.0);
}

// Verification Helper
template <typename T>
inline bool verifyData(const T* data, size_t count, T expected, const char* label) {
  for (size_t i = 0; i < count; i++) {
    if (data[i] != expected) {
      fprintf(stderr, "Verification failed at %s[%zu]: expected %d, got %d\n", label, i, (int)expected, (int)data[i]);
      return false;
    }
  }
  return true;
}

struct BarrierData {
  pthread_barrier_t barrier;
  volatile int initialized_magic;
};

const int BARRIER_MAGIC = 0x12345678;

class ProcessBarrier {
 private:
  std::string name;
  int fd;
  BarrierData* data;
  bool is_owner;

 public:
  ProcessBarrier(const std::string& shm_name, int count, bool create_owner) : name(shm_name), is_owner(create_owner) {
    if (is_owner) {
      shm_unlink(name.c_str());

      fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
      if (fd == -1)
        throw std::runtime_error("shm_open failed");

      if (ftruncate(fd, sizeof(BarrierData)) == -1)
        throw std::runtime_error("ftruncate failed");

      data = (BarrierData*)mmap(NULL, sizeof(BarrierData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (data == MAP_FAILED)
        throw std::runtime_error("mmap failed");

      data->initialized_magic = 0;

      pthread_barrierattr_t attr;
      pthread_barrierattr_init(&attr);
      pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);

      int ret = pthread_barrier_init(&data->barrier, &attr, count);
      pthread_barrierattr_destroy(&attr);
      if (ret != 0)
        throw std::runtime_error("pthread_barrier_init failed");

      __sync_synchronize();
      data->initialized_magic = BARRIER_MAGIC;

    } else {
      int retries = 0;
      while ((fd = shm_open(name.c_str(), O_RDWR, 0666)) == -1) {
        if (fd == -1) {
          printf("shm_open failed: %s\n", strerror(errno));
        }
        if (++retries > 300)
          throw std::runtime_error("Timeout waiting for shm creation");
        usleep(100000);  // 100ms
      }

      data = (BarrierData*)mmap(NULL, sizeof(BarrierData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (data == MAP_FAILED)
        throw std::runtime_error("mmap failed");
      retries = 0;
      while (data->initialized_magic != BARRIER_MAGIC) {
        if (++retries > 300)
          throw std::runtime_error("Timeout waiting for barrier init");
        usleep(10000);  // 10ms
      }
    }
  }

  ~ProcessBarrier() {
    if (is_owner) {
      if (data && data != MAP_FAILED) {
        data->initialized_magic = 0;
        pthread_barrier_destroy(&data->barrier);
      }
      shm_unlink(name.c_str());
    }
    if (data && data != MAP_FAILED) {
      munmap(data, sizeof(BarrierData));
    }
    if (fd != -1) {
      close(fd);
    }
  }

  void wait() {
    if (data->initialized_magic != BARRIER_MAGIC) {
      throw std::runtime_error("Barrier corrupted or not initialized");
    }
    pthread_barrier_wait(&data->barrier);
  }
};

}  // namespace pccl
