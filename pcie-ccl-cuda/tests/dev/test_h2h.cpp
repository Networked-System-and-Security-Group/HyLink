/*
 * H2H (Host-to-Host) bandwidth test.
 *
 * Benchmarks hostMemcpy (copy) and hostReduce (reduce) under:
 *   - Intra-NUMA: src and dst on the same NUMA node
 *   - Cross-NUMA: src and dst on different NUMA nodes
 * with {1, 2, 4, 8, 16} concurrent threads.
 *
 * Transfers are split into H2H_CHUNK_SIZE (1 MB) increments, matching the
 * SUBCHUNK_SLICE_SIZE used by H2HEngine in production.
 * Each configuration runs WARMUP_ITERS warm-up passes then MEASURED_ITERS
 * timed passes; the average bandwidth is reported.
 *
 * Memory is allocated with devMallocHost *inside* pinThreadToNuma so that
 * the kernel assigns pages to the intended NUMA node on first touch.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#include "hal/host_rt.hpp"
#include "test_common.hpp"
#include "topology/numa.hpp"

using namespace pccl;

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#ifndef H2H_TOTAL_BYTES
#define H2H_TOTAL_BYTES (512ULL * 1024 * 1024)
#endif
#define H2H_CHUNK_SIZE (1ULL * 1024 * 1024)

static const int H2H_WARMUP_ITERS   = 2;
static const int H2H_MEASURED_ITERS = 10;
static const int THREAD_COUNTS[]    = {1, 2, 4, 8, 16};
static const int NUM_THREAD_COUNTS  = (int)(sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0]));

/* ------------------------------------------------------------------ */
/*  Op types                                                           */
/* ------------------------------------------------------------------ */

enum class OpType { COPY, COPYADD };

static const char* opName(OpType op) {
  return op == OpType::COPY ? "copy" : "copyadd";
}

/* ------------------------------------------------------------------ */
/*  NUMA detection (no GPU required)                                   */
/* ------------------------------------------------------------------ */

static int detectNumaCount() {
  int count = 0;
  for (int n = 0; n < 8; ++n) {
    char path[64];
    std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", n);
    if (access(path, F_OK) == 0)
      count = n + 1;
    else
      break;
  }
  return std::max(1, count);
}

/* ------------------------------------------------------------------ */
/*  NUMA-aware page-locked allocation                                  */
/* ------------------------------------------------------------------ */

/* devMallocHost must be called inside the pinThreadToNuma block so
 * that the kernel maps pages to the requested NUMA node on first touch. */
static float* allocOnNuma(size_t bytes, int physical_numa) {
  void* p = nullptr;
  pinThreadToNuma(physical_numa);
  CHECK_DEV(devMallocHost(&p, bytes));
  std::memset(p, 0x42, bytes);  /* first-touch while pinned */
  pinThreadToNuma(-1);
  return static_cast<float*>(p);
}

/* ------------------------------------------------------------------ */
/*  Per-thread work: process elem_count elements in 1 MB chunks       */
/* ------------------------------------------------------------------ */

static void threadWork(float* dst, const float* src,
                       size_t elem_start, size_t elem_count, OpType op) {
  const size_t chunk_elems = H2H_CHUNK_SIZE / sizeof(float);
  for (size_t off = 0; off < elem_count; off += chunk_elems) {
    size_t n   = std::min(chunk_elems, elem_count - off);
    size_t pos = elem_start + off;
    if (op == OpType::COPY)
      hostMemcpy(dst + pos, src + pos, n * sizeof(float));
    else
      hostReduce(dst + pos, dst + pos, src + pos, n);
  }
}

/* ------------------------------------------------------------------ */
/*  Benchmark: run num_threads workers, return average GB/s           */
/* ------------------------------------------------------------------ */

static double benchH2H(float* dst, const float* src, size_t total_bytes,
                       int num_threads, OpType op, int dst_physical_numa) {
  size_t total_elems      = total_bytes / sizeof(float);
  size_t elems_per_thread = (total_elems + num_threads - 1) / num_threads;

  /* Threads are pinned to dst_physical_numa, mirroring H2HEngine workers. */
  auto runOnce = [&]() {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      size_t start = (size_t)t * elems_per_thread;
      if (start >= total_elems)
        break;
      size_t count = std::min(elems_per_thread, total_elems - start);
      threads.emplace_back([=]() {
        pinThreadToNuma(dst_physical_numa);
        threadWork(dst, src, start, count, op);
        pinThreadToNuma(-1);
      });
    }
    for (auto& th : threads)
      th.join();
  };

  /* Warmup */
  for (int i = 0; i < H2H_WARMUP_ITERS; ++i)
    runOnce();

  /* Timed iterations */
  double total_ms = 0.0;
  for (int i = 0; i < H2H_MEASURED_ITERS; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    runOnce();
    auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  double avg_ms = total_ms / H2H_MEASURED_ITERS;
  return (total_bytes / 1e9) / (avg_ms / 1e3);
}

/* ------------------------------------------------------------------ */
/*  Print one section: all (op, thread_count) combinations            */
/* ------------------------------------------------------------------ */

static void runSection(float* dst, const float* src, int dst_physical_numa) {
  std::printf("%-8s  %7s  %10s\n", "Op", "Threads", "BW(GB/s)");
  std::printf("%-8s  %7s  %10s\n", "--------", "-------", "----------");

  for (int oi = 0; oi < 2; ++oi) {
    OpType op = (oi == 0) ? OpType::COPY : OpType::COPYADD;
    for (int ti = 0; ti < NUM_THREAD_COUNTS; ++ti) {
      int nthreads = THREAD_COUNTS[ti];
      double bw = benchH2H(dst, src, H2H_TOTAL_BYTES, nthreads, op, dst_physical_numa);
      std::printf("%-8s  %7d  %10.2f\n", opName(op), nthreads, bw);
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int /*argc*/, char** /*argv*/) {
  CHECK_DEV(devInit());
  CHECK_DEV(devSetDevice(0));

  int num_numas = detectNumaCount();

  std::printf("=== H2H Bandwidth (%zu MB, %zu KB chunks, warmup=%d, iters=%d) ===\n\n",
              (size_t)(H2H_TOTAL_BYTES / (1024 * 1024)),
              (size_t)(H2H_CHUNK_SIZE / 1024),
              H2H_WARMUP_ITERS, H2H_MEASURED_ITERS);
  std::printf("Detected %d NUMA node(s)\n\n", num_numas);

  /* ---- Intra-NUMA ---- */
  {
    int phys = 0;  /* NUMA 0 for both src and dst */
    std::printf("[Intra-NUMA: src=NUMA%d  dst=NUMA%d]\n\n", phys, phys);

    float* src = allocOnNuma(H2H_TOTAL_BYTES, phys);
    float* dst = allocOnNuma(H2H_TOTAL_BYTES, phys);

    runSection(dst, src, phys);

    devFreeHost(src);
    devFreeHost(dst);
    std::printf("\n");
  }

  /* ---- Cross-NUMA ---- */
  if (num_numas >= 2) {
    int src_phys = 0;
    int dst_phys = 1;
    std::printf("[Cross-NUMA: src=NUMA%d  dst=NUMA%d]\n\n", src_phys, dst_phys);

    float* src = allocOnNuma(H2H_TOTAL_BYTES, src_phys);
    float* dst = allocOnNuma(H2H_TOTAL_BYTES, dst_phys);

    runSection(dst, src, dst_phys);

    devFreeHost(src);
    devFreeHost(dst);
    std::printf("\n");
  } else {
    std::printf("[Cross-NUMA: skipped — only 1 NUMA node detected]\n\n");
  }

  std::printf("Done.\n");
  return 0;
}
