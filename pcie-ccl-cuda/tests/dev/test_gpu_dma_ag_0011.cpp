/*
 * AllGather simulation: 4-GPU host-relay bandwidth test.
 *
 * Memory layout matches PCCL's allgather_0011 IR:
 *
 *   Per NUMA node, 4 host chunks of X bytes (total 4X per NUMA):
 *     hc[n][0] : D2H dest for 1st local GPU  (also H2H source for remote NUMA)
 *     hc[n][1] : D2H dest for 2nd local GPU  (also H2H source for remote NUMA)
 *     hc[n][2] : H2H dest from remote NUMA   (shared H2D source for BOTH local GPUs)
 *     hc[n][3] : H2H dest from remote NUMA   (shared H2D source for BOTH local GPUs)
 *
 *   Host memory footprint: 4X × 2 NUMA = 8X  (e.g. 1024 MB @ X=128 MB)
 *   (vs. original test:   ~24X = 3072 MB with separate per-GPU buffers)
 *
 * Data flow per GPU (matches allgather_0011.xml):
 *   D2H : GPU writes 1×X to its own host chunk
 *   H2D : GPU reads 3×X from local NUMA — peer D2H chunk + 2 shared H2H chunks
 *   H2H : bridge copies 2×X per direction (2 chunks × X)
 *
 * All transfers start concurrently (no dependency enforcement).
 *
 * Usage: ./test_gpu_dma_ag_0011 [num_gpus [X_MB]]
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "hal/host_rt.hpp"
#include "test_common.hpp"
#include "topology/numa.hpp"
#include "topology/topology.hpp"

using namespace pccl;

#ifndef ALLGATHER_X_MB
#define ALLGATHER_X_MB 128ULL
#endif
#define DMA_CHUNK (1ULL * 1024 * 1024)
#define H2H_CHUNK (1ULL * 1024 * 1024)
static const int H2H_THREADS = 8; /* per directed link */

/* ------------------------------------------------------------------ */

static double bw_gbps(size_t bytes, double ms) {
  return (bytes / 1e9) / (ms / 1e3);
}

static void dmaChunked(void* dst, const void* src, size_t total,
                       MemcpyDirection dir, devStream s) {
  for (size_t off = 0; off < total; off += DMA_CHUNK) {
    size_t sz = std::min<size_t>(DMA_CHUNK, total - off);
    devMemcpyAsync((char*)dst + off, (const char*)src + off, sz, dir, s);
  }
}

static float* allocOnNuma(size_t bytes, int phys) {
  void* p = nullptr;
  pinThreadToNuma(phys);
  CHECK_DEV(devMallocHost(&p, bytes));
  std::memset(p, 0, bytes);
  pinThreadToNuma(-1);
  return (float*)p;
}

/* ------------------------------------------------------------------ */
/*  Resident H2H thread pool (matches PCCL H2HEngine design)          */
/* ------------------------------------------------------------------ */

struct H2HWorkItem {
  float* dst;
  const float* src;
  size_t elem_start;
  size_t elem_count;
};

struct H2HPool {
  int phys_numa;
  int nworkers;
  std::vector<std::thread> workers;
  std::mutex mu;
  std::deque<H2HWorkItem> queue;
  std::atomic<int> pending{0};   /* items not yet finished */
  std::atomic<bool> stop{false};

  void start(int n, int phys) {
    phys_numa = phys;
    nworkers = n;
    for (int i = 0; i < n; i++)
      workers.emplace_back(&H2HPool::loop, this);
  }

  void shutdown() {
    stop.store(true, std::memory_order_release);
    for (auto& w : workers)
      if (w.joinable()) w.join();
  }

  /* Submit work items and return immediately. */
  void submit(float* dst, const float* src, size_t bytes, int nthreads) {
    size_t total = bytes / sizeof(float);
    size_t per_th = (total + nthreads - 1) / nthreads;
    std::lock_guard<std::mutex> lk(mu);
    for (int t = 0; t < nthreads; t++) {
      size_t start = (size_t)t * per_th;
      if (start >= total) break;
      size_t count = std::min(per_th, total - start);
      queue.push_back({dst, src, start, count});
      pending.fetch_add(1, std::memory_order_relaxed);
    }
  }

  /* Spin until all submitted items are done. */
  void drain() {
    while (pending.load(std::memory_order_acquire) > 0)
      ;
  }

private:
  void loop() {
    pinThreadToNuma(phys_numa);
    const size_t chunk = H2H_CHUNK / sizeof(float);
    while (!stop.load(std::memory_order_acquire)) {
      H2HWorkItem item;
      bool got = false;
      {
        std::lock_guard<std::mutex> lk(mu);
        if (!queue.empty()) {
          item = queue.front();
          queue.pop_front();
          got = true;
        }
      }
      if (!got) { std::this_thread::yield(); continue; }

      for (size_t off = 0; off < item.elem_count; off += chunk) {
        size_t n = std::min(chunk, item.elem_count - off);
        hostMemcpy(item.dst + item.elem_start + off,
                   item.src + item.elem_start + off,
                   n * sizeof(float));
      }
      pending.fetch_sub(1, std::memory_order_release);
    }
    pinThreadToNuma(-1);
  }
};

struct H2HResult { double elapsed_ms = 0.0; };

/* ------------------------------------------------------------------ */

static void printRow(const char* label, const char* numa_str,
                     size_t bytes, double ms) {
  std::printf("%-22s  %-6s  %9zu  %10.2f  %10.2f\n",
              label, numa_str,
              bytes / (1024 * 1024), ms, bw_gbps(bytes, ms));
}

/* ================================================================== */

int main(int argc, char** argv) {
  CHECK_DEV(devInit());

  int    num_gpus = 4;
  size_t X        = ALLGATHER_X_MB * 1024 * 1024;
  if (argc > 1) num_gpus = std::atoi(argv[1]);
  if (argc > 2) X = (size_t)std::atoll(argv[2]) * 1024 * 1024;

  if (num_gpus != 4) {
    std::fprintf(stderr, "This test is designed for 4 GPUs (0011 topology)\n");
    return 1;
  }

  LocalTopology topo = discoverLocalTopology(num_gpus);

  if (topo.getNumNUMANodes() < 2) {
    std::fprintf(stderr, "This test requires 2 NUMA nodes (0011 topology)\n");
    return 1;
  }

  /* Map logical NUMA → physical NUMA (use topology directly, NOT sorted) */
  int phys_numa[2] = {topo.getPhysicalNUMA(0), topo.getPhysicalNUMA(1)};
  int pa = phys_numa[0];
  int pb = phys_numa[1];

  /* Map each GPU to its logical NUMA index (0 or 1) and local rank within NUMA */
  int gpu_numa[4], gpu_local[4];
  int numa_count[2] = {0, 0};
  for (int g = 0; g < num_gpus; ++g) {
    int n = topo.getNUMAForGPU(g);
    gpu_numa[g] = n;
    gpu_local[g] = numa_count[n]++;
  }

  /* Transfer sizes (same as before) */
  size_t d2h_bytes = 1 * X;   /* per GPU */
  size_t h2d_bytes = 3 * X;   /* per GPU (3 chunks × X) */
  size_t h2h_bytes = 2 * X;   /* per direction (2 chunks × X) */

  /* ================================================================ */
  /*  Shared host chunk pool (matches PCCL allgather_0011 IR layout)  */
  /*                                                                  */
  /*  hc[n][0] : D2H dest for local_rank 0   \  also H2H src         */
  /*  hc[n][1] : D2H dest for local_rank 1   /  for remote NUMA      */
  /*  hc[n][2] : H2H dest (remote chunk 0)   \  shared H2D src       */
  /*  hc[n][3] : H2H dest (remote chunk 1)   /  for both local GPUs  */
  /* ================================================================ */

  /* Find one representative GPU per logical NUMA for device context */
  int numa_rep_gpu[2] = {-1, -1};
  for (int g = 0; g < num_gpus; ++g) {
    int n = gpu_numa[g];
    if (numa_rep_gpu[n] < 0) numa_rep_gpu[n] = g;
  }

  float* hc[2][4];
  for (int n = 0; n < 2; n++) {
    CHECK_DEV(devSetDevice(numa_rep_gpu[n]));  /* pin to a GPU on this NUMA */
    for (int c = 0; c < 4; c++) {
      hc[n][c] = allocOnNuma(X, phys_numa[n]);
    }
  }

  /* ---- GPU device resources ---- */
  std::vector<void*>     dev_d2h(num_gpus), dev_h2d(num_gpus);
  std::vector<devStream> s_d2h(num_gpus),   s_h2d(num_gpus);
  std::vector<BenchTimer> t_d2h(num_gpus),  t_h2d(num_gpus);

  /*
   * H2D source pointers per GPU: 3 host chunks from local NUMA.
   *
   * From allgather_0011.xml:
   *   GPU on NUMA n, local_rank lr:
   *     h2d_src[0] = hc[n][1 - lr]  (peer's D2H chunk on same NUMA)
   *     h2d_src[1] = hc[n][2]       (H2H'd remote chunk — shared by both GPUs)
   *     h2d_src[2] = hc[n][3]       (H2H'd remote chunk — shared by both GPUs)
   */
  void* h2d_src[4][3];
  for (int g = 0; g < num_gpus; ++g) {
    int n  = gpu_numa[g];
    int lr = gpu_local[g];
    h2d_src[g][0] = hc[n][1 - lr];   /* peer D2H chunk */
    h2d_src[g][1] = hc[n][2];        /* shared H2H chunk */
    h2d_src[g][2] = hc[n][3];        /* shared H2H chunk */
  }

  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));

    CHECK_DEV(devMalloc(&dev_d2h[g], d2h_bytes));
    CHECK_DEV(devMalloc(&dev_h2d[g], h2d_bytes));

    CHECK_DEV(devCreateStream(&s_d2h[g]));
    CHECK_DEV(devCreateStream(&s_h2d[g]));
    t_d2h[g].init();
    t_h2d[g].init();

    int n  = gpu_numa[g];
    int lr = gpu_local[g];

    /* Warmup — use the actual shared-chunk addresses */
    dmaChunked(hc[n][lr], dev_d2h[g], d2h_bytes, MemcpyDirection::D2H, s_d2h[g]);
    CHECK_DEV(devSynchronizeStream(s_d2h[g]));
    for (int i = 0; i < 3; ++i) {
      dmaChunked((char*)dev_h2d[g] + i * X, h2d_src[g][i], X, MemcpyDirection::H2D, s_h2d[g]);
    }
    CHECK_DEV(devSynchronizeStream(s_h2d[g]));
  }

  /* ---- Resident H2H thread pools (one per NUMA, pre-pinned) ---- */
  H2HPool h2h_pool[2];
  h2h_pool[0].start(H2H_THREADS, pa);
  h2h_pool[1].start(H2H_THREADS, pb);

  /* H2H warmup */
  h2h_pool[1].submit(hc[1][2], hc[0][0], X, H2H_THREADS / 2);
  h2h_pool[1].submit(hc[1][3], hc[0][1], X, H2H_THREADS / 2);
  h2h_pool[1].drain();
  h2h_pool[0].submit(hc[0][2], hc[1][0], X, H2H_THREADS / 2);
  h2h_pool[0].submit(hc[0][3], hc[1][1], X, H2H_THREADS / 2);
  h2h_pool[0].drain();

  /* ================================================================ */
  /*  Concurrent launch                                               */
  /* ================================================================ */

  auto wall_start = std::chrono::steady_clock::now();

  /* H2H — submit to resident pools (zero thread-creation overhead).
   * Each direction: 2 copies × X bytes, H2H_THREADS/2 workers each. */
  auto h2h_start = std::chrono::steady_clock::now();
  /* A→B: NUMA0 chunks → NUMA1 H2H dest (pool on NUMA1) */
  h2h_pool[1].submit(hc[1][2], hc[0][0], X, H2H_THREADS / 2);
  h2h_pool[1].submit(hc[1][3], hc[0][1], X, H2H_THREADS / 2);
  /* B→A: NUMA1 chunks → NUMA0 H2H dest (pool on NUMA0) */
  h2h_pool[0].submit(hc[0][2], hc[1][0], X, H2H_THREADS / 2);
  h2h_pool[0].submit(hc[0][3], hc[1][1], X, H2H_THREADS / 2);

  /* GPU DMA — enqueue all (async) */
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    int n  = gpu_numa[g];
    int lr = gpu_local[g];

    t_d2h[g].record_start(s_d2h[g]);
    dmaChunked(hc[n][lr], dev_d2h[g], d2h_bytes, MemcpyDirection::D2H, s_d2h[g]);
    t_d2h[g].record_stop(s_d2h[g]);

    t_h2d[g].record_start(s_h2d[g]);
    for (int i = 0; i < 3; ++i) {
      dmaChunked((char*)dev_h2d[g] + i * X, h2d_src[g][i], X, MemcpyDirection::H2D, s_h2d[g]);
    }
    t_h2d[g].record_stop(s_h2d[g]);
  }

  /* Sync GPU */
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    CHECK_DEV(devSynchronizeStream(s_d2h[g]));
    CHECK_DEV(devSynchronizeStream(s_h2d[g]));
  }
  /* Sync H2H */
  h2h_pool[0].drain();
  h2h_pool[1].drain();
  auto h2h_end = std::chrono::steady_clock::now();
  double h2h_atob_ms = std::chrono::duration<double, std::milli>(h2h_end - h2h_start).count();
  double h2h_btoa_ms = h2h_atob_ms;  /* both directions ran concurrently */

  auto wall_end = std::chrono::steady_clock::now();
  double wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  /* ================================================================ */
  /*  Report                                                          */
  /* ================================================================ */

  std::printf("=== AllGather Simulation [PCCL layout]  (X=%zu MB, %d GPUs) ===\n",
              X / (1024 * 1024), num_gpus);
  std::printf("  D2H per GPU : 1×X = %zu MB\n",  d2h_bytes / (1024 * 1024));
  std::printf("  H2D per GPU : 3×X = %zu MB  (3 chunks from local NUMA)\n", h2d_bytes / (1024 * 1024));
  std::printf("  H2H per dir : 2×X = %zu MB  (2 chunks × %d threads)\n\n",
              h2h_bytes / (1024 * 1024), H2H_THREADS);
  std::printf("  Host memory : %zu MB per NUMA (4 chunks × %zu MB)\n\n",
              4 * X / (1024 * 1024), X / (1024 * 1024));

  std::printf("%-22s  %-6s  %9s  %10s  %10s\n",
              "Transfer", "NUMA", "Data(MB)", "Time(ms)", "BW(GB/s)");
  std::printf("%-22s  %-6s  %9s  %10s  %10s\n",
              "----------------------", "------",
              "---------", "----------", "----------");

  for (int g = 0; g < num_gpus; ++g) {
    char label[32], numa_s[16];
    std::snprintf(label,  sizeof(label),  "GPU%d D2H", g);
    std::snprintf(numa_s, sizeof(numa_s), "%d", phys_numa[gpu_numa[g]]);
    printRow(label, numa_s, d2h_bytes, t_d2h[g].elapsed_ms());
  }

  for (int g = 0; g < num_gpus; ++g) {
    char label[32], numa_s[16];
    std::snprintf(label,  sizeof(label),  "GPU%d H2D (3 chunks)", g);
    std::snprintf(numa_s, sizeof(numa_s), "%d", phys_numa[gpu_numa[g]]);
    printRow(label, numa_s, h2d_bytes, t_h2d[g].elapsed_ms());
  }

  double atob_ms = h2h_atob_ms;
  double btoa_ms = h2h_btoa_ms;
  {
    char lab[32];
    std::snprintf(lab, sizeof(lab), "NUMA%d->NUMA%d H2H", pa, pb);
    printRow(lab, "-", h2h_bytes, atob_ms);
    std::snprintf(lab, sizeof(lab), "NUMA%d->NUMA%d H2H", pb, pa);
    printRow(lab, "-", h2h_bytes, btoa_ms);
  }

  std::printf("\nTotal wall time: %.2f ms\n", wall_ms);

  /* Aggregate stats */
  size_t total_d2h = (size_t)num_gpus * d2h_bytes;
  size_t total_h2d = (size_t)num_gpus * h2d_bytes;
  size_t total_h2h = 2 * h2h_bytes;
  size_t host_footprint = 2 * 4 * X;
  std::printf("Aggregate D2H  : %zu MB\n", total_d2h / (1024 * 1024));
  std::printf("Aggregate H2D  : %zu MB\n", total_h2d / (1024 * 1024));
  std::printf("Aggregate H2H  : %zu MB  (2 directed links)\n",
              total_h2h / (1024 * 1024));
  std::printf("Aggregate total: %zu MB\n",
              (total_d2h + total_h2d + total_h2h) / (1024 * 1024));
  std::printf("Host footprint : %zu MB  (vs ~%zu MB with separate buffers)\n",
              host_footprint / (1024 * 1024),
              (total_d2h + total_h2d + total_h2h) / (1024 * 1024));

  /* ---- Cleanup ---- */
  h2h_pool[0].shutdown();
  h2h_pool[1].shutdown();
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    t_d2h[g].destroy();
    t_h2d[g].destroy();
    devDestroyStream(s_d2h[g]);
    devDestroyStream(s_h2d[g]);
    devFree(dev_d2h[g]);
    devFree(dev_h2d[g]);
  }
  for (int n = 0; n < 2; n++) {
    for (int c = 0; c < 4; c++) {
      devFreeHost(hc[n][c]);
    }
  }

  std::printf("Done.\n");
  return 0;
}
