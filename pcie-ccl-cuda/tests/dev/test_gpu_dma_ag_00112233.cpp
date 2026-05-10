/*
 * AllGather simulation: 8-GPU, 4-NUMA (allgather_00112233) bandwidth test.
 *
 * Models the concurrent data movement in allgather_00112233.xml for
 * per-GPU slice size X (2 GPUs per NUMA, 4 NUMA nodes, 8 ranks total):
 *
 *   D2H : each of 8 GPUs sends    1×X  to its local NUMA host
 *   H2D : each of 8 GPUs receives 7×X  from its local NUMA host
 *   H2H : each directed inter-NUMA link carries (gpus_per_numa)×X = 2×X
 *         (nn*(nn-1) directed links, all launched concurrently)
 *
 * Cross-NUMA H2H traffic (4-NUMA): 12 × 2×X = 24×X total.
 * Each of the 6 bidirectional NUMA links is fully saturated.
 *
 * All transfers start concurrently. Per-transfer data/time/BW and total
 * wall-clock completion time are reported.
 *
 * Usage: ./test_gpu_dma_ag_00112233 [num_gpus [X_MB]]
 *   num_gpus : number of GPUs (default 8)
 *   X_MB     : per-GPU slice in MB (default 64)
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "hal/host_rt.hpp"
#include "test_common.hpp"
#include "topology/numa.hpp"
#include "topology/topology.hpp"

using namespace pccl;

#ifndef AG00112233_X_MB
#define AG00112233_X_MB 64ULL
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

struct H2HResult { double elapsed_ms = 0.0; };

static void h2hWorker(float* dst, const float* src,
                      size_t elem_start, size_t elem_count, int pin_numa) {
  pinThreadToNuma(pin_numa);
  const size_t chunk = H2H_CHUNK / sizeof(float);
  for (size_t off = 0; off < elem_count; off += chunk) {
    size_t n = std::min(chunk, elem_count - off);
    hostMemcpy(dst + elem_start + off, src + elem_start + off, n * sizeof(float));
  }
  pinThreadToNuma(-1);
}

static H2HResult doH2H(float* dst, const float* src, size_t bytes,
                       int nthreads, int dst_phys) {
  size_t total  = bytes / sizeof(float);
  size_t per_th = (total + nthreads - 1) / nthreads;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> tv;
  tv.reserve(nthreads);
  for (int t = 0; t < nthreads; ++t) {
    size_t start = (size_t)t * per_th;
    if (start >= total) break;
    size_t count = std::min(per_th, total - start);
    tv.emplace_back(h2hWorker, dst, src, start, count, dst_phys);
  }
  for (auto& th : tv) th.join();
  auto t1 = std::chrono::steady_clock::now();
  return { std::chrono::duration<double, std::milli>(t1 - t0).count() };
}

/* ------------------------------------------------------------------ */

static void printRow(const char* label, const char* numa_str,
                     size_t bytes, double ms) {
  std::printf("%-28s  %-6s  %9zu  %10.2f  %10.2f\n",
              label, numa_str,
              bytes / (1024 * 1024), ms, bw_gbps(bytes, ms));
}

/* ================================================================== */

int main(int argc, char** argv) {
  CHECK_DEV(devInit());

  int    num_gpus = 8;
  size_t X        = AG00112233_X_MB * 1024 * 1024;
  if (argc > 1) num_gpus = std::atoi(argv[1]);
  if (argc > 2) X = (size_t)std::atoll(argv[2]) * 1024 * 1024;

  LocalTopology topo = discoverLocalTopology(num_gpus);

  /* Collect distinct physical NUMA nodes (sorted) */
  std::set<int> ns;
  for (int i = 0; i < topo.getNumNUMANodes(); ++i)
    ns.insert(topo.getPhysicalNUMA(i));
  std::vector<int> numas(ns.begin(), ns.end());
  int nn = (int)numas.size();

  /* GPUs per NUMA (determines H2H payload per directed link) */
  int gpus_per_numa = (nn > 0) ? (num_gpus / nn) : num_gpus;

  /* Transfer sizes */
  size_t d2h_bytes = 1 * X;                     /* per GPU: own slice */
  size_t h2d_bytes = (num_gpus - 1) * X;        /* per GPU: all remote slices */
  size_t h2h_bytes = (size_t)gpus_per_numa * X; /* per directed link */
  int    num_links  = nn * (nn - 1);             /* ordered NUMA pairs */

  std::printf("=== AllGather-00112233 Simulation"
              "  (X=%zu MB, %d GPUs, %d NUMAs, %d GPUs/NUMA) ===\n",
              X / (1024 * 1024), num_gpus, nn, gpus_per_numa);
  std::printf("  D2H per GPU  : 1×X = %zu MB\n", d2h_bytes / (1024 * 1024));
  std::printf("  H2D per GPU  : %d×X = %zu MB\n",
              num_gpus - 1, h2d_bytes / (1024 * 1024));
  if (nn >= 2)
    std::printf("  H2H per link : %d×X = %zu MB  (%d threads/link, %d links)\n\n",
                gpus_per_numa, h2h_bytes / (1024 * 1024), H2H_THREADS, num_links);
  else
    std::printf("  H2H : skipped (single NUMA)\n\n");

  /* ---- GPU resources ---- */
  std::vector<void*>      dev_d2h(num_gpus), host_d2h(num_gpus);
  std::vector<void*>      dev_h2d(num_gpus), host_h2d(num_gpus);
  std::vector<devStream>  s_d2h(num_gpus),   s_h2d(num_gpus);
  std::vector<BenchTimer> t_d2h(num_gpus),   t_h2d(num_gpus);

  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    int phys = topo.getPhysicalNUMA(topo.getNUMAForGPU(g));

    CHECK_DEV(devMalloc(&dev_d2h[g], d2h_bytes));
    CHECK_DEV(devMalloc(&dev_h2d[g], h2d_bytes));

    pinThreadToNuma(phys);
    CHECK_DEV(devMallocHost(&host_d2h[g], d2h_bytes));
    CHECK_DEV(devMallocHost(&host_h2d[g], h2d_bytes));
    std::memset(host_d2h[g], 0x42, d2h_bytes);
    std::memset(host_h2d[g], 0xAB, h2d_bytes);
    pinThreadToNuma(-1);

    CHECK_DEV(devCreateStream(&s_d2h[g]));
    CHECK_DEV(devCreateStream(&s_h2d[g]));
    t_d2h[g].init();
    t_h2d[g].init();

    /* Warmup */
    dmaChunked(host_d2h[g], dev_d2h[g], d2h_bytes, MemcpyDirection::D2H, s_d2h[g]);
    CHECK_DEV(devSynchronizeStream(s_d2h[g]));
    dmaChunked(dev_h2d[g], host_h2d[g], h2d_bytes, MemcpyDirection::H2D, s_h2d[g]);
    CHECK_DEV(devSynchronizeStream(s_h2d[g]));
  }

  /* ---- H2H resources ---- */
  /*
   * Each NUMA has one outgoing source buffer (shared by all outgoing links).
   * Each directed link has its own destination buffer on the target NUMA,
   * so concurrent incoming writes do not compete on the same cache lines.
   */
  struct H2HLink {
    int src_ni, dst_ni; /* indices into numas[] */
    float* src;         /* allocated on numas[src_ni] */
    float* dst;         /* allocated on numas[dst_ni] */
    H2HResult result;
  };

  std::vector<float*>  h2h_src(nn, nullptr);
  std::vector<H2HLink> links;

  if (nn >= 2) {
    for (int ni = 0; ni < nn; ++ni)
      h2h_src[ni] = allocOnNuma(h2h_bytes, numas[ni]);

    links.reserve(num_links);
    for (int a = 0; a < nn; ++a) {
      for (int b = 0; b < nn; ++b) {
        if (a == b) continue;
        float* dst = allocOnNuma(h2h_bytes, numas[b]);
        H2HLink lk;
        lk.src_ni = a;
        lk.dst_ni = b;
        lk.src = h2h_src[a];
        lk.dst = dst;
        links.push_back(lk);
      }
    }

    /* Warmup H2H (all links, sequential) */
    for (auto& lk : links)
      doH2H(lk.dst, lk.src, h2h_bytes, H2H_THREADS, numas[lk.dst_ni]);
  }

  /* ================================================================ */
  /*  Concurrent launch                                               */
  /* ================================================================ */

  auto wall_start = std::chrono::steady_clock::now();

  /* H2H — launch all directed links concurrently first */
  std::vector<std::thread> h2h_threads;
  h2h_threads.reserve(links.size());
  for (size_t li = 0; li < links.size(); ++li) {
    float* dst    = links[li].dst;
    float* src    = links[li].src;
    int    dphys  = numas[links[li].dst_ni];
    h2h_threads.emplace_back([&links, li, dst, src, h2h_bytes, dphys]() {
      links[li].result = doH2H(dst, src, h2h_bytes, H2H_THREADS, dphys);
    });
  }

  /* GPU DMA — enqueue all streams (async) */
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    t_d2h[g].record_start(s_d2h[g]);
    t_h2d[g].record_start(s_h2d[g]);
    dmaChunked(host_d2h[g], dev_d2h[g], d2h_bytes, MemcpyDirection::D2H, s_d2h[g]);
    dmaChunked(dev_h2d[g], host_h2d[g], h2d_bytes, MemcpyDirection::H2D, s_h2d[g]);
    t_d2h[g].record_stop(s_d2h[g]);
    t_h2d[g].record_stop(s_h2d[g]);
  }

  /* Sync GPU streams */
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    CHECK_DEV(devSynchronizeStream(s_d2h[g]));
    CHECK_DEV(devSynchronizeStream(s_h2d[g]));
  }
  /* Sync H2H threads */
  for (auto& th : h2h_threads) th.join();

  auto wall_end = std::chrono::steady_clock::now();
  double wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  /* ================================================================ */
  /*  Report                                                          */
  /* ================================================================ */

  std::printf("%-28s  %-6s  %9s  %10s  %10s\n",
              "Transfer", "NUMA", "Data(MB)", "Time(ms)", "BW(GB/s)");
  std::printf("%-28s  %-6s  %9s  %10s  %10s\n",
              "----------------------------", "------",
              "---------", "----------", "----------");

  for (int g = 0; g < num_gpus; ++g) {
    int phys = topo.getPhysicalNUMA(topo.getNUMAForGPU(g));
    char label[40], numa_s[16];
    std::snprintf(label,  sizeof(label),  "GPU%d D2H", g);
    std::snprintf(numa_s, sizeof(numa_s), "%d", phys);
    printRow(label, numa_s, d2h_bytes, t_d2h[g].elapsed_ms());
  }

  for (int g = 0; g < num_gpus; ++g) {
    int phys = topo.getPhysicalNUMA(topo.getNUMAForGPU(g));
    char label[40], numa_s[16];
    std::snprintf(label,  sizeof(label),  "GPU%d H2D", g);
    std::snprintf(numa_s, sizeof(numa_s), "%d", phys);
    printRow(label, numa_s, h2d_bytes, t_h2d[g].elapsed_ms());
  }

  for (auto& lk : links) {
    char label[48];
    std::snprintf(label, sizeof(label), "NUMA%d->NUMA%d H2H",
                  numas[lk.src_ni], numas[lk.dst_ni]);
    printRow(label, "-", h2h_bytes, lk.result.elapsed_ms);
  }

  std::printf("\nTotal wall time: %.2f ms\n", wall_ms);

  /* Aggregate stats */
  size_t total_d2h = (size_t)num_gpus * d2h_bytes;
  size_t total_h2d = (size_t)num_gpus * h2d_bytes;
  size_t total_h2h = (size_t)num_links * h2h_bytes;
  std::printf("Aggregate D2H  : %zu MB\n", total_d2h / (1024 * 1024));
  std::printf("Aggregate H2D  : %zu MB\n", total_h2d / (1024 * 1024));
  std::printf("Aggregate H2H  : %zu MB  (%d directed links)\n",
              total_h2h / (1024 * 1024), num_links);
  std::printf("Aggregate total: %zu MB\n",
              (total_d2h + total_h2d + total_h2h) / (1024 * 1024));

  /* ---- Cleanup ---- */
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_DEV(devSetDevice(g));
    t_d2h[g].destroy();
    t_h2d[g].destroy();
    devDestroyStream(s_d2h[g]);
    devDestroyStream(s_h2d[g]);
    devFreeHost(host_d2h[g]);
    devFreeHost(host_h2d[g]);
    devFree(dev_d2h[g]);
    devFree(dev_h2d[g]);
  }
  for (int ni = 0; ni < nn; ++ni)
    if (h2h_src[ni]) devFreeHost(h2h_src[ni]);
  for (auto& lk : links)
    devFreeHost(lk.dst);

  std::printf("Done.\n");
  return 0;
}
