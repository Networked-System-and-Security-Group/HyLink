/*
 * GPU DMA bandwidth test — all GPUs simultaneously.
 *
 * Runs THREE scenarios back-to-back on the same streams, with all GPUs
 * submitting in lock-step (iter-outer / GPU-inner) so they flood PCIe
 * concurrently:
 *   (1) PER-GPU NUMA-LOCAL (D2H-first): each GPU owns its own pinned
 *       host chunks on its physical NUMA node. In each iter, D2H is
 *       submitted before H2D.
 *   (2) SHARED CHUNK (D2H-first): same as (1) but all GPUs share ONE
 *       D2H destination chunk and ONE H2D source chunk, isolating
 *       DMA-vs-LLC interaction (DDIO/writeback/shared-read-hit).
 *   (3) PER-GPU NUMA-LOCAL (H2D-first): identical to (1) but the
 *       submission order inside each iter is REVERSED — H2D first,
 *       then D2H. Tests the "whoever starts first wins PCIe
 *       arbitration" hypothesis. If the single-iter transient is
 *       governed by submission order, (3) should mirror (1): H2D
 *       near-solo peak, D2H halved.
 *
 * Compare (1) vs (3) to confirm submission-order dictates PCIe arbitration
 * at transient scale. Compare (1) vs (2) for cache/NUMA effects.
 */

#include "test_common.hpp"
#include "topology/numa.hpp"
#include "topology/topology.hpp"

#ifndef DMA_TOTAL_BYTES
#define DMA_TOTAL_BYTES (512ULL * 1024 * 1024)
#endif
#define DMA_CHUNK_SIZE (1ULL * 1024 * 1024)
#ifndef ITERATIONS
#define ITERATIONS 1
#endif

static void dmaChunkedCopy(void* dst, const void* src, size_t total, MemcpyDirection dir, devStream stream) {
  for (size_t off = 0; off < total; off += DMA_CHUNK_SIZE) {
    size_t sz = std::min<size_t>(DMA_CHUNK_SIZE, total - off);
    devMemcpyAsync(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, sz, dir, stream);
  }
}

static double bw_gbps(size_t bytes, float ms) {
  return (bytes / 1e9) / (ms / 1e3);
}

static void printTable(const char* title,
                       int num_gpus,
                       const LocalTopology& topo,
                       const std::vector<float>& d2h_ms,
                       const std::vector<float>& h2d_ms,
                       size_t d2h_bytes_per_gpu,
                       size_t h2d_bytes_per_gpu) {
  std::printf("=== %s ===\n", title);
  std::printf("%-5s  %-5s  %12s  %12s  %12s\n",
              "GPU", "NUMA", "D2H(GB/s)", "H2D(GB/s)", "Bidir(GB/s)");
  std::printf("%-5s  %-5s  %12s  %12s  %12s\n",
              "-----", "-----", "------------", "------------", "------------");

  double total_d2h = 0, total_h2d = 0, total_bidir = 0;
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    int physical_numa = topo.getPhysicalNUMA(topo.getNUMAForGPU(gpu));
    double d2h_bw = bw_gbps(d2h_bytes_per_gpu, d2h_ms[gpu]);
    double h2d_bw = bw_gbps(h2d_bytes_per_gpu, h2d_ms[gpu]);
    double bidir = d2h_bw + h2d_bw;
    total_d2h += d2h_bw;
    total_h2d += h2d_bw;
    total_bidir += bidir;
    std::printf("%-5d  %-5d  %12.2f  %12.2f  %12.2f\n", gpu, physical_numa, d2h_bw, h2d_bw, bidir);
  }
  std::printf("%-5s  %-5s  %12s  %12s  %12s\n",
              "-----", "-----", "------------", "------------", "------------");
  std::printf("%-5s  %-5s  %12.2f  %12.2f  %12.2f\n\n",
              "Total", "-", total_d2h, total_h2d, total_bidir);
}

int main(int argc, char** argv) {
  CHECK_DEV(devInit());

  int num_gpus = 2;
  if (argc > 1)
    num_gpus = std::atoi(argv[1]);

  LocalTopology topo = discoverLocalTopology(num_gpus);

  /* Per-GPU resources (device-side + streams + timers) */
  std::vector<void*> dev_buf(num_gpus);
  std::vector<void*> dev_buf2(num_gpus);
  std::vector<devStream> stream_d2h(num_gpus);
  std::vector<devStream> stream_h2d(num_gpus);

  /* --- Scenario 1: per-GPU NUMA-local host chunks --- */
  std::vector<void*> host_buf_local(num_gpus);   // D2H dst, per-GPU
  std::vector<void*> host_buf2_local(num_gpus);  // H2D src, per-GPU

  /* --- Scenario 2: shared host chunks (all GPUs contend) --- */
  int shared_numa = topo.getPhysicalNUMA(topo.getNUMAForGPU(0));
  void* shared_host_d2h = nullptr;
  void* shared_host_h2d = nullptr;
  pinThreadToNuma(shared_numa);
  CHECK_DEV(devMallocHost(&shared_host_d2h, DMA_TOTAL_BYTES));
  CHECK_DEV(devMallocHost(&shared_host_h2d, DMA_TOTAL_BYTES));
  std::memset(shared_host_d2h, 0x42, DMA_TOTAL_BYTES);
  std::memset(shared_host_h2d, 0xAB, DMA_TOTAL_BYTES);
  pinThreadToNuma(-1);

  /* Separate timer sets per scenario to avoid event reuse issues */
  std::vector<BenchTimer> t_d2h_local(num_gpus),  t_h2d_local(num_gpus);
  std::vector<BenchTimer> t_d2h_shared(num_gpus), t_h2d_shared(num_gpus);
  std::vector<BenchTimer> t_d2h_rev(num_gpus),    t_h2d_rev(num_gpus);

  /* Phase A: allocate all pinned host buffers up-front.
   * devMallocHost is expensive (page-locking + driver accounting, can be
   * 100s of ms for 512MB). Keep it out of per-GPU setup so it doesn't
   * interleave with stream/timer creation or warmup. */
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    int logical_numa = topo.getNUMAForGPU(gpu);
    int physical_numa = topo.getPhysicalNUMA(logical_numa);
    pinThreadToNuma(physical_numa);
    CHECK_DEV(devMallocHost(&host_buf_local[gpu], DMA_TOTAL_BYTES));
    CHECK_DEV(devMallocHost(&host_buf2_local[gpu], DMA_TOTAL_BYTES));
    std::memset(host_buf_local[gpu], 0x42, DMA_TOTAL_BYTES);
    std::memset(host_buf2_local[gpu], 0xAB, DMA_TOTAL_BYTES);
    pinThreadToNuma(-1);
  }

  /* Phase B: per-GPU device-side setup (buffers, streams, timers) + warmup. */
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));

    CHECK_DEV(devMalloc(&dev_buf[gpu], DMA_TOTAL_BYTES));
    CHECK_DEV(devMalloc(&dev_buf2[gpu], DMA_TOTAL_BYTES));

    CHECK_DEV(devCreateStream(&stream_d2h[gpu]));
    CHECK_DEV(devCreateStream(&stream_h2d[gpu]));
    t_d2h_local[gpu].init();  t_h2d_local[gpu].init();
    t_d2h_shared[gpu].init(); t_h2d_shared[gpu].init();
    t_d2h_rev[gpu].init();    t_h2d_rev[gpu].init();

    /* Warmup — touch both sets of host buffers once */
    dmaChunkedCopy(host_buf_local[gpu], dev_buf[gpu], DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h[gpu]);
    dmaChunkedCopy(dev_buf[gpu], host_buf2_local[gpu], DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d[gpu]);
    dmaChunkedCopy(shared_host_d2h, dev_buf[gpu], DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h[gpu]);
    dmaChunkedCopy(dev_buf[gpu], shared_host_h2d, DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d[gpu]);
    CHECK_DEV(devSynchronizeStream(stream_d2h[gpu]));
    CHECK_DEV(devSynchronizeStream(stream_h2d[gpu]));
  }

  /* ── Scenario 1: PER-GPU NUMA-LOCAL (ITERATIONS rounds) ─────────── */
  /* Iter-outer / GPU-inner: all GPUs' streams get submissions interleaved
   * in wall-clock time so they start flooding PCIe roughly together.
   * With the opposite loop order, CPU command-queue back-pressure causes
   * GPU 0 to run to completion before GPU 1 even starts — the trace would
   * show serial, not concurrent, execution. */
  /* record_start needs current device == event's device (events were created
   * on each GPU during Phase B). Inside the tight submission loop we don't
   * need devSetDevice: cudaMemcpyAsync honours the stream's bound device
   * regardless of the current device, so we skip it there to minimize the
   * per-call overhead between cudaMemcpyAsync submissions. */
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_local[gpu].record_start(stream_d2h[gpu]);
    t_h2d_local[gpu].record_start(stream_h2d[gpu]);
  }
  for (int it = 0; it < ITERATIONS; ++it) {
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      dmaChunkedCopy(host_buf_local[gpu], dev_buf[gpu], DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h[gpu]);
      dmaChunkedCopy(dev_buf2[gpu], host_buf2_local[gpu], DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d[gpu]);
    }
  }
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_local[gpu].record_stop(stream_d2h[gpu]);
    t_h2d_local[gpu].record_stop(stream_h2d[gpu]);
  }
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSynchronizeStream(stream_d2h[gpu]));
    CHECK_DEV(devSynchronizeStream(stream_h2d[gpu]));
  }

  /* ── Scenario 2: SHARED CHUNK (ITERATIONS rounds) ──────────────── */
  /* Same iter-outer / GPU-inner pattern as Scenario 1. */
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_shared[gpu].record_start(stream_d2h[gpu]);
    t_h2d_shared[gpu].record_start(stream_h2d[gpu]);
  }
  for (int it = 0; it < ITERATIONS; ++it) {
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      dmaChunkedCopy(shared_host_d2h, dev_buf[gpu], DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h[gpu]);
      dmaChunkedCopy(dev_buf2[gpu], shared_host_h2d, DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d[gpu]);
    }
  }
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_shared[gpu].record_stop(stream_d2h[gpu]);
    t_h2d_shared[gpu].record_stop(stream_h2d[gpu]);
  }
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSynchronizeStream(stream_d2h[gpu]));
    CHECK_DEV(devSynchronizeStream(stream_h2d[gpu]));
  }

  /* ── Scenario 3: PER-GPU NUMA-LOCAL, H2D-FIRST submission order ─── */
  /* Same buffers and streams as Scenario 1, but submit H2D BEFORE D2H
   * in each iter. If the 1-iter asymmetry is dictated by "whoever submits
   * first captures PCIe credits", this should mirror Scenario 1: H2D
   * near-solo peak, D2H halved. */
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_rev[gpu].record_start(stream_d2h[gpu]);
    t_h2d_rev[gpu].record_start(stream_h2d[gpu]);
  }
  for (int it = 0; it < ITERATIONS; ++it) {
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      dmaChunkedCopy(dev_buf2[gpu], host_buf2_local[gpu], DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d[gpu]);
      dmaChunkedCopy(host_buf_local[gpu], dev_buf[gpu], DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h[gpu]);
    }
  }
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_rev[gpu].record_stop(stream_d2h[gpu]);
    t_h2d_rev[gpu].record_stop(stream_h2d[gpu]);
  }
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSynchronizeStream(stream_d2h[gpu]));
    CHECK_DEV(devSynchronizeStream(stream_h2d[gpu]));
  }

  /* ── Report ────────────────────────────────────────────────────── */
  std::printf("\nGPU DMA All | %zu GPUs | %zu MB payload | %zu KB DMA chunks | %d iters each\n\n",
              (size_t)num_gpus,
              (size_t)(DMA_TOTAL_BYTES / (1024 * 1024)),
              (size_t)(DMA_CHUNK_SIZE / 1024),
              (int)ITERATIONS);

  const size_t bytes_sym = (size_t)ITERATIONS * DMA_TOTAL_BYTES;

  std::vector<float> d2h_local_ms(num_gpus), h2d_local_ms(num_gpus);
  std::vector<float> d2h_shared_ms(num_gpus), h2d_shared_ms(num_gpus);
  std::vector<float> d2h_rev_ms(num_gpus),    h2d_rev_ms(num_gpus);
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    d2h_local_ms[gpu]  = t_d2h_local[gpu].elapsed_ms();
    h2d_local_ms[gpu]  = t_h2d_local[gpu].elapsed_ms();
    d2h_shared_ms[gpu] = t_d2h_shared[gpu].elapsed_ms();
    h2d_shared_ms[gpu] = t_h2d_shared[gpu].elapsed_ms();
    d2h_rev_ms[gpu]    = t_d2h_rev[gpu].elapsed_ms();
    h2d_rev_ms[gpu]    = t_h2d_rev[gpu].elapsed_ms();
  }
  printTable("Scenario 1: PER-GPU NUMA-LOCAL, D2H submitted first",
             num_gpus, topo, d2h_local_ms, h2d_local_ms, bytes_sym, bytes_sym);
  char title2[96];
  std::snprintf(title2, sizeof(title2),
                "Scenario 2: SHARED chunks (both pinned on NUMA %d), D2H first", shared_numa);
  printTable(title2, num_gpus, topo, d2h_shared_ms, h2d_shared_ms, bytes_sym, bytes_sym);
  printTable("Scenario 3: PER-GPU NUMA-LOCAL, H2D submitted first (mirror of S1)",
             num_gpus, topo, d2h_rev_ms, h2d_rev_ms, bytes_sym, bytes_sym);

  /* Cleanup */
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));
    t_d2h_local[gpu].destroy();  t_h2d_local[gpu].destroy();
    t_d2h_shared[gpu].destroy(); t_h2d_shared[gpu].destroy();
    t_d2h_rev[gpu].destroy();    t_h2d_rev[gpu].destroy();
    devDestroyStream(stream_d2h[gpu]);
    devDestroyStream(stream_h2d[gpu]);
    devFreeHost(host_buf_local[gpu]);
    devFreeHost(host_buf2_local[gpu]);
    devFree(dev_buf[gpu]);
    devFree(dev_buf2[gpu]);
  }
  devFreeHost(shared_host_d2h);
  devFreeHost(shared_host_h2d);

  std::printf("Done.\n");
  return 0;
}
