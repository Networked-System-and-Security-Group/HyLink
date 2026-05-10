/*
 * GPU DMA bandwidth test — single GPU at a time.
 *
 * For each GPU: measure D2H, H2D, and bidirectional (D2H+H2D) bandwidth
 * using NUMA-local pinned host memory (First-Touch policy).
 * Copies are issued in DMA_CHUNK_SIZE increments via devMemcpyAsync.
 */

#include "test_common.hpp"
#include "topology/numa.hpp"
#include "topology/topology.hpp"

#ifndef DMA_TOTAL_BYTES
#define DMA_TOTAL_BYTES (512ULL * 1024 * 1024)
#endif
#define DMA_CHUNK_SIZE (1ULL * 1024 * 1024)

static void dmaChunkedCopy(void* dst, const void* src, size_t total, MemcpyDirection dir, devStream stream) {
  for (size_t off = 0; off < total; off += DMA_CHUNK_SIZE) {
    size_t sz = std::min<size_t>(DMA_CHUNK_SIZE, total - off);
    devMemcpyAsync(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, sz, dir, stream);
  }
}

static double bw_gbps(size_t bytes, float ms) {
  return (bytes / 1e9) / (ms / 1e3);
}

int main(int argc, char** argv) {
  CHECK_DEV(devInit());

  int num_gpus = 2;
  if (argc > 1)
    num_gpus = std::atoi(argv[1]);

  LocalTopology topo = discoverLocalTopology(num_gpus);

  std::printf("=== GPU DMA Single (%zu MB, %zu KB chunks) ===\n\n", (size_t)(DMA_TOTAL_BYTES / (1024 * 1024)),
              (size_t)(DMA_CHUNK_SIZE / 1024));
  std::printf("%-5s  %-5s  %12s  %12s  %14s  %14s  %14s\n", "GPU", "NUMA", "D2H(GB/s)", "H2D(GB/s)", "Bidir-D2H",
              "Bidir-H2D", "Bidir-Agg");
  std::printf("%-5s  %-5s  %12s  %12s  %14s  %14s  %14s\n", "-----", "-----", "------------", "------------",
              "--------------", "--------------", "--------------");

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CHECK_DEV(devSetDevice(gpu));

    int logical_numa = topo.getNUMAForGPU(gpu);
    int physical_numa = topo.getPhysicalNUMA(logical_numa);

    /* Allocate device memory */
    void* dev_buf = nullptr;
    void* dev_buf2 = nullptr;
    CHECK_DEV(devMalloc(&dev_buf, DMA_TOTAL_BYTES));
    CHECK_DEV(devMalloc(&dev_buf2, DMA_TOTAL_BYTES));

    /* Allocate NUMA-local pinned host memory via First-Touch */
    void* host_buf = nullptr;
    void* host_buf2 = nullptr;
    CHECK_DEV(devMallocHost(&host_buf, DMA_TOTAL_BYTES));
    CHECK_DEV(devMallocHost(&host_buf2, DMA_TOTAL_BYTES));
    pinThreadToNuma(physical_numa);
    std::memset(host_buf, 0x42, DMA_TOTAL_BYTES);
    std::memset(host_buf2, 0xAB, DMA_TOTAL_BYTES);
    pinThreadToNuma(-1);

    devStream stream_d2h, stream_h2d;
    CHECK_DEV(devCreateStream(&stream_d2h));
    CHECK_DEV(devCreateStream(&stream_h2d));

    BenchTimer t_d2h, t_h2d;
    t_d2h.init();
    t_h2d.init();

    /* Warm up — cover both buffer pairs and both directions */
    dmaChunkedCopy(host_buf,  dev_buf,  DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h);
    dmaChunkedCopy(dev_buf2, host_buf2, DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d);
    CHECK_DEV(devSynchronizeStream(stream_d2h));
    CHECK_DEV(devSynchronizeStream(stream_h2d));
    dmaChunkedCopy(dev_buf,  host_buf,  DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d);
    CHECK_DEV(devSynchronizeStream(stream_h2d));

    /* D2H */
    t_d2h.record_start(stream_d2h);
    dmaChunkedCopy(host_buf, dev_buf, DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h);
    t_d2h.record_stop(stream_d2h);
    CHECK_DEV(devSynchronizeStream(stream_d2h));
    float d2h_ms = t_d2h.elapsed_ms();

    /* H2D */
    t_h2d.record_start(stream_h2d);
    dmaChunkedCopy(dev_buf, host_buf, DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d);
    t_h2d.record_stop(stream_h2d);
    CHECK_DEV(devSynchronizeStream(stream_h2d));
    float h2d_ms = t_h2d.elapsed_ms();

    /* Bidirectional: D2H on stream_d2h + H2D on stream_h2d concurrently */
    BenchTimer t_bd2h, t_bh2d;
    t_bd2h.init();
    t_bh2d.init();

    t_bd2h.record_start(stream_d2h);
    t_bh2d.record_start(stream_h2d);
    dmaChunkedCopy(host_buf, dev_buf, DMA_TOTAL_BYTES, MemcpyDirection::D2H, stream_d2h);
    dmaChunkedCopy(dev_buf2, host_buf2, DMA_TOTAL_BYTES, MemcpyDirection::H2D, stream_h2d);
    t_bd2h.record_stop(stream_d2h);
    t_bh2d.record_stop(stream_h2d);
    CHECK_DEV(devSynchronizeStream(stream_d2h));
    CHECK_DEV(devSynchronizeStream(stream_h2d));

    float bd2h_ms = t_bd2h.elapsed_ms();
    float bh2d_ms = t_bh2d.elapsed_ms();
    float bidir_wall_ms = std::max(bd2h_ms, bh2d_ms);

    double d2h_bw = bw_gbps(DMA_TOTAL_BYTES, d2h_ms);
    double h2d_bw = bw_gbps(DMA_TOTAL_BYTES, h2d_ms);
    double bd2h_bw = bw_gbps(DMA_TOTAL_BYTES, bd2h_ms);
    double bh2d_bw = bw_gbps(DMA_TOTAL_BYTES, bh2d_ms);
    double bidir_bw = bw_gbps(DMA_TOTAL_BYTES, bd2h_ms) + bw_gbps(DMA_TOTAL_BYTES, bh2d_ms);

    std::printf("%-5d  %-5d  %12.2f  %12.2f  %14.2f  %14.2f  %14.2f\n", gpu, physical_numa, d2h_bw, h2d_bw, bd2h_bw,
                bh2d_bw, bidir_bw);

    t_d2h.destroy();
    t_h2d.destroy();
    t_bd2h.destroy();
    t_bh2d.destroy();
    devDestroyStream(stream_d2h);
    devDestroyStream(stream_h2d);
    devFreeHost(host_buf);
    devFreeHost(host_buf2);
    devFree(dev_buf);
    devFree(dev_buf2);
  }

  std::printf("\nDone.\n");
  return 0;
}
