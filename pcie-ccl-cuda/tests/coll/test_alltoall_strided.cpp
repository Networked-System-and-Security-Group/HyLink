/*
 * AllToAllStrided correctness + bandwidth test.
 *
 * Usage: test_alltoall_strided <nranks>
 *
 * Rank r fills send segment j:
 *   [0..stride-1]         = 0.0f          (fast-path region)
 *   [stride..seg_count-1] = float(r * nranks + j + 1)  (PCIe region)
 *
 * After AllToAll, rank r's recv segment j:
 *   [0..stride-1]         = 0.0f
 *   [stride..seg_count-1] = float(j * nranks + r + 1)
 */

#include "test_common.hpp"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <nranks>\n", argv[0]);
    return 1;
  }
  const int nranks = std::atoi(argv[1]);
  if (nranks <= 0) {
    std::fprintf(stderr, "nranks must be positive\n");
    return 1;
  }

  CHECK_DEV(devInit());
  printNumaTopology(nranks);

  std::vector<pcclComm_t> comms(nranks);
  std::vector<devStream> streams(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_PCCL(pcclCommInitRank(&comms[r], nranks, singleProcessId(), r));
    CHECK_DEV(devCreateStream(&streams[r]));
  }

  size_t max_seg_count = 0;
  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c)
    if (STRIDE_CONFIGS[c].seg_count > max_seg_count)
      max_seg_count = STRIDE_CONFIGS[c].seg_count;
  size_t maxBytes = max_seg_count * nranks * sizeof(float);

  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMalloc(&d_send[r], maxBytes));
    CHECK_DEV(devMalloc(&d_recv[r], maxBytes));
  }

  const double busFactor = busBwFactor(COLL_ALLTOALL, nranks);

  CHECK_DEV(devSetDevice(0));
  bool allPassed = true;

  char title[64];
  std::snprintf(title, sizeof(title), "AllToAllStrided | %d GPUs", nranks);
  printBenchHeader(title);

  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c) {
    size_t seg_count = STRIDE_CONFIGS[c].seg_count;
    size_t stride = STRIDE_CONFIGS[c].stride;
    size_t totalFloats = seg_count * nranks;
    size_t bytes = totalFloats * sizeof(float);

    std::printf("\n  --- %s (seg_count=%zu, stride=%zu) ---\n", STRIDE_CONFIGS[c].label, seg_count, stride);

    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> h(totalFloats, 0.0f);
      for (int j = 0; j < nranks; ++j) {
        float val = static_cast<float>(r * nranks + j + 1);
        size_t base = static_cast<size_t>(j) * seg_count;
        for (size_t k = stride; k < seg_count; ++k)
          h[base + k] = val;
      }
      CHECK_DEV(devMemcpyAsync(d_send[r], h.data(), bytes, MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> z(totalFloats, 0.0f);
      CHECK_DEV(devMemcpyAsync(d_recv[r], z.data(), bytes, MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllToAllStrided(d_send[r], d_recv[r], totalFloats, stride, comms[r],
                                       static_cast<pcclStream_t>(streams[r].handle)));
      }
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_DEV(devSynchronizeStream(streams[r]));
      }
    }

    CHECK_DEV(devSetDevice(0));
    IterTimings timings;
    timings.reserve(MEASURED_ITERS);
    for (int iter = 0; iter < MEASURED_ITERS; ++iter) {
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_DEV(devSynchronizeStream(streams[r]));
      }
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllToAllStrided(d_send[r], d_recv[r], totalFloats, stride, comms[r],
                                       static_cast<pcclStream_t>(streams[r].handle)));
      }
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_DEV(devSynchronizeStream(streams[r]));
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      timings.add(ms);
    }

    double avg_ms = timings.avg();
    double pcieBytes = static_cast<double>(seg_count - stride) * nranks * sizeof(float);
    double alg_bw = (pcieBytes / 1e9) / (avg_ms / 1e3);
    double bus_bw = alg_bw * busFactor;

    bool correct = true;
    for (int r = 0; r < nranks && correct; ++r) {
      std::vector<float> h(totalFloats);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], bytes, MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
      for (int j = 0; j < nranks && correct; ++j) {
        size_t base = static_cast<size_t>(j) * seg_count;
        correct = verifyBuffer(h.data() + base, stride, 0.0f, r, "alltoall_strided fast-path");
        if (!correct)
          break;
        float expected = static_cast<float>(j * nranks + r + 1);
        correct = verifyBuffer(h.data() + base + stride, seg_count - stride, expected, r, "alltoall_strided pcie");
      }
    }
    if (!correct)
      allPassed = false;

    printBenchRow(STRIDE_CONFIGS[c].label, avg_ms, alg_bw, bus_bw, correct);
    printPercentileTable(timings, pcieBytes, busFactor);
  }

  CHECK_DEV(devSetDevice(0));
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    devFree(d_send[r]);
    devFree(d_recv[r]);
    devDestroyStream(streams[r]);
    pcclCommDestroy(comms[r]);
  }
  pcclFinalize();

  std::printf("\ntest_alltoall_strided (%d GPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
