/*
 * AllToAllStrided correctness test.
 *
 * Usage: test_alltoall_strided <nranks>
 *
 * Rank r fills send segment j:
 *   [0..stride-1]         = 0.0f          (reserved prefix)
 *   [stride..seg_count-1] = float(r * nranks + j + 1)  (active region)
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

  CHECK_DEV(devSetDevice(0));
  bool allPassed = true;
  const int perf_iters = stridedPerfIters();
  const int warmup_iters = stridedPerfWarmupIters();
  const double bus_factor = stridedBusFactor(nranks);

  char title[64];
  std::snprintf(title, sizeof(title), "AllToAllStrided | %d NPUs", nranks);
  std::printf("\n=== %s correctness/perf ===\n\n", title);
  printStridedPerfHeader(perf_iters, warmup_iters);

  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c) {
    size_t seg_count = STRIDE_CONFIGS[c].seg_count;
    size_t stride = STRIDE_CONFIGS[c].stride;
    size_t totalFloats = seg_count * nranks;
    size_t bytes = totalFloats * sizeof(float);

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

    auto submit = [&]() {
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllToAllStrided(d_send[r], d_recv[r], totalFloats, stride, comms[r],
                                       static_cast<pcclStream_t>(streams[r].handle)));
      }
      for (int r = 0; r < nranks; ++r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_DEV(devSynchronizeStream(streams[r]));
      }
    };

    for (int i = 0; i < warmup_iters; ++i)
      submit();

    double total_us = 0.0;
    for (int i = 0; i < perf_iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      submit();
      auto t1 = std::chrono::steady_clock::now();
      total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    double avg_us = total_us / static_cast<double>(perf_iters);
    size_t active_bytes = (seg_count - stride) * static_cast<size_t>(nranks) * sizeof(float);
    double alg_bw = avg_us > 0.0 ? static_cast<double>(active_bytes) / avg_us / 1000.0 : 0.0;
    double bus_bw = alg_bw * bus_factor;

    bool correct = true;
    for (int r = 0; r < nranks && correct; ++r) {
      std::vector<float> h(totalFloats);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], bytes, MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
      for (int j = 0; j < nranks && correct; ++j) {
        size_t base = static_cast<size_t>(j) * seg_count;
        correct = verifyBuffer(h.data() + base, stride, 0.0f, r, "alltoall_strided reserved-prefix");
        if (!correct)
          break;
        float expected = static_cast<float>(j * nranks + r + 1);
        correct = verifyBuffer(h.data() + base + stride, seg_count - stride, expected, r, "alltoall_strided pcie");
      }
    }
    if (!correct)
      allPassed = false;

    printStridedPerfRow(STRIDE_CONFIGS[c].label, correct, avg_us, alg_bw, bus_bw);
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

  std::printf("\ntest_alltoall_strided (%d NPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
