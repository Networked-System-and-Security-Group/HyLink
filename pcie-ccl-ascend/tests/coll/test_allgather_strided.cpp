/*
 * AllGatherStrided correctness test.
 *
 * Usage: test_allgather_strided <nranks>
 *
 * Rank r sends seg_count floats: [0..stride-1] = 0, [stride..seg_count-1] = (r+1).
 * After gather, segment k: [0..stride-1] = 0, [stride..seg_count-1] = (k+1).
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

  size_t maxSegCount = 0;
  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c)
    if (STRIDE_CONFIGS[c].seg_count > maxSegCount)
      maxSegCount = STRIDE_CONFIGS[c].seg_count;

  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMalloc(&d_send[r], maxSegCount * sizeof(float)));
    CHECK_DEV(devMalloc(&d_recv[r], maxSegCount * nranks * sizeof(float)));
  }

  CHECK_DEV(devSetDevice(0));
  bool allPassed = true;
  RankThreadPool pool(nranks);
  const int perf_iters = stridedPerfIters();
  const int warmup_iters = stridedPerfWarmupIters();
  const double bus_factor = stridedBusFactor(nranks);

  char title[64];
  std::snprintf(title, sizeof(title), "AllGatherStrided | %d NPUs", nranks);
  std::printf("\n=== %s correctness/perf ===\n\n", title);
  printStridedPerfHeader(perf_iters, warmup_iters);

  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c) {
    size_t seg_count = STRIDE_CONFIGS[c].seg_count;
    size_t stride = STRIDE_CONFIGS[c].stride;

    /* [0..stride-1] = 0, [stride..seg_count-1] = (r+1) */
    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> h(seg_count, 0.0f);
      float val = static_cast<float>(r + 1);
      for (size_t i = stride; i < seg_count; ++i)
        h[i] = val;
      CHECK_DEV(devMemcpyAsync(d_send[r], h.data(), seg_count * sizeof(float), MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> z(seg_count * nranks, 0.0f);
      CHECK_DEV(devMemcpyAsync(d_recv[r], z.data(), seg_count * nranks * sizeof(float), MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    auto submit = [&]() {
      pool.run([&](int r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllGatherStrided(d_send[r], d_recv[r], seg_count, stride, comms[r],
                                        static_cast<pcclStream_t>(streams[r].handle)));
      });
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
      std::vector<float> h(seg_count * nranks);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], seg_count * nranks * sizeof(float), MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));

      for (int k = 0; k < nranks && correct; ++k) {
        float* seg = h.data() + k * seg_count;
        for (size_t i = 0; i < stride && correct; ++i) {
          if (std::fabs(seg[i] - 0.0f) > 1e-5f) {
            std::fprintf(stderr, "  FAIL rank %d seg %d reserved prefix: buf[%zu] = %f, expected 0.0\n", r, k, i, seg[i]);
            correct = false;
          }
        }
        float expected = static_cast<float>(k + 1);
        for (size_t i = stride; i < seg_count && correct; ++i) {
          if (std::fabs(seg[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "  FAIL rank %d seg %d active: buf[%zu] = %f, expected %f\n", r, k, i, seg[i], expected);
            correct = false;
          }
        }
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

  std::printf("\ntest_allgather_strided (%d NPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
