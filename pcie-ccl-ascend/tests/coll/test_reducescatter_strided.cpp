/*
 * ReduceScatterStrided correctness test.
 *
 * Usage: test_reducescatter_strided <nranks>
 *
 * Rank r fills each segment's active region with (r+1).0f and zeroes the
 * reserved prefix.  After the collective each rank's recv buffer should
 * have 0.0f in [0..stride-1] and n*(n+1)/2 in [stride..seg_count-1].
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

  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMalloc(&d_send[r], max_seg_count * nranks * sizeof(float)));
    CHECK_DEV(devMalloc(&d_recv[r], max_seg_count * sizeof(float)));
  }

  const float expected = static_cast<float>(nranks) * static_cast<float>(nranks + 1) / 2.0f;
  CHECK_DEV(devSetDevice(0));
  bool allPassed = true;
  RankThreadPool pool(nranks);
  const int perf_iters = stridedPerfIters();
  const int warmup_iters = stridedPerfWarmupIters();
  const double bus_factor = stridedBusFactor(nranks);

  char title[64];
  std::snprintf(title, sizeof(title), "ReduceScatterStrided | %d NPUs", nranks);
  std::printf("\n=== %s correctness/perf ===\n\n", title);
  printStridedPerfHeader(perf_iters, warmup_iters);

  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c) {
    size_t seg_count = STRIDE_CONFIGS[c].seg_count;
    size_t stride = STRIDE_CONFIGS[c].stride;
    size_t totalFloats = seg_count * nranks;
    size_t totalBytes = totalFloats * sizeof(float);
    size_t recvFloats = seg_count;
    size_t recvBytes = recvFloats * sizeof(float);

    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> h(totalFloats);
      float fill = static_cast<float>(r + 1);
      for (int seg = 0; seg < nranks; ++seg) {
        size_t base = static_cast<size_t>(seg) * seg_count;
        for (size_t i = 0; i < stride; ++i)
          h[base + i] = 0.0f;
        for (size_t i = stride; i < seg_count; ++i)
          h[base + i] = fill;
      }
      CHECK_DEV(devMemcpyAsync(d_send[r], h.data(), totalBytes, MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> z(recvFloats, 0.0f);
      CHECK_DEV(devMemcpyAsync(d_recv[r], z.data(), recvBytes, MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    auto submit = [&]() {
      pool.run([&](int r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclReduceScatterStrided(d_send[r], d_recv[r], totalFloats, stride, comms[r],
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
      std::vector<float> h(recvFloats);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], recvBytes, MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
      for (size_t i = 0; i < stride && correct; ++i) {
        if (std::fabs(h[i]) > 1e-5f) {
          std::fprintf(stderr, "  FAIL rank %d reducescatter_strided: recv[%zu] = %f, expected 0.0\n", r, i, h[i]);
          correct = false;
        }
      }
      for (size_t i = stride; i < recvFloats && correct; ++i) {
        if (std::fabs(h[i] - expected) > 1e-5f) {
          std::fprintf(stderr, "  FAIL rank %d reducescatter_strided: recv[%zu] = %f, expected %f\n", r, i, h[i],
                       expected);
          correct = false;
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

  std::printf("\ntest_reducescatter_strided (%d NPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
