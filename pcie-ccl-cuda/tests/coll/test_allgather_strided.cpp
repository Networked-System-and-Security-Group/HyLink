/*
 * AllGatherStrided correctness + bandwidth test.
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

  const double busFactor = busBwFactor(COLL_ALLGATHER, nranks);

  CHECK_DEV(devSetDevice(0));
  bool allPassed = true;
  RankThreadPool pool(nranks);

  char title[64];
  std::snprintf(title, sizeof(title), "AllGatherStrided | %d GPUs", nranks);
  printBenchHeader(title);

  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c) {
    size_t seg_count = STRIDE_CONFIGS[c].seg_count;
    size_t stride = STRIDE_CONFIGS[c].stride;

    std::printf("\n--- Stride: %s (seg_count=%zu, stride=%zu) ---\n", STRIDE_CONFIGS[c].label, seg_count, stride);

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

    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
      pool.run([&](int r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllGatherStrided(d_send[r], d_recv[r], seg_count, stride, comms[r],
                                        static_cast<pcclStream_t>(streams[r].handle)));
      });
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
      pool.run([&](int r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllGatherStrided(d_send[r], d_recv[r], seg_count, stride, comms[r],
                                        static_cast<pcclStream_t>(streams[r].handle)));
      });
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
      std::vector<float> h(seg_count * nranks);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], seg_count * nranks * sizeof(float), MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));

      for (int k = 0; k < nranks && correct; ++k) {
        float* seg = h.data() + k * seg_count;
        for (size_t i = 0; i < stride && correct; ++i) {
          if (std::fabs(seg[i] - 0.0f) > 1e-5f) {
            std::fprintf(stderr, "  FAIL rank %d seg %d fast-path: buf[%zu] = %f, expected 0.0\n", r, k, i, seg[i]);
            correct = false;
          }
        }
        float expected = static_cast<float>(k + 1);
        for (size_t i = stride; i < seg_count && correct; ++i) {
          if (std::fabs(seg[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "  FAIL rank %d seg %d pcie: buf[%zu] = %f, expected %f\n", r, k, i, seg[i], expected);
            correct = false;
          }
        }
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

  std::printf("\ntest_allgather_strided (%d GPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
