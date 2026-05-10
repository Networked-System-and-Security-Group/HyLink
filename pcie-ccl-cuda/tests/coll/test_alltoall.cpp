/*
 * AllToAll correctness + bandwidth test.
 *
 * Usage: test_alltoall <nranks>
 *
 * Rank r fills send segment j with float(r * nranks + j + 1).
 * After AllToAll, rank r's recv segment j = float(j * nranks + r + 1).
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

  size_t maxBytes = benchMaxBytes();
  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMalloc(&d_send[r], maxBytes));
    CHECK_DEV(devMalloc(&d_recv[r], maxBytes));
  }

  const double busFactor = busBwFactor(COLL_ALLTOALL, nranks);

  CHECK_DEV(devSetDevice(0));
  bool allPassed = true;
  RankThreadPool pool(nranks);

  char title[64];
  std::snprintf(title, sizeof(title), "AllToAll | %d GPUs", nranks);
  printBenchHeader(title);

  for (int s = 0; s < numBenchSizes(); ++s) {
    size_t bytes = BENCH_SIZES[s].bytes;
    size_t count = bytes / sizeof(float);
    size_t per_rank_count = count / nranks;

    /* Rank r, segment j = float(r * nranks + j + 1) */
    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      std::vector<float> h(count);
      for (int j = 0; j < nranks; ++j) {
        float val = static_cast<float>(r * nranks + j + 1);
        for (size_t k = 0; k < per_rank_count; ++k)
          h[j * per_rank_count + k] = val;
      }
      CHECK_DEV(devMemcpyAsync(d_send[r], h.data(), bytes, MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
      pool.run([&](int r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllToAll(d_send[r], d_recv[r], count, comms[r], streams[r]));
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
        CHECK_PCCL(pcclAllToAll(d_send[r], d_recv[r], count, comms[r], streams[r]));
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
    double alg_bw = (bytes / 1e9) / (avg_ms / 1e3);
    double bus_bw = alg_bw * busFactor;

    /* Rank r, segment j should be float(j * nranks + r + 1) */
    bool correct = true;
    for (int r = 0; r < nranks && correct; ++r) {
      std::vector<float> h(count);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], bytes, MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
      for (int j = 0; j < nranks && correct; ++j) {
        float expected = static_cast<float>(j * nranks + r + 1);
        correct = verifyBuffer(h.data() + j * per_rank_count, per_rank_count, expected, r, "alltoall");
      }
    }
    if (!correct)
      allPassed = false;

    printBenchRow(BENCH_SIZES[s].label, avg_ms, alg_bw, bus_bw, correct);
    printPercentileTable(timings, static_cast<double>(bytes), busFactor);
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

  std::printf("\ntest_alltoall (%d GPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
