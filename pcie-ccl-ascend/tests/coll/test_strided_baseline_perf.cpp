/*
 * Non-strided PCCL baseline for strided active-byte comparisons.
 *
 * Usage: test_strided_baseline_perf <nranks>
 *
 * For every STRIDE_CONFIGS entry, this runs non-strided AllGather and
 * ReduceScatter with the same active byte count used by the strided tests.
 */

#include "test_common.hpp"

namespace {

struct PerfResult {
  bool correct;
  double avg_us;
  double alg_bw;
  double bus_bw;
};

PerfResult runAllGather(int nranks, size_t active_count, int perf_iters, int warmup_iters, double bus_factor) {
  std::vector<pcclComm_t> comms(nranks);
  std::vector<devStream> streams(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_PCCL(pcclCommInitRank(&comms[r], nranks, singleProcessId(), r));
    CHECK_DEV(devCreateStream(&streams[r]));
  }

  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMalloc(&d_send[r], active_count * sizeof(float)));
    CHECK_DEV(devMalloc(&d_recv[r], active_count * static_cast<size_t>(nranks) * sizeof(float)));
  }

  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    std::vector<float> h(active_count, static_cast<float>(r + 1));
    CHECK_DEV(devMemcpyAsync(d_send[r], h.data(), active_count * sizeof(float), MemcpyDirection::H2D, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));

    std::vector<float> z(active_count * static_cast<size_t>(nranks), 0.0f);
    CHECK_DEV(devMemcpyAsync(d_recv[r], z.data(), z.size() * sizeof(float), MemcpyDirection::H2D, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));
  }

  RankThreadPool pool(nranks);
  auto submit = [&]() {
    pool.run([&](int r) {
      CHECK_DEV(devSetDevice(r));
      CHECK_PCCL(pcclAllGather(d_send[r], d_recv[r], active_count, comms[r],
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

  bool correct = true;
  for (int r = 0; r < nranks && correct; ++r) {
    std::vector<float> h(active_count * static_cast<size_t>(nranks));
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], h.size() * sizeof(float), MemcpyDirection::D2H, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));
    for (int src = 0; src < nranks && correct; ++src) {
      correct = verifyBuffer(h.data() + active_count * static_cast<size_t>(src), active_count,
                             static_cast<float>(src + 1), r, "allgather_baseline");
    }
  }

  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    devFree(d_send[r]);
    devFree(d_recv[r]);
    devDestroyStream(streams[r]);
    pcclCommDestroy(comms[r]);
  }

  double avg_us = total_us / static_cast<double>(perf_iters);
  size_t active_bytes = active_count * static_cast<size_t>(nranks) * sizeof(float);
  double alg_bw = avg_us > 0.0 ? static_cast<double>(active_bytes) / avg_us / 1000.0 : 0.0;
  return {correct, avg_us, alg_bw, alg_bw * bus_factor};
}

PerfResult runReduceScatter(int nranks, size_t active_count, int perf_iters, int warmup_iters, double bus_factor) {
  std::vector<pcclComm_t> comms(nranks);
  std::vector<devStream> streams(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_PCCL(pcclCommInitRank(&comms[r], nranks, singleProcessId(), r));
    CHECK_DEV(devCreateStream(&streams[r]));
  }

  size_t total_count = active_count * static_cast<size_t>(nranks);
  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMalloc(&d_send[r], total_count * sizeof(float)));
    CHECK_DEV(devMalloc(&d_recv[r], active_count * sizeof(float)));
  }

  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    std::vector<float> h(total_count, static_cast<float>(r + 1));
    CHECK_DEV(devMemcpyAsync(d_send[r], h.data(), total_count * sizeof(float), MemcpyDirection::H2D, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));

    std::vector<float> z(active_count, 0.0f);
    CHECK_DEV(devMemcpyAsync(d_recv[r], z.data(), active_count * sizeof(float), MemcpyDirection::H2D, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));
  }

  RankThreadPool pool(nranks);
  auto submit = [&]() {
    pool.run([&](int r) {
      CHECK_DEV(devSetDevice(r));
      CHECK_PCCL(pcclReduceScatter(d_send[r], d_recv[r], active_count, comms[r],
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

  float expected = static_cast<float>(nranks) * static_cast<float>(nranks + 1) / 2.0f;
  bool correct = true;
  for (int r = 0; r < nranks && correct; ++r) {
    std::vector<float> h(active_count);
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devMemcpyAsync(h.data(), d_recv[r], active_count * sizeof(float), MemcpyDirection::D2H, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));
    correct = verifyBuffer(h.data(), active_count, expected, r, "reducescatter_baseline");
  }

  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    devFree(d_send[r]);
    devFree(d_recv[r]);
    devDestroyStream(streams[r]);
    pcclCommDestroy(comms[r]);
  }

  double avg_us = total_us / static_cast<double>(perf_iters);
  size_t active_bytes = active_count * static_cast<size_t>(nranks) * sizeof(float);
  double alg_bw = avg_us > 0.0 ? static_cast<double>(active_bytes) / avg_us / 1000.0 : 0.0;
  return {correct, avg_us, alg_bw, alg_bw * bus_factor};
}

void printPerfLine(const char* primitive, const char* label, const PerfResult& result) {
  std::printf("%-16s  %-22s  %-7s  %-13.2f  %-14.5f  %-14.5f\n", primitive, label,
              result.correct ? "PASS" : "FAIL", result.avg_us, result.alg_bw, result.bus_bw);
}

}  // namespace

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

  const int perf_iters = stridedPerfIters();
  const int warmup_iters = stridedPerfWarmupIters();
  const double bus_factor = stridedBusFactor(nranks);
  bool allPassed = true;

  std::printf("\n=== NonStridedBaseline | %d NPUs correctness/perf ===\n\n", nranks);
  std::printf("Perf: measured_iters=%d warmup_iters=%d, bandwidth uses strided active-byte equivalents\n", perf_iters,
              warmup_iters);
  std::printf("%-16s  %-22s  %-7s  %-13s  %-14s  %-14s\n", "Primitive", "Case", "Correct", "avg_time(us)",
              "alg_bw(GB/s)", "bus_bw(GB/s)");
  std::printf("%-16s  %-22s  %-7s  %-13s  %-14s  %-14s\n", "----------------", "----------------------", "-------",
              "------------", "-------------", "-------------");

  for (int c = 0; c < NUM_STRIDE_CONFIGS; ++c) {
    size_t active_count = STRIDE_CONFIGS[c].seg_count - STRIDE_CONFIGS[c].stride;

    PerfResult ag = runAllGather(nranks, active_count, perf_iters, warmup_iters, bus_factor);
    printPerfLine("AllGather", STRIDE_CONFIGS[c].label, ag);
    if (!ag.correct)
      allPassed = false;

    PerfResult rs = runReduceScatter(nranks, active_count, perf_iters, warmup_iters, bus_factor);
    printPerfLine("ReduceScatter", STRIDE_CONFIGS[c].label, rs);
    if (!rs.correct)
      allPassed = false;
  }

  pcclFinalize();
  std::printf("\ntest_strided_baseline_perf (%d NPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  return allPassed ? 0 : 1;
}
