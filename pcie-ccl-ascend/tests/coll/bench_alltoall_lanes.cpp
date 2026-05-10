#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "test_common.hpp"

namespace {

constexpr size_t kMiB = 1u << 20;

size_t parseByteSize(const char* text) {
  if (!text || text[0] == '\0')
    return 0;

  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || value == 0)
    return 0;

  unsigned long long scale = 1;
  if (*end != '\0') {
    if (end[1] != '\0')
      return 0;
    if (*end == 'K' || *end == 'k')
      scale = 1ull << 10;
    else if (*end == 'M' || *end == 'm')
      scale = 1ull << 20;
    else if (*end == 'G' || *end == 'g')
      scale = 1ull << 30;
    else
      return 0;
  }

  if (value > std::numeric_limits<size_t>::max() / scale)
    return 0;
  return static_cast<size_t>(value * scale);
}

bool parsePositiveInt(const char* text, int* value) {
  if (!text || !value || text[0] == '\0')
    return false;

  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int>::max()))
    return false;
  *value = static_cast<int>(parsed);
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "Usage: %s <nranks> <lanes|auto> [max_message_bytes]\n", argv[0]);
    return 1;
  }

  const int nranks = std::atoi(argv[1]);
  const bool auto_policy = std::strcmp(argv[2], "auto") == 0;
  int lanes = 0;
  if (nranks <= 0 || (!auto_policy && !parsePositiveInt(argv[2], &lanes))) {
    std::fprintf(stderr, "nranks must be positive and lanes must be a positive integer or auto\n");
    return 1;
  }

  const size_t max_message_bytes = argc == 4 ? parseByteSize(argv[3]) : 128 * kMiB;
  if (max_message_bytes == 0) {
    std::fprintf(stderr, "max_message_bytes must be a positive byte size, for example 128M\n");
    return 1;
  }

  if (auto_policy) {
    if (::unsetenv("PCCL_FORCE_DMA_LANES") != 0) {
      std::fprintf(stderr, "failed to clear PCCL_FORCE_DMA_LANES\n");
      return 1;
    }
  } else if (::setenv("PCCL_FORCE_DMA_LANES", argv[2], 1) != 0) {
    std::fprintf(stderr, "failed to set PCCL_FORCE_DMA_LANES\n");
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

  const size_t rank_bytes = static_cast<size_t>(nranks) * sizeof(float);
  const size_t message_sizes[] = {
      1 * kMiB, 4 * kMiB, 8 * kMiB, 16 * kMiB, 32 * kMiB, 64 * kMiB, 128 * kMiB,
  };
  const int perf_iters = stridedPerfIters();
  const int warmup_iters = stridedPerfWarmupIters();
  const double bus_factor = stridedBusFactor(nranks);

  const char* lane_label = auto_policy ? "auto" : argv[2];
  std::printf("\n=== AllToAll host DMA lane benchmark | %d NPUs | forced_lanes=%s ===\n\n", nranks,
              lane_label);
  std::printf("Perf: measured_iters=%d warmup_iters=%d, message_bytes is total per rank\n", perf_iters,
              warmup_iters);
  std::printf("%-15s  %-7s  %-7s  %-13s  %-14s  %-14s\n", "message_bytes", "lanes", "Correct",
              "avg_time(us)", "alg_bw(GB/s)", "bus_bw(GB/s)");
  std::printf("%-15s  %-7s  %-7s  %-13s  %-14s  %-14s\n", "---------------", "-------", "-------",
              "------------", "-------------", "-------------");

  RankThreadPool pool(nranks);
  bool all_passed = true;
  bool measured_one_size = false;

  for (size_t requested_bytes : message_sizes) {
    if (requested_bytes > max_message_bytes)
      continue;

    const size_t total_bytes = requested_bytes - requested_bytes % rank_bytes;
    if (total_bytes < rank_bytes)
      continue;
    const size_t total_count = total_bytes / sizeof(float);
    const size_t segment_count = total_count / static_cast<size_t>(nranks);
    measured_one_size = true;

    std::vector<void*> d_send(nranks, nullptr);
    std::vector<void*> d_recv(nranks, nullptr);
    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMalloc(&d_send[r], total_bytes));
      CHECK_DEV(devMalloc(&d_recv[r], total_bytes));

      std::vector<float> h_send(total_count, 0.0f);
      for (int dest = 0; dest < nranks; ++dest) {
        const float value = static_cast<float>(r * nranks + dest + 1);
        const size_t begin = static_cast<size_t>(dest) * segment_count;
        std::fill(h_send.begin() + begin, h_send.begin() + begin + segment_count, value);
      }
      CHECK_DEV(devMemcpyAsync(d_send[r], h_send.data(), total_bytes, MemcpyDirection::H2D, streams[r]));

      std::vector<float> zeros(total_count, 0.0f);
      CHECK_DEV(devMemcpyAsync(d_recv[r], zeros.data(), total_bytes, MemcpyDirection::H2D, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
    }

    auto submit = [&]() {
      pool.run([&](int r) {
        CHECK_DEV(devSetDevice(r));
        CHECK_PCCL(pcclAllToAll(d_send[r], d_recv[r], total_count, comms[r],
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
      const auto start = std::chrono::steady_clock::now();
      submit();
      const auto end = std::chrono::steady_clock::now();
      total_us += std::chrono::duration<double, std::micro>(end - start).count();
    }
    const double avg_us = total_us / static_cast<double>(perf_iters);
    const double alg_bw = avg_us > 0.0 ? static_cast<double>(total_bytes) / avg_us / 1000.0 : 0.0;
    const double bus_bw = alg_bw * bus_factor;

    bool correct = true;
    for (int r = 0; r < nranks && correct; ++r) {
      std::vector<float> h_recv(total_count, 0.0f);
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devMemcpyAsync(h_recv.data(), d_recv[r], total_bytes, MemcpyDirection::D2H, streams[r]));
      CHECK_DEV(devSynchronizeStream(streams[r]));
      for (int source = 0; source < nranks && correct; ++source) {
        const float expected = static_cast<float>(source * nranks + r + 1);
        const float* segment = h_recv.data() + static_cast<size_t>(source) * segment_count;
        correct = verifyBuffer(segment, segment_count, expected, r, "alltoall lane");
      }
    }
    all_passed = all_passed && correct;

    std::printf("%-15zu  %-7s  %-7s  %-13.2f  %-14.5f  %-14.5f\n", total_bytes, lane_label,
                correct ? "PASS" : "FAIL", avg_us, alg_bw, bus_bw);

    for (int r = 0; r < nranks; ++r) {
      CHECK_DEV(devSetDevice(r));
      CHECK_DEV(devFree(d_send[r]));
      CHECK_DEV(devFree(d_recv[r]));
    }
  }

  if (!measured_one_size) {
    std::fprintf(stderr, "max_message_bytes is smaller than the 1 MiB benchmark size\n");
    all_passed = false;
  }

  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devDestroyStream(streams[r]));
    CHECK_PCCL(pcclCommDestroy(comms[r]));
  }
  CHECK_PCCL(pcclFinalize());

  std::printf("\nbench_alltoall_lanes (%d NPUs, forced_lanes=%s): %s\n", nranks, lane_label,
              all_passed ? "PASSED" : "FAILED");
  return all_passed ? 0 : 1;
}
