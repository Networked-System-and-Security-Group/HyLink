/*
 * ReduceScatter multi-process test (one process per GPU).
 *
 * Uses MPI for process management and UniqueId broadcast.
 *
 * Usage: mpirun -np <nranks> ./test_reducescatter_mp
 *
 * Rank r sends [r+1.0, ...]. Expected result on each rank: n*(n+1)/2.
 */

#include <mpi.h>

#include "test_common.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  pcclUniqueId uniqueId;
  if (rank == 0) {
    CHECK_PCCL(pcclGetUniqueId(&uniqueId));
  }
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  CHECK_DEV(devInit());
  CHECK_DEV(devSetDevice(rank));

  pcclComm_t comm;
  CHECK_PCCL(pcclCommInitRank(&comm, nranks, uniqueId, rank));

  devStream stream;
  CHECK_DEV(devCreateStream(&stream));

  size_t maxBytes = benchMaxBytes();
  void* d_send = nullptr;
  void* d_recv = nullptr;
  CHECK_DEV(devMalloc(&d_send, maxBytes));
  CHECK_DEV(devMalloc(&d_recv, maxBytes));

  const float expected = static_cast<float>(nranks) * static_cast<float>(nranks + 1) / 2.0f;
  const double busFactor = busBwFactor(COLL_REDUCESCATTER, nranks);

  bool allPassed = true;

  if (rank == 0) {
    printNumaTopology(nranks);
    char title[64];
    std::snprintf(title, sizeof(title), "ReduceScatter MP | %d GPUs", nranks);
    printBenchHeader(title);
  }

  for (int s = 0; s < numBenchSizes(); ++s) {
    size_t bytes = BENCH_SIZES[s].bytes;
    size_t count = bytes / sizeof(float);
    size_t recvcount = count / nranks;

    // Fill send buffer: rank r sends [r+1.0, ...] for all `count` elements
    {
      std::vector<float> h(count, static_cast<float>(rank + 1));
      CHECK_DEV(devMemcpyAsync(d_send, h.data(), bytes, MemcpyDirection::H2D, stream));
      CHECK_DEV(devSynchronizeStream(stream));
    }

    // Warmup
    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
      CHECK_PCCL(pcclReduceScatter(d_send, d_recv, recvcount, comm, stream));
      CHECK_DEV(devSynchronizeStream(stream));
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Measured iterations
    IterTimings timings;
    timings.reserve(MEASURED_ITERS);
    for (int iter = 0; iter < MEASURED_ITERS; ++iter) {
      CHECK_DEV(devSynchronizeStream(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      auto t0 = std::chrono::high_resolution_clock::now();
      CHECK_PCCL(pcclReduceScatter(d_send, d_recv, recvcount, comm, stream));
      CHECK_DEV(devSynchronizeStream(stream));
      MPI_Barrier(MPI_COMM_WORLD);
      auto t1 = std::chrono::high_resolution_clock::now();

      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      timings.add(ms);
    }

    double avg_ms = timings.avg();
    double alg_bw = (bytes / 1e9) / (avg_ms / 1e3);
    double bus_bw = alg_bw * busFactor;

    // Verify correctness (each rank checks its own recvcount elements)
    bool correct = true;
    {
      size_t recvBytes = recvcount * sizeof(float);
      std::vector<float> h(recvcount);
      CHECK_DEV(devMemcpyAsync(h.data(), d_recv, recvBytes, MemcpyDirection::D2H, stream));
      CHECK_DEV(devSynchronizeStream(stream));
      correct = verifyBuffer(h.data(), recvcount, expected, rank, "reducescatter_mp");
    }

    int local_ok = correct ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    correct = (global_ok == 1);
    if (!correct) allPassed = false;

    if (rank == 0) {
      printBenchRow(BENCH_SIZES[s].label, avg_ms, alg_bw, bus_bw, correct);
      printPercentileTable(timings, static_cast<double>(bytes), busFactor);
    }
  }

  devFree(d_send);
  devFree(d_recv);
  devDestroyStream(stream);
  pcclCommDestroy(comm);
  pcclFinalize();

  if (rank == 0) {
    std::printf("\ntest_reducescatter_mp (%d GPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  }

  MPI_Finalize();
  return allPassed ? 0 : 1;
}
