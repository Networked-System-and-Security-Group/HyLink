/*
 * Broadcast multi-process test using the typed internal PCCL submission path.
 *
 * Broadcast is normally reached through the NCCL compatibility layer, which
 * names collectives as broadcast_rootX.
 */

#include <mpi.h>

#include "ir/compiler.hpp"
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

  pccl::CompileOptions options;
  options.element_size = sizeof(float);
  options.scalar_type = pccl::ScalarDataType::Float32;
  options.reduce_op = pccl::ReduceOpKind::Sum;

  bool allPassed = true;
  if (rank == 0) {
    printNumaTopology(nranks);
    char title[64];
    std::snprintf(title, sizeof(title), "Broadcast MP | %d GPUs", nranks);
    printBenchHeader(title);
  }

  for (int s = 0; s < numBenchSizes(); ++s) {
    size_t bytes = BENCH_SIZES[s].bytes;
    size_t count = bytes / sizeof(float);
    bool size_correct = true;

    for (int root = 0; root < nranks; ++root) {
      float send_value = static_cast<float>(root + 11);
      std::vector<float> h_send(count, rank == root ? send_value : -1.0f);
      std::vector<float> h_recv(count, -7.0f);
      CHECK_DEV(devMemcpyAsync(d_send, h_send.data(), bytes, MemcpyDirection::H2D, stream));
      CHECK_DEV(devMemcpyAsync(d_recv, h_recv.data(), bytes, MemcpyDirection::H2D, stream));
      CHECK_DEV(devSynchronizeStream(stream));

      std::string collective = "broadcast_root" + std::to_string(root);
      const void* local_send = (rank == root) ? d_send : d_recv;
      CHECK_PCCL(pccl::pcclSubmitTypedCollective(comm, collective.c_str(), local_send, d_recv, count, 0, 1, 1,
                                                 stream, &options));
      CHECK_DEV(devSynchronizeStream(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      std::vector<float> h_out(count);
      CHECK_DEV(devMemcpyAsync(h_out.data(), d_recv, bytes, MemcpyDirection::D2H, stream));
      CHECK_DEV(devSynchronizeStream(stream));
      bool correct = verifyBuffer(h_out.data(), count, send_value, rank, "broadcast_mp");
      int local_ok = correct ? 1 : 0;
      int global_ok = 0;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (global_ok != 1)
        size_correct = false;
    }

    if (!size_correct)
      allPassed = false;
    if (rank == 0) {
      printBenchRow(BENCH_SIZES[s].label, 0.0, 0.0, 0.0, size_correct);
    }
  }

  devFree(d_send);
  devFree(d_recv);
  devDestroyStream(stream);
  pcclCommDestroy(comm);
  pcclFinalize();

  if (rank == 0) {
    std::printf("\ntest_broadcast_mp (%d GPUs): %s\n", nranks, allPassed ? "PASSED" : "FAILED");
  }

  MPI_Finalize();
  return allPassed ? 0 : 1;
}
