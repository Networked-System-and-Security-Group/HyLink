// Multi-process IN-PLACE AllGather repro mirroring Megatron TP gather.
//   sendbuff = recvbuff + rank*sendcount  (in-place), tiny sendcount, fp32.
// Usage: mpirun -np 4 ./repro_ag_inplace_mp [sendcount] [iters]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <mpi.h>

#include "test_common.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size_t sendcount = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 2;
  int iters = (argc > 2) ? std::atoi(argv[2]) : 5;

  pcclUniqueId uniqueId;
  if (rank == 0) CHECK_PCCL(pcclGetUniqueId(&uniqueId));
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  CHECK_DEV(devInit());
  CHECK_DEV(devSetDevice(rank));

  pcclComm_t comm;
  CHECK_PCCL(pcclCommInitRank(&comm, nranks, uniqueId, rank));

  devStream stream;
  CHECK_DEV(devCreateStream(&stream));

  size_t totalcount = sendcount * nranks;
  void* d_recv = nullptr;
  CHECK_DEV(devMalloc(&d_recv, totalcount * sizeof(float)));
  // In-place: this rank's send region is its slice of the recv buffer.
  float* d_send = static_cast<float*>(d_recv) + rank * sendcount;

  if (rank == 0) std::printf("repro in-place allgather: nranks=%d sendcount=%zu iters=%d\n", nranks, sendcount, iters);

  // Allocate a fleet of independent recv buffers so we can submit back-to-back
  // WITHOUT syncing between them (true compute/comm overlap -> queue depth > 1).
  const int BURST = 4;
  std::vector<void*> bufs(BURST, nullptr);
  for (int b = 0; b < BURST; ++b) {
    CHECK_DEV(devMalloc(&bufs[b], totalcount * sizeof(float)));
  }

  for (int it = 0; it < iters; ++it) {
    // Seed each buffer's own slice.
    for (int b = 0; b < BURST; ++b) {
      float* send = static_cast<float*>(bufs[b]) + rank * sendcount;
      std::vector<float> h(sendcount, static_cast<float>((rank + 1) * 100 + it * 10 + b));
      CHECK_DEV(devMemcpyAsync(send, h.data(), sendcount * sizeof(float), MemcpyDirection::H2D, stream));
    }
    CHECK_DEV(devSynchronizeStream(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::printf("  iter %d: submitting %d back-to-back allgathers (no sync between)...\n", it, BURST), std::fflush(stdout);
    // BURST submits, NO sync in between -> overlap.
    for (int b = 0; b < BURST; ++b) {
      float* send = static_cast<float*>(bufs[b]) + rank * sendcount;
      CHECK_PCCL(pcclAllGather(send, bufs[b], sendcount, comm, stream));
    }
    // Single sync at the very end of the burst.
    CHECK_DEV(devSynchronizeStream(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("  iter %d: DONE\n", it), std::fflush(stdout);

    // Verify every buffer.
    for (int b = 0; b < BURST; ++b) {
      std::vector<float> out(totalcount);
      CHECK_DEV(devMemcpyAsync(out.data(), bufs[b], totalcount * sizeof(float), MemcpyDirection::D2H, stream));
      CHECK_DEV(devSynchronizeStream(stream));
      bool ok = true;
      for (int k = 0; k < nranks && ok; ++k) {
        float exp = static_cast<float>((k + 1) * 100 + it * 10 + b);
        ok = verifyBuffer(out.data() + k * sendcount, sendcount, exp, rank, "ag_inplace");
      }
      if (!ok) std::printf("  rank %d iter %d buf %d: VERIFY FAILED\n", rank, it, b);
    }
  }
  for (int b = 0; b < BURST; ++b) devFree(bufs[b]);

  devFree(d_recv);
  devDestroyStream(stream);
  pcclCommDestroy(comm);
  pcclFinalize();
  if (rank == 0) std::printf("repro_ag_inplace_mp: COMPLETED\n");
  MPI_Finalize();
  return 0;
}
