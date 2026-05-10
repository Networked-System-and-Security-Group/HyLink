/*
 * Minimal reproducer for the fp16 AllReduce hang (D2DReduce node never completes).
 *
 * Single process, one thread per rank (RankThreadPool), single-process comm —
 * exactly like test_allreduce, but submits via pcclSubmitTypedCollective with a
 * selectable scalar type. fp32 (scalar=7) completes; fp16 (scalar=6) hangs.
 *
 * Usage: repro_fp16_allreduce <nranks> <scalar:6=fp16,7=fp32> <count>
 */
#include "test_common.hpp"
#include "../../src/ir/compiler.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {
  int nranks = argc > 1 ? std::atoi(argv[1]) : 4;
  int scalar = argc > 2 ? std::atoi(argv[2]) : 6;   // 6=fp16, 7=fp32
  size_t count = argc > 3 ? static_cast<size_t>(std::atoll(argv[3])) : 1024;
  size_t elem = (scalar == 7) ? 4 : 2;
  size_t bytes = count * elem;

  CHECK_DEV(devInit());
  std::vector<pcclComm_t> comms(nranks);
  std::vector<devStream> streams(nranks);
  std::vector<void*> d_send(nranks), d_recv(nranks);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_PCCL(pcclCommInitRank(&comms[r], nranks, singleProcessId(), r));
    CHECK_DEV(devCreateStream(&streams[r]));
    CHECK_DEV(devMalloc(&d_send[r], bytes));
    CHECK_DEV(devMalloc(&d_recv[r], bytes));
    std::vector<char> zero(bytes, 0);
    CHECK_DEV(devMemcpyAsync(d_send[r], zero.data(), bytes, MemcpyDirection::H2D, streams[r]));
    CHECK_DEV(devSynchronizeStream(streams[r]));
  }

  std::printf("repro: nranks=%d scalar=%d (%s) count=%zu bytes=%zu\n", nranks, scalar,
              scalar == 7 ? "fp32" : "fp16", count, bytes);
  std::fflush(stdout);

  RankThreadPool pool(nranks);
  pool.run([&](int r) {
    CHECK_DEV(devSetDevice(r));
    pccl::CompileOptions opts;
    opts.seg_count = count;
    opts.element_size = elem;
    opts.scalar_type = static_cast<pccl::ScalarDataType>(scalar);
    opts.reduce_op = pccl::ReduceOpKind::Sum;
    opts.stride = 0;
    CHECK_PCCL(pccl::pcclSubmitTypedCollective(comms[r], "allreduce", d_send[r], d_recv[r], count, 0, 1, 1,
                                               streams[r], &opts));
  });
  std::printf("submitted on all ranks; synchronizing...\n");
  std::fflush(stdout);
  for (int r = 0; r < nranks; ++r) {
    CHECK_DEV(devSetDevice(r));
    CHECK_DEV(devSynchronizeStream(streams[r]));
    std::printf("rank %d synchronized\n", r);
    std::fflush(stdout);
  }
  std::printf("ALL RANKS COMPLETE\n");
  return 0;
}
