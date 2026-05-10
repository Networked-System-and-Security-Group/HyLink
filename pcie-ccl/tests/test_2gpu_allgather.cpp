#include "include/comm.hpp"
#include "include/hal/device.hpp"
#include "include/ir.hpp"
#include "test_utils.hpp"

using namespace pccl;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <rank> [size] [dev0_id] [dev1_id]\n", argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  size_t size = argc > 2 ? atoll(argv[2]) : DEFAULT_SIZE;

  printf("\n=== Test: 2-GPU AllGather ===\n");
  printf("Rank: %d, Size: %.0f MB\n", rank, size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, 2, &comm));

  void* sendbuff;
  void* recvbuff;
  size_t chunk_size = size;
  size_t count = chunk_size / sizeof(float);

  if (devMalloc(&sendbuff, chunk_size) != 0) {
    fprintf(stderr, "Failed to allocate sendbuff\n");
    return 1;
  }
  if (devMalloc(&recvbuff, chunk_size * 2) != 0) {
    fprintf(stderr, "Failed to allocate recvbuff\n");
    return 1;
  }

  // Allocate host buffers for initialization and verification
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, chunk_size));
  DEV_CHECK(devMallocHost(&h_verify, chunk_size * 2));

  // Initialize sendbuff with rank-specific value (rank 0 -> 0xAA, rank 1 -> 0xBB)
  char init_value = (rank == 0) ? 0xAA : 0xBB;
  memset(h_init, init_value, chunk_size);
  DEV_CHECK(devMemcpySync(sendbuff, h_init, chunk_size, MemcpyDirection::H2D));

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Failed to create stream\n");
    return 1;
  }

  IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = 2;

  if (rank == 0) {
    Instruction inst0;
    inst0.op = OpCode::D2H;
    inst0.src_chunk_idx = 0;
    inst0.dst_chunk_idx = 0;
    inst0.effects = {{0}};

    Instruction inst1;
    inst1.op = OpCode::D2D;
    inst1.src_chunk_idx = 0;
    inst1.dst_chunk_idx = 0;

    Instruction inst2;
    inst2.op = OpCode::H2D;
    inst2.src_chunk_idx = 1;
    inst2.dst_chunk_idx = 1;
    inst2.deps = {{0, 1, 1}};

    program.instructions = {inst0, inst1, inst2};
  } else {
    Instruction inst0;
    inst0.op = OpCode::D2H;
    inst0.src_chunk_idx = 0;
    inst0.dst_chunk_idx = 1;
    inst0.effects = {{1}};

    Instruction inst1;
    inst1.op = OpCode::D2D;
    inst1.src_chunk_idx = 0;
    inst1.dst_chunk_idx = 1;

    Instruction inst2;
    inst2.op = OpCode::H2D;
    inst2.src_chunk_idx = 0;
    inst2.dst_chunk_idx = 0;
    inst2.deps = {{0, 0, 1}};

    program.instructions = {inst0, inst1, inst2};
  }

  HostTimer timer;
  std::vector<float> times;
  int num_ranks = 2;
  size_t alg_data_size = chunk_size * num_ranks;
  double bus_bw_factor = (num_ranks - 1.0) / num_ranks;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer.start();
    PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
    if (devSynchronizeStream(stream) != 0) {
      fprintf(stderr, "Failed to synchronize stream\n");
      return 1;
    }
    timer.end();

    float ms = timer.getElapsedMs();
    double alg_bw = calculateAlgBandwidth(alg_data_size, ms);
    double bus_bw = alg_bw * bus_bw_factor;

    if (i >= WARMUP_ITERS) {
      times.push_back(ms);
      printf("Iteration %d: %.2f ms, alg %.2f GB/s, bus %.2f GB/s\n", i, ms, alg_bw, bus_bw);
    } else {
      printf("Iteration %d (warmup): %.2f ms, alg %.2f GB/s, bus %.2f GB/s\n", i, ms, alg_bw, bus_bw);
    }
  }

  auto stats = calculateStats(times, alg_data_size, bus_bw_factor);
  printf("\nAvg Time: %.2f ms, Alg BW: %.2f GB/s, Bus BW: %.2f GB/s\n", stats.avg_time_ms, stats.avg_bw_gbps,
         stats.avg_bus_bw_gbps);

  // Synchronize internal streams before verification and freeing memory
  pcclSynchronizeInternalStreams(comm);

  // Verify data after all iterations complete
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, chunk_size * 2, MemcpyDirection::D2H));
  // recvbuff[0] should contain rank 0's data (0xAA)
  // recvbuff[1] should contain rank 1's data (0xBB)
  bool chunk0_ok = verifyData((char*)h_verify, chunk_size, (char)0xAA, "recvbuff[0]");
  bool chunk1_ok = verifyData((char*)h_verify + chunk_size, chunk_size, (char)0xBB, "recvbuff[1]");
  if (chunk0_ok && chunk1_ok) {
    printf("Verification: PASSED\n");
  } else {
    printf("Verification: FAILED\n");
  }

  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  PCCL_CHECK(pcclDestroy(comm));
  DEV_CHECK(devFreeHost(h_init));
  DEV_CHECK(devFreeHost(h_verify));

  return 0;
}
