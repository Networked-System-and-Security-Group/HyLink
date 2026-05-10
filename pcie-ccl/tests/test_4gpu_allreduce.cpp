#include "include/comm.hpp"
#include "include/hal/device.hpp"
#include "include/ir.hpp"
#include "test_utils.hpp"

using namespace pccl;

static size_t parseSize(const char* str) {
  char* end;
  size_t val = strtoull(str, &end, 10);
  if (*end == 'K' || *end == 'k') val *= 1024;
  else if (*end == 'M' || *end == 'm') val *= 1024 * 1024;
  else if (*end == 'G' || *end == 'g') val *= 1024 * 1024 * 1024;
  return val;
}

static const int NUM_RANKS = 4;
static const int NUM_CHUNKS = 4;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <rank> [size] [dev_id]\n", argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  size_t size = argc > 2 ? parseSize(argv[2]) : DEFAULT_SIZE;

  if (rank < 0 || rank >= NUM_RANKS) {
    fprintf(stderr, "Rank must be 0-%d\n", NUM_RANKS - 1);
    return 1;
  }

  printf("\n=== Test: 4-GPU AllReduce ===\n");
  printf("Rank: %d, Size: %.0f MB\n", rank, size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, NUM_RANKS, &comm));

  // Each rank has `size` bytes total, split into NUM_CHUNKS chunks
  size_t total_size = size;
  size_t count = total_size / sizeof(float);

  void* sendbuff;
  void* recvbuff;

  if (devMalloc(&sendbuff, total_size) != 0) {
    fprintf(stderr, "Failed to allocate sendbuff\n");
    return 1;
  }
  if (devMalloc(&recvbuff, total_size) != 0) {
    fprintf(stderr, "Failed to allocate recvbuff\n");
    return 1;
  }

  // Allocate host buffers for initialization and verification
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, total_size));
  DEV_CHECK(devMallocHost(&h_verify, total_size));

  // Initialize sendbuff: rank r fills with float(r + 1)
  float* h_init_f = (float*)h_init;
  float fill_val = (float)(rank + 1);
  for (size_t i = 0; i < count; i++) {
    h_init_f[i] = fill_val;
  }
  DEV_CHECK(devMemcpySync(sendbuff, h_init, total_size, MemcpyDirection::H2D));

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Failed to create stream\n");
    return 1;
  }

  // Build IR program for AllReduce: Gather -> Reduce -> Broadcast
  //
  // Host chunk layout (20 chunks total):
  //   Chunks 0-15:  D2H destinations (4 ranks x 4 chunks each)
  //                 rank r, chunk i -> host chunk [r*4 + i]
  //   Chunks 16-19: H2H_REDUCE output (one per position)
  //
  // Step 1: D2H - each rank uploads its 4 chunks to host[r*4..r*4+3]
  // Step 2: H2H_REDUCE - rank r reduces position r across all ranks
  //         For each source rank s: reduce host[s*4+r] into host[16+r]
  // Step 3: H2D - each rank pulls all 4 reduced chunks back to device
  IRProgram program;
  program.input_chunk_count = NUM_CHUNKS;
  program.output_chunk_count = NUM_CHUNKS;

  // Step 1: D2H (4 instructions)
  for (int i = 0; i < NUM_CHUNKS; i++) {
    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = i;
    d2h.dst_chunk_idx = rank * NUM_CHUNKS + i;
    d2h.effects = {{rank * NUM_CHUNKS + i, 1}};
    program.instructions.push_back(d2h);
  }

  // Step 2: H2H_REDUCE (4 instructions)
  // Rank r reduces position r: gather from host[s*4+r] for all s, output to host[16+r]
  for (int s = 0; s < NUM_RANKS; s++) {
    Instruction reduce;
    reduce.op = OpCode::H2H_REDUCE;
    reduce.src_numa = 0;
    reduce.src_chunk_idx = s * NUM_CHUNKS + rank;
    reduce.dst_chunk_idx = 16 + rank;
    reduce.deps = {{0, s * NUM_CHUNKS + rank, 1}};
    reduce.effects = {{16 + rank, 1}};
    program.instructions.push_back(reduce);
  }

  // Step 3: H2D (4 instructions)
  // Each rank pulls all 4 reduced chunks from host[16..19] to recvbuff
  for (int i = 0; i < NUM_CHUNKS; i++) {
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = 16 + i;
    h2d.dst_chunk_idx = i;
    h2d.deps = {{0, 16 + i, 4}};
    program.instructions.push_back(h2d);
  }

  HostTimer timer;
  std::vector<float> times;
  double bus_bw_factor = 2.0 * (NUM_RANKS - 1) / NUM_RANKS;

  for (int iter = 0; iter < WARMUP_ITERS + MEASURE_ITERS; iter++) {
    timer.start();
    PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
    if (devSynchronizeStream(stream) != 0) {
      fprintf(stderr, "Failed to synchronize stream\n");
      return 1;
    }
    timer.end();

    float ms = timer.getElapsedMs();
    double alg_bw = calculateAlgBandwidth(total_size, ms);
    double bus_bw = alg_bw * bus_bw_factor;

    if (iter >= WARMUP_ITERS) {
      times.push_back(ms);
      printf("Iteration %d: %.2f ms, alg %.2f GB/s, bus %.2f GB/s\n", iter, ms, alg_bw, bus_bw);
    } else {
      printf("Iteration %d (warmup): %.2f ms, alg %.2f GB/s, bus %.2f GB/s\n", iter, ms, alg_bw, bus_bw);
    }
  }

  auto stats = calculateStats(times, total_size, bus_bw_factor);
  printf("\nAvg Time: %.2f ms, Alg BW: %.2f GB/s, Bus BW: %.2f GB/s\n", stats.avg_time_ms, stats.avg_bw_gbps,
         stats.avg_bus_bw_gbps);

  // Synchronize internal streams before verification
  pcclSynchronizeInternalStreams(comm);

  // Verify: every element in recvbuff should be 1+2+3+4 = 10.0f
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, total_size, MemcpyDirection::D2H));

  float expected = 0.0f;
  for (int r = 0; r < NUM_RANKS; r++) {
    expected += (float)(r + 1);
  }

  float* h_verify_f = (float*)h_verify;
  bool all_ok = true;
  for (size_t i = 0; i < count; i++) {
    if (h_verify_f[i] != expected) {
      fprintf(stderr, "Verification failed at recvbuff[%zu]: expected %.1f, got %.1f\n", i, expected, h_verify_f[i]);
      all_ok = false;
      break;
    }
  }

  if (all_ok) {
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
