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
static const char INIT_VALUES[NUM_RANKS] = {(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD};

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

  printf("\n=== Test: 4-GPU AllGather ===\n");
  printf("Rank: %d, Size: %.0f MB\n", rank, size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, NUM_RANKS, &comm));

  void* sendbuff;
  void* recvbuff;
  size_t chunk_size = size;
  size_t count = chunk_size / sizeof(float);

  if (devMalloc(&sendbuff, chunk_size) != 0) {
    fprintf(stderr, "Failed to allocate sendbuff\n");
    return 1;
  }
  if (devMalloc(&recvbuff, chunk_size * NUM_RANKS) != 0) {
    fprintf(stderr, "Failed to allocate recvbuff\n");
    return 1;
  }

  // Allocate host buffers for initialization and verification
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, chunk_size));
  DEV_CHECK(devMallocHost(&h_verify, chunk_size * NUM_RANKS));

  // Initialize sendbuff with rank-specific value
  memset(h_init, INIT_VALUES[rank], chunk_size);
  DEV_CHECK(devMemcpySync(sendbuff, h_init, chunk_size, MemcpyDirection::H2D));

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Failed to create stream\n");
    return 1;
  }

  // Build IR program generically for any rank
  IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = NUM_RANKS;

  // D2H: upload own data to host chunk[rank]
  Instruction d2h;
  d2h.op = OpCode::D2H;
  d2h.src_chunk_idx = 0;
  d2h.dst_chunk_idx = rank;
  d2h.effects = {{rank}};
  program.instructions.push_back(d2h);

  // D2D: local copy of own data to recvbuff[rank]
  Instruction d2d;
  d2d.op = OpCode::D2D;
  d2d.src_chunk_idx = 0;
  d2d.dst_chunk_idx = rank;
  program.instructions.push_back(d2d);

  // H2D: pull other ranks' data from host
  for (int j = 0; j < NUM_RANKS; j++) {
    if (j == rank) continue;
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = j;
    h2d.dst_chunk_idx = j;
    h2d.deps = {{0, j, 1}};
    program.instructions.push_back(h2d);
  }

  HostTimer timer;
  std::vector<float> times;
  size_t alg_data_size = chunk_size * NUM_RANKS;
  double bus_bw_factor = (NUM_RANKS - 1.0) / NUM_RANKS;

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

  // Verify: recvbuff[r] should contain rank r's init value for all r
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, chunk_size * NUM_RANKS, MemcpyDirection::D2H));

  bool all_ok = true;
  for (int r = 0; r < NUM_RANKS; r++) {
    char label[32];
    snprintf(label, sizeof(label), "recvbuff[%d]", r);
    bool ok = verifyData((char*)h_verify + chunk_size * r, chunk_size, INIT_VALUES[r], label);
    if (!ok) all_ok = false;
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
