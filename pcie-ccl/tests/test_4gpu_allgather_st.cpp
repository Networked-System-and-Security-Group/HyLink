#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "include/comm.hpp"
#include "include/hal/device.hpp"
#include "include/ir.hpp"
#include "test_utils.hpp"

using namespace pccl;

static size_t parseSize(const char* str) {
  char* end;
  size_t val = strtoull(str, &end, 10);
  if (*end == 'K' || *end == 'k')
    val *= 1024;
  else if (*end == 'M' || *end == 'm')
    val *= 1024 * 1024;
  else if (*end == 'G' || *end == 'g')
    val *= 1024 * 1024 * 1024;
  return val;
}

static const int NUM_RANKS = 4;
static const char INIT_VALUES[NUM_RANKS] = {(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD};

struct ThreadResult {
  bool passed;
  double avg_time_ms;
  double avg_alg_bw_gbps;
  double avg_bus_bw_gbps;
  // Device event timing
  double avg_dev_time_ms;
  double avg_dev_alg_bw_gbps;
  double avg_dev_bus_bw_gbps;
};

static void rankThread(int rank, size_t size, ThreadResult* result) {
  result->passed = false;
  result->avg_time_ms = 0.0;
  result->avg_alg_bw_gbps = 0.0;
  result->avg_bus_bw_gbps = 0.0;
  result->avg_dev_time_ms = 0.0;
  result->avg_dev_alg_bw_gbps = 0.0;
  result->avg_dev_bus_bw_gbps = 0.0;

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, NUM_RANKS, &comm));

  void* sendbuff;
  void* recvbuff;
  size_t chunk_size = size;
  size_t count = chunk_size / sizeof(float);

  if (devMalloc(&sendbuff, chunk_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate sendbuff\n", rank);
    return;
  }
  if (devMalloc(&recvbuff, chunk_size * NUM_RANKS) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate recvbuff\n", rank);
    devFree(sendbuff);
    return;
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
    fprintf(stderr, "Rank %d: Failed to create stream\n", rank);
    return;
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
    if (j == rank)
      continue;
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = j;
    h2d.dst_chunk_idx = j;
    h2d.deps = {{0, j, 1}};
    program.instructions.push_back(h2d);
  }

  HostTimer timer;
  DeviceTimer dev_timer;
  std::vector<float> times;
  std::vector<float> dev_times;
  size_t alg_data_size = chunk_size * NUM_RANKS;
  double bus_bw_factor = (NUM_RANKS - 1.0) / NUM_RANKS;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    dev_timer.recordStart(stream);
    timer.start();
    PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
    dev_timer.recordEnd(stream);
    if (devSynchronizeStream(stream) != 0) {
      fprintf(stderr, "Rank %d: Failed to synchronize stream\n", rank);
      return;
    }
    timer.end();

    float ms = timer.getElapsedMs();
    float dev_ms = dev_timer.getElapsedMs();
    double alg_bw = calculateAlgBandwidth(alg_data_size, ms);
    double bus_bw = alg_bw * bus_bw_factor;
    double dev_alg_bw = calculateAlgBandwidth(alg_data_size, dev_ms);
    double dev_bus_bw = dev_alg_bw * bus_bw_factor;

    if (i >= WARMUP_ITERS) {
      times.push_back(ms);
      dev_times.push_back(dev_ms);
      printf("Rank %d, Iteration %d: host %.2f ms (alg %.2f, bus %.2f GB/s), dev %.2f ms (alg %.2f, bus %.2f GB/s)\n",
             rank, i, ms, alg_bw, bus_bw, dev_ms, dev_alg_bw, dev_bus_bw);
    } else {
      printf(
          "Rank %d, Iteration %d (warmup): host %.2f ms (alg %.2f, bus %.2f GB/s), dev %.2f ms (alg %.2f, bus %.2f "
          "GB/s)\n",
          rank, i, ms, alg_bw, bus_bw, dev_ms, dev_alg_bw, dev_bus_bw);
    }
  }

  auto stats = calculateStats(times, alg_data_size, bus_bw_factor);
  result->avg_time_ms = stats.avg_time_ms;
  result->avg_alg_bw_gbps = stats.avg_bw_gbps;
  result->avg_bus_bw_gbps = stats.avg_bus_bw_gbps;

  auto dev_stats = calculateStats(dev_times, alg_data_size, bus_bw_factor);
  result->avg_dev_time_ms = dev_stats.avg_time_ms;
  result->avg_dev_alg_bw_gbps = dev_stats.avg_bw_gbps;
  result->avg_dev_bus_bw_gbps = dev_stats.avg_bus_bw_gbps;

  // Synchronize internal streams before verification and freeing memory
  pcclSynchronizeInternalStreams(comm);

  // Verify: recvbuff[r] should contain rank r's init value for all r
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, chunk_size * NUM_RANKS, MemcpyDirection::D2H));

  bool all_ok = true;
  for (int r = 0; r < NUM_RANKS; r++) {
    char label[32];
    snprintf(label, sizeof(label), "rank%d_recvbuff[%d]", rank, r);
    bool ok = verifyData((char*)h_verify + chunk_size * r, chunk_size, INIT_VALUES[r], label);
    if (!ok)
      all_ok = false;
  }

  result->passed = all_ok;

  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  PCCL_CHECK(pcclDestroy(comm));
  DEV_CHECK(devFreeHost(h_init));
  DEV_CHECK(devFreeHost(h_verify));
}

int main(int argc, char** argv) {
  size_t size = argc > 1 ? parseSize(argv[1]) : DEFAULT_SIZE;

  printf("\n=== Test: 4-GPU AllGather (Single-Process Multi-Thread) ===\n");
  printf("Size: %.0f MB\n", size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  std::vector<ThreadResult> results(NUM_RANKS);
  std::vector<std::thread> threads;

  for (int r = 0; r < NUM_RANKS; r++) {
    threads.emplace_back(rankThread, r, size, &results[r]);
  }

  for (auto& t : threads) {
    t.join();
  }

  // Summarize results
  printf("\n=== Summary ===\n");
  bool all_passed = true;
  for (int r = 0; r < NUM_RANKS; r++) {
    printf("Rank %d: %s\n", r, results[r].passed ? "PASSED" : "FAILED");
    printf("  Host: Avg %.2f ms, Alg %.2f GB/s, Bus %.2f GB/s\n", results[r].avg_time_ms, results[r].avg_alg_bw_gbps,
           results[r].avg_bus_bw_gbps);
    printf("  Dev:  Avg %.2f ms, Alg %.2f GB/s, Bus %.2f GB/s\n", results[r].avg_dev_time_ms,
           results[r].avg_dev_alg_bw_gbps, results[r].avg_dev_bus_bw_gbps);
    if (!results[r].passed)
      all_passed = false;
  }

  printf("\nOverall: %s\n", all_passed ? "PASSED" : "FAILED");
  return all_passed ? 0 : 1;
}
