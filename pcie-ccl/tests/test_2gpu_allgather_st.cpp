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

static const int NUM_RANKS = 2;

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

static void rankThread(int rank, size_t chunk_size, ThreadResult* result) {
  result->passed = false;
  result->avg_time_ms = 0.0;
  result->avg_alg_bw_gbps = 0.0;
  result->avg_bus_bw_gbps = 0.0;
  result->avg_dev_time_ms = 0.0;
  result->avg_dev_alg_bw_gbps = 0.0;
  result->avg_dev_bus_bw_gbps = 0.0;

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, NUM_RANKS, &comm));

  // AllGather: each rank contributes 1 chunk, output has NUM_RANKS chunks
  size_t count = chunk_size / sizeof(float);
  size_t total_recv_size = chunk_size * NUM_RANKS;

  void* sendbuff;
  void* recvbuff;

  if (devMalloc(&sendbuff, chunk_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate sendbuff\n", rank);
    return;
  }
  if (devMalloc(&recvbuff, total_recv_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate recvbuff\n", rank);
    devFree(sendbuff);
    return;
  }

  // Allocate host buffers for initialization and verification
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, chunk_size));
  DEV_CHECK(devMallocHost(&h_verify, total_recv_size));

  // Initialize sendbuff: rank 0 -> 0xAA, rank 1 -> 0xBB
  char init_value = (rank == 0) ? 0xAA : 0xBB;
  memset(h_init, init_value, chunk_size);
  DEV_CHECK(devMemcpySync(sendbuff, h_init, chunk_size, MemcpyDirection::H2D));

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Rank %d: Failed to create stream\n", rank);
    return;
  }

  // Build IR program for 2-rank AllGather
  //
  // Host chunk layout (shared memory):
  //   Chunk 0: rank 0's D2H output
  //   Chunk 1: rank 1's D2H output
  //
  // Each rank:
  //   1. D2H: copy own data to host chunk[rank]
  //   2. D2D: copy own data to recvbuff[rank]
  //   3. H2D: copy remote rank's host chunk to recvbuff[remote_rank]
  IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = NUM_RANKS;

  if (rank == 0) {
    // D2H: GPU chunk 0 -> host chunk 0
    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = 0;
    d2h.dst_chunk_idx = 0;
    d2h.effects = {{0}};

    // D2D: GPU sendbuff chunk 0 -> GPU recvbuff chunk 0
    Instruction d2d;
    d2d.op = OpCode::D2D;
    d2d.src_chunk_idx = 0;
    d2d.dst_chunk_idx = 0;

    // H2D: host chunk 1 (rank 1's data) -> GPU recvbuff chunk 1
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = 1;
    h2d.dst_chunk_idx = 1;
    h2d.deps = {{0, 1, 1}};

    program.instructions = {d2h, d2d, h2d};
  } else {
    // D2H: GPU chunk 0 -> host chunk 1
    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = 0;
    d2h.dst_chunk_idx = 1;
    d2h.effects = {{1}};

    // D2D: GPU sendbuff chunk 0 -> GPU recvbuff chunk 1
    Instruction d2d;
    d2d.op = OpCode::D2D;
    d2d.src_chunk_idx = 0;
    d2d.dst_chunk_idx = 1;

    // H2D: host chunk 0 (rank 0's data) -> GPU recvbuff chunk 0
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = 0;
    h2d.dst_chunk_idx = 0;
    h2d.deps = {{0, 0, 1}};

    program.instructions = {d2h, d2d, h2d};
  }

  HostTimer timer;
  DeviceTimer dev_timer;
  std::vector<float> times;
  std::vector<float> dev_times;
  size_t alg_data_size = total_recv_size;
  double bus_bw_factor = (NUM_RANKS - 1.0) / NUM_RANKS;

  for (int iter = 0; iter < WARMUP_ITERS + MEASURE_ITERS; iter++) {
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

    if (iter >= WARMUP_ITERS) {
      times.push_back(ms);
      dev_times.push_back(dev_ms);
      printf("Rank %d, Iteration %d: host %.2f ms (alg %.2f, bus %.2f GB/s), dev %.2f ms (alg %.2f, bus %.2f GB/s)\n",
             rank, iter, ms, alg_bw, bus_bw, dev_ms, dev_alg_bw, dev_bus_bw);
    } else {
      printf(
          "Rank %d, Iteration %d (warmup): host %.2f ms (alg %.2f, bus %.2f GB/s), dev %.2f ms (alg %.2f, bus %.2f "
          "GB/s)\n",
          rank, iter, ms, alg_bw, bus_bw, dev_ms, dev_alg_bw, dev_bus_bw);
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

  // Synchronize internal streams before verification
  pcclSynchronizeInternalStreams(comm);

  // Verify: recvbuff[0] = rank 0's data (0xAA), recvbuff[1] = rank 1's data (0xBB)
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, total_recv_size, MemcpyDirection::D2H));

  bool chunk0_ok = verifyData((char*)h_verify, chunk_size, (char)0xAA, "recvbuff[0]");
  bool chunk1_ok = verifyData((char*)h_verify + chunk_size, chunk_size, (char)0xBB, "recvbuff[1]");
  result->passed = chunk0_ok && chunk1_ok;

  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  PCCL_CHECK(pcclDestroy(comm));
  DEV_CHECK(devFreeHost(h_init));
  DEV_CHECK(devFreeHost(h_verify));
}

int main(int argc, char** argv) {
  size_t size = argc > 1 ? parseSize(argv[1]) : DEFAULT_SIZE;

  printf("\n=== Test: 2-GPU AllGather (Single-Process Multi-Thread) ===\n");
  printf("Size per rank: %.0f MB, Total: %.0f MB\n", size / (1024.0 * 1024.0),
         size * NUM_RANKS / (1024.0 * 1024.0));
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
