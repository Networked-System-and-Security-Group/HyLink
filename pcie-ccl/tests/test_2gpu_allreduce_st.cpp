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
static const int NUM_CHUNKS = 2;

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

  // Each rank has `size` bytes total, split into NUM_CHUNKS chunks
  size_t total_size = size;
  size_t count = total_size / sizeof(float);

  void* sendbuff;
  void* recvbuff;

  if (devMalloc(&sendbuff, total_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate sendbuff\n", rank);
    return;
  }
  if (devMalloc(&recvbuff, total_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate recvbuff\n", rank);
    devFree(sendbuff);
    return;
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
    fprintf(stderr, "Rank %d: Failed to create stream\n", rank);
    return;
  }

  // Build IR program for 2-rank AllReduce (reduce-scatter + allgather pattern)
  //
  // Host chunk layout:
  //   Chunks 0-1: rank 0's D2H output
  //   Chunks 2-3: rank 1's D2H output
  //   Chunk 4:    reduced result for position 0 (all ranks' chunk 0 combined)
  //   Chunk 5:    reduced result for position 1 (all ranks' chunk 1 combined)
  //
  // Each rank reduces one position, then both ranks H2D all reduced chunks.
  // Reduce output chunks (4-5) are separate from D2H chunks (0-3) to avoid
  // version conflicts between D2H and H2H_REDUCE on the same chunk.
  IRProgram program;
  program.input_chunk_count = NUM_CHUNKS;
  program.output_chunk_count = NUM_CHUNKS;

  // Step 1: D2H — each rank copies its GPU chunks to host
  for (int i = 0; i < NUM_CHUNKS; i++) {
    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = i;
    d2h.dst_chunk_idx = rank * NUM_CHUNKS + i;
    d2h.effects = {{rank * NUM_CHUNKS + i, 1}};
    program.instructions.push_back(d2h);
  }

  // Step 2: H2H_REDUCE — rank r reduces position r
  // Rank 0 reduces chunks {0, 2} → chunk 4 (position 0)
  // Rank 1 reduces chunks {1, 3} → chunk 5 (position 1)
  for (int s = 0; s < NUM_RANKS; s++) {
    Instruction reduce;
    reduce.op = OpCode::H2H_REDUCE;
    reduce.src_numa = 0;
    reduce.src_chunk_idx = s * NUM_CHUNKS + rank;
    reduce.dst_chunk_idx = NUM_RANKS * NUM_CHUNKS + rank;
    reduce.deps = {{0, s * NUM_CHUNKS + rank, 1}};
    reduce.effects = {{NUM_RANKS * NUM_CHUNKS + rank, 1}};
    program.instructions.push_back(reduce);
  }

  // Step 3: H2D — each rank pulls all reduced chunks back to GPU
  for (int i = 0; i < NUM_CHUNKS; i++) {
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = NUM_RANKS * NUM_CHUNKS + i;
    h2d.dst_chunk_idx = i;
    h2d.deps = {{0, NUM_RANKS * NUM_CHUNKS + i, NUM_RANKS}};
    program.instructions.push_back(h2d);
  }

  HostTimer timer;
  DeviceTimer dev_timer;
  std::vector<float> times;
  std::vector<float> dev_times;
  double bus_bw_factor = 2.0 * (NUM_RANKS - 1) / NUM_RANKS;

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
    double alg_bw = calculateAlgBandwidth(total_size, ms);
    double bus_bw = alg_bw * bus_bw_factor;
    double dev_alg_bw = calculateAlgBandwidth(total_size, dev_ms);
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

  auto stats = calculateStats(times, total_size, bus_bw_factor);
  result->avg_time_ms = stats.avg_time_ms;
  result->avg_alg_bw_gbps = stats.avg_bw_gbps;
  result->avg_bus_bw_gbps = stats.avg_bus_bw_gbps;

  auto dev_stats = calculateStats(dev_times, total_size, bus_bw_factor);
  result->avg_dev_time_ms = dev_stats.avg_time_ms;
  result->avg_dev_alg_bw_gbps = dev_stats.avg_bw_gbps;
  result->avg_dev_bus_bw_gbps = dev_stats.avg_bus_bw_gbps;

  // Synchronize internal streams before verification
  pcclSynchronizeInternalStreams(comm);

  // Verify: every element in recvbuff should be 1+2 = 3.0f
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, total_size, MemcpyDirection::D2H));

  float expected = 0.0f;
  for (int r = 0; r < NUM_RANKS; r++) {
    expected += (float)(r + 1);
  }

  float* h_verify_f = (float*)h_verify;
  bool all_ok = true;
  for (size_t i = 0; i < count; i++) {
    if (h_verify_f[i] != expected) {
      fprintf(stderr, "Rank %d: Verification failed at recvbuff[%zu]: expected %.1f, got %.1f\n", rank, i, expected,
              h_verify_f[i]);
      all_ok = false;
      break;
    }
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

  printf("\n=== Test: 2-GPU AllReduce (Single-Process Multi-Thread) ===\n");
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
