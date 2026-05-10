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

// -----------------------------------------------------------
// Configuration & Result Structures
// -----------------------------------------------------------

struct BenchmarkConfig {
  enum Operation { ALLGATHER, ALLREDUCE };
  Operation op;
  int num_gpus;
  size_t min_size;
  size_t max_size;
  int warmup_iters;
  int measure_iters;
};

struct SizeResult {
  size_t size;
  double avg_time_ms;
  double alg_bw_gbps;
  double bus_bw_gbps;
  bool correct;
};

// -----------------------------------------------------------
// Helpers
// -----------------------------------------------------------

static size_t parseSize(const char* str) {
  char* end;
  size_t val = strtoull(str, &end, 10);
  if (*end == 'K' || *end == 'k')
    val *= 1024;
  else if (*end == 'M' || *end == 'm')
    val *= 1024ULL * 1024;
  else if (*end == 'G' || *end == 'g')
    val *= 1024ULL * 1024 * 1024;
  return val;
}

static const char INIT_VALUES[] = {(char)0xAA, (char)0xBB, (char)0xCC, (char)0xDD,
                                   (char)0xEE, (char)0xFF, (char)0x11, (char)0x22};

// -----------------------------------------------------------
// IR Program Builders
// -----------------------------------------------------------

static IRProgram buildAllGatherProgram(int rank, int num_ranks) {
  IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = num_ranks;

  // D2H: upload own data to host chunk[rank]
  Instruction d2h;
  d2h.op = OpCode::D2H;
  d2h.src_chunk_idx = 0;
  d2h.dst_chunk_idx = rank;
  d2h.effects = {{rank}};
  program.instructions.push_back(d2h);

  // D2D: local copy to recvbuff[rank]
  Instruction d2d;
  d2d.op = OpCode::D2D;
  d2d.src_chunk_idx = 0;
  d2d.dst_chunk_idx = rank;
  program.instructions.push_back(d2d);

  // H2D: pull remote ranks' data from host
  for (int j = 0; j < num_ranks; j++) {
    if (j == rank)
      continue;
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = j;
    h2d.dst_chunk_idx = j;
    h2d.deps = {{0, j, 1}};
    program.instructions.push_back(h2d);
  }

  return program;
}

static IRProgram buildAllReduceProgram(int rank, int num_ranks) {
  int nc = num_ranks;  // num_chunks = num_ranks

  IRProgram program;
  program.input_chunk_count = nc;
  program.output_chunk_count = nc;

  // Step 1: D2H — each rank copies all its chunks to host
  for (int i = 0; i < nc; i++) {
    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = i;
    d2h.dst_chunk_idx = rank * nc + i;
    d2h.effects = {{rank * nc + i, 1}};
    program.instructions.push_back(d2h);
  }

  // Step 2: H2H_REDUCE — each rank reduces its assigned position
  for (int s = 0; s < num_ranks; s++) {
    Instruction reduce;
    reduce.op = OpCode::H2H_REDUCE;
    reduce.src_numa = 0;
    reduce.src_chunk_idx = s * nc + rank;
    reduce.dst_chunk_idx = num_ranks * nc + rank;
    reduce.deps = {{0, s * nc + rank, 1}};
    reduce.effects = {{num_ranks * nc + rank, 1}};
    program.instructions.push_back(reduce);
  }

  // Step 3: H2D — each rank pulls all reduced chunks back to GPU
  for (int i = 0; i < nc; i++) {
    Instruction h2d;
    h2d.op = OpCode::H2D;
    h2d.src_chunk_idx = num_ranks * nc + i;
    h2d.dst_chunk_idx = i;
    h2d.deps = {{0, num_ranks * nc + i, (uint64_t)num_ranks}};
    program.instructions.push_back(h2d);
  }

  return program;
}

// -----------------------------------------------------------
// Per-Rank Benchmark Thread
// -----------------------------------------------------------

static void rankBenchThread(int rank, BenchmarkConfig config, std::vector<SizeResult>* results) {
  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, config.num_gpus, &comm));

  int n = config.num_gpus;

  // Compute max buffer sizes (allocate once, reuse for all sizes)
  size_t max_sendbuff_size, max_recvbuff_size, max_verify_size;
  if (config.op == BenchmarkConfig::ALLGATHER) {
    max_sendbuff_size = config.max_size / n;
    max_recvbuff_size = config.max_size;
    max_verify_size = config.max_size;
  } else {
    max_sendbuff_size = config.max_size;
    max_recvbuff_size = config.max_size;
    max_verify_size = config.max_size;
  }

  // Allocate device buffers
  void* sendbuff;
  void* recvbuff;
  if (devMalloc(&sendbuff, max_sendbuff_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate sendbuff (%zu bytes)\n", rank, max_sendbuff_size);
    pcclDestroy(comm);
    return;
  }
  if (devMalloc(&recvbuff, max_recvbuff_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate recvbuff (%zu bytes)\n", rank, max_recvbuff_size);
    devFree(sendbuff);
    pcclDestroy(comm);
    return;
  }

  // Allocate pinned host buffers
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, max_sendbuff_size));
  DEV_CHECK(devMallocHost(&h_verify, max_verify_size));

  // Build IR program once (size-independent)
  IRProgram program;
  if (config.op == BenchmarkConfig::ALLGATHER) {
    program = buildAllGatherProgram(rank, n);
  } else {
    program = buildAllReduceProgram(rank, n);
  }

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Rank %d: Failed to create stream\n", rank);
    devFree(sendbuff);
    devFree(recvbuff);
    devFreeHost(h_init);
    devFreeHost(h_verify);
    pcclDestroy(comm);
    return;
  }

  // Bandwidth formula parameters
  double bus_bw_factor;
  if (config.op == BenchmarkConfig::ALLGATHER) {
    bus_bw_factor = (n - 1.0) / n;
  } else {
    bus_bw_factor = 2.0 * (n - 1) / n;
  }

  HostTimer timer;

  // --- Size sweep ---
  for (size_t size = config.min_size; size <= config.max_size; size *= 2) {
    // Compute per-size parameters
    size_t per_rank_bytes, sendbuff_size, recvbuff_size, alg_data_size;
    size_t count;

    if (config.op == BenchmarkConfig::ALLGATHER) {
      // S = total output size, each rank inputs S/n bytes
      per_rank_bytes = size / n;
      sendbuff_size = per_rank_bytes;
      recvbuff_size = size;
      alg_data_size = size;
      count = per_rank_bytes / sizeof(float);
    } else {
      // S = per-rank array size
      per_rank_bytes = size;
      sendbuff_size = size;
      recvbuff_size = size;
      alg_data_size = size;
      count = size / sizeof(float);
    }

    // Initialize sendbuff
    if (config.op == BenchmarkConfig::ALLGATHER) {
      memset(h_init, INIT_VALUES[rank], sendbuff_size);
    } else {
      float* h_f = (float*)h_init;
      float fill_val = (float)(rank + 1);
      for (size_t i = 0; i < count; i++)
        h_f[i] = fill_val;
    }
    DEV_CHECK(devMemcpySync(sendbuff, h_init, sendbuff_size, MemcpyDirection::H2D));

    // Run warmup + measurement iterations
    std::vector<float> times;
    int total_iters = config.warmup_iters + config.measure_iters;

    for (int iter = 0; iter < total_iters; iter++) {
      timer.start();
      PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
      if (devSynchronizeStream(stream) != 0) {
        fprintf(stderr, "Rank %d: Stream sync failed at size %zu iter %d\n", rank, size, iter);
        return;
      }
      timer.end();

      float ms = timer.getElapsedMs();
      double alg_bw = calculateAlgBandwidth(alg_data_size, ms);
      double bus_bw = alg_bw * bus_bw_factor;

      if (iter < config.warmup_iters) {
        printf("Rank %d, Size %zu, Iter %d (warmup): %.3f ms, alg %.4f GB/s, bus %.4f GB/s\n", rank, size, iter, ms,
               alg_bw, bus_bw);
      } else {
        times.push_back(ms);
        printf("Rank %d, Size %zu, Iter %d: %.3f ms, alg %.4f GB/s, bus %.4f GB/s\n", rank, size, iter, ms, alg_bw,
               bus_bw);
      }
    }

    // Synchronize internal streams before verification
    pcclSynchronizeInternalStreams(comm);

    // Verify correctness
    DEV_CHECK(devMemcpySync(h_verify, recvbuff, recvbuff_size, MemcpyDirection::D2H));
    bool correct = true;

    if (config.op == BenchmarkConfig::ALLGATHER) {
      for (int r = 0; r < n; r++) {
        char label[64];
        snprintf(label, sizeof(label), "rank%d_recv[%d]", rank, r);
        if (!verifyData((char*)h_verify + per_rank_bytes * r, per_rank_bytes, INIT_VALUES[r], label))
          correct = false;
      }
    } else {
      float expected = 0.0f;
      for (int r = 0; r < n; r++)
        expected += (float)(r + 1);
      float* vf = (float*)h_verify;
      for (size_t i = 0; i < count; i++) {
        if (vf[i] != expected) {
          fprintf(stderr, "Rank %d, Size %zu: verify failed at [%zu] expected %.1f got %.1f\n", rank, size, i, expected,
                  vf[i]);
          correct = false;
          break;
        }
      }
    }

    // Compute statistics
    auto stats = calculateStats(times, alg_data_size, bus_bw_factor);

    SizeResult sr;
    sr.size = size;
    sr.avg_time_ms = stats.avg_time_ms;
    sr.alg_bw_gbps = stats.avg_bw_gbps;
    sr.bus_bw_gbps = stats.avg_bus_bw_gbps;
    sr.correct = correct;
    results->push_back(sr);
  }

  // Cleanup
  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  DEV_CHECK(devFreeHost(h_init));
  DEV_CHECK(devFreeHost(h_verify));
  PCCL_CHECK(pcclDestroy(comm));
}

// -----------------------------------------------------------
// CLI Parsing
// -----------------------------------------------------------

static void printUsage(const char* prog) {
  printf("Usage: %s <allgather|allreduce> <num_gpus> [options]\n", prog);
  printf("Options:\n");
  printf("  --min-size SIZE   Minimum data size [default: 1K]\n");
  printf("  --max-size SIZE   Maximum data size [default: 1G]\n");
  printf("  --warmup N        Warmup iterations [default: 5]\n");
  printf("  --iters N         Measurement iterations [default: 20]\n");
}

static bool parseArgs(int argc, char** argv, BenchmarkConfig& config) {
  if (argc < 3) {
    printUsage(argv[0]);
    return false;
  }

  // Parse operation
  if (strcmp(argv[1], "allgather") == 0) {
    config.op = BenchmarkConfig::ALLGATHER;
  } else if (strcmp(argv[1], "allreduce") == 0) {
    config.op = BenchmarkConfig::ALLREDUCE;
  } else {
    fprintf(stderr, "Unknown operation: %s (use allgather or allreduce)\n", argv[1]);
    return false;
  }

  // Parse num_gpus
  config.num_gpus = atoi(argv[2]);
  if (config.num_gpus < 2 || config.num_gpus > MAX_RANKS) {
    fprintf(stderr, "num_gpus must be between 2 and %d\n", MAX_RANKS);
    return false;
  }

  // AllReduce requires num_gpus*(num_gpus+1) host chunks <= MAX_CHUNKS
  if (config.op == BenchmarkConfig::ALLREDUCE && config.num_gpus * (config.num_gpus + 1) > MAX_CHUNKS) {
    fprintf(stderr, "AllReduce with %d GPUs requires %d host chunks (max %d)\n", config.num_gpus,
            config.num_gpus * (config.num_gpus + 1), MAX_CHUNKS);
    return false;
  }

  // Defaults
  config.min_size = 1024;
  config.max_size = 1024ULL * 1024 * 1024;
  config.warmup_iters = 5;
  config.measure_iters = 20;

  // Parse optional arguments
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--min-size") == 0 && i + 1 < argc) {
      config.min_size = parseSize(argv[++i]);
    } else if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
      config.max_size = parseSize(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      config.warmup_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      config.measure_iters = atoi(argv[++i]);
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      printUsage(argv[0]);
      return false;
    }
  }

  if (config.measure_iters < 1) {
    fprintf(stderr, "Must have at least 1 measurement iteration\n");
    return false;
  }

  // Host memory pool has a fixed 1GB per-chunk capacity.
  // The scheduler computes chunk_size = count * sizeof(float) / input_chunk_count.
  // For both AllGather (input_chunk_count=1, count=size/n/sizeof(float))
  // and AllReduce (input_chunk_count=n, count=size/sizeof(float)),
  // the per-chunk size is size/n, so max_size must be <= 1GB * n.
  size_t host_chunk_capacity = 1024ULL * 1024 * 1024;
  size_t max_allowed = host_chunk_capacity * config.num_gpus;
  if (config.max_size > max_allowed) {
    fprintf(stderr, "Note: capping max_size from %zu to %zu (host pool: 1GB/chunk, %d GPUs)\n", config.max_size,
            max_allowed, config.num_gpus);
    config.max_size = max_allowed;
  }

  return true;
}

// -----------------------------------------------------------
// Main
// -----------------------------------------------------------

int main(int argc, char** argv) {
  BenchmarkConfig config;
  if (!parseArgs(argc, argv, config))
    return 1;

  const char* op_name = (config.op == BenchmarkConfig::ALLGATHER) ? "AllGather" : "AllReduce";
  int n = config.num_gpus;

  printf("\n=== PCIeCCL %s %d-GPU Performance Benchmark ===\n", op_name, n);
  printf("Size range: %zu - %zu bytes\n", config.min_size, config.max_size);
  printf("Warmup: %d iters, Measure: %d iters\n\n", config.warmup_iters, config.measure_iters);

  // Launch rank threads
  std::vector<std::vector<SizeResult>> all_results(n);
  std::vector<std::thread> threads;

  for (int r = 0; r < n; r++) {
    threads.emplace_back(rankBenchThread, r, config, &all_results[r]);
  }

  for (auto& t : threads)
    t.join();

  // Check results collected
  if (all_results[0].empty()) {
    fprintf(stderr, "No results collected\n");
    return 1;
  }

  // --- Per-size summary ---
  size_t num_sizes = all_results[0].size();
  for (size_t si = 0; si < num_sizes; si++) {
    printf("\n--- Size %zu Summary (%d iters) ---\n", all_results[0][si].size,
           config.measure_iters);
    for (int r = 0; r < n; r++) {
      const auto& sr = all_results[r][si];
      printf("  Rank %d: Avg %.3f ms, Alg %.4f GB/s, Bus %.4f GB/s  [%s]\n", r, sr.avg_time_ms, sr.alg_bw_gbps,
             sr.bus_bw_gbps, sr.correct ? "PASS" : "FAIL");
    }
  }

  // --- Final NCCL-style summary table (rank 0 results) ---
  printf("\n# PCIeCCL %s %d-GPU Performance (avg)\n", op_name, n);
  printf("#%16s%13s%14s%14s%9s\n", "Size(B)", "Time(ms)", "AlgBW(GB/s)", "BusBW(GB/s)", "Verify");

  bool all_passed = true;
  for (size_t si = 0; si < num_sizes; si++) {
    const auto& sr = all_results[0][si];
    printf("%17zu%13.3f%14.4f%14.4f%9s\n", sr.size, sr.avg_time_ms, sr.alg_bw_gbps, sr.bus_bw_gbps,
           sr.correct ? "PASS" : "FAIL");
    for (int r = 0; r < n; r++) {
      if (!all_results[r][si].correct)
        all_passed = false;
    }
  }

  printf("\nOverall: %s\n", all_passed ? "PASSED" : "FAILED");
  return all_passed ? 0 : 1;
}
