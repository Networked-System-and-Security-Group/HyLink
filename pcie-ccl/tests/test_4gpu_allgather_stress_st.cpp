#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "include/comm.hpp"
#include "include/hal/device.hpp"
#include "include/ir.hpp"
#include "test_utils.hpp"

using namespace pccl;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int NUM_RANKS = 4;

// Mixed-size phase cycles through these sizes
static const size_t MIXED_SIZES[] = {
    4096,            // 4KB   -- sub-subchunk, 1 partial slice
    1048576,         // 1MB   -- exactly 1 subchunk boundary
    1048576 + 4096,  // 1MB+4KB -- forces 2 subchunks
    64 * 1048576     // 64MB  -- 64 subchunks, full pipeline stress
};
static const int NUM_MIXED_SIZES = sizeof(MIXED_SIZES) / sizeof(MIXED_SIZES[0]);

// Largest buffer we ever need (per-chunk) across all phases
static const size_t MAX_CHUNK_SIZE = 64 * 1048576;  // 64 MB

// ---------------------------------------------------------------------------
// Iteration-unique fill byte
// ---------------------------------------------------------------------------
static inline uint8_t fillByte(int rank, int iteration) {
  uint8_t val = (uint8_t)(((iteration * NUM_RANKS + rank) * 137 + 43) & 0xFF);
  return val == 0 ? 1 : val;  // avoid zero (matches uninitialized memory)
}

// ---------------------------------------------------------------------------
// Size parser (supports K/M/G suffixes)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Phase configuration
// ---------------------------------------------------------------------------
struct PhaseConfig {
  std::string name;
  size_t fixed_size;  // 0 means mixed mode
  bool mixed;         // cycle through MIXED_SIZES[]
  bool staggered;     // introduce submission delays
  int num_iters;
};

// ---------------------------------------------------------------------------
// Per-rank result
// ---------------------------------------------------------------------------
struct RankResult {
  bool passed;
  int completed_iters;
};

// ---------------------------------------------------------------------------
// Verification with diagnostics
// ---------------------------------------------------------------------------
static bool verifyChunks(const void* h_verify, size_t chunk_size, int rank, int iter, int num_ranks,
                         bool abort_on_fail) {
  const uint8_t* data = (const uint8_t*)h_verify;
  bool all_ok = true;

  for (int r = 0; r < num_ranks; r++) {
    const uint8_t* chunk = data + chunk_size * r;
    uint8_t expected = fillByte(r, iter);
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;
    uint8_t first_actual = 0;

    for (size_t i = 0; i < chunk_size; i++) {
      if (chunk[i] != expected) {
        if (mismatch_count == 0) {
          first_mismatch = i;
          first_actual = chunk[i];
        }
        mismatch_count++;
      }
    }

    if (mismatch_count > 0) {
      all_ok = false;

      // Diagnose: stale data from a previous iteration?
      const char* tag = "";
      for (int prev_iter = iter - 1; prev_iter >= 0 && prev_iter >= iter - 5; prev_iter--) {
        if (first_actual == fillByte(r, prev_iter)) {
          tag = " [STALE DATA]";
          break;
        }
      }
      // Diagnose: wrong rank's data for this iteration?
      if (tag[0] == '\0') {
        for (int other = 0; other < num_ranks; other++) {
          if (other != r && first_actual == fillByte(other, iter)) {
            tag = " [WRONG RANK]";
            break;
          }
        }
      }

      fprintf(stderr,
              "FAIL: rank=%d iter=%d chunk[%d] size=%zu expected=0x%02X "
              "first_mismatch=%zu actual=0x%02X mismatches=%zu/%zu%s\n",
              rank, iter, r, chunk_size, (unsigned)expected, first_mismatch, (unsigned)first_actual, mismatch_count,
              chunk_size, tag);

      if (abort_on_fail)
        return false;
    }
  }
  return all_ok;
}

// ---------------------------------------------------------------------------
// Rank thread
// ---------------------------------------------------------------------------
static void rankThread(int rank, const PhaseConfig& phase, bool abort_on_fail, RankResult* result) {
  result->passed = false;
  result->completed_iters = 0;

  // Determine max chunk size for buffer allocation
  size_t max_chunk_size;
  if (phase.mixed) {
    max_chunk_size = MAX_CHUNK_SIZE;
  } else {
    max_chunk_size = phase.fixed_size;
  }

  // Init communicator
  pcclComm_t comm;
  PCCL_CHECK(pcclInit(rank, NUM_RANKS, &comm));

  // Allocate device buffers
  void* sendbuff;
  void* recvbuff;
  if (devMalloc(&sendbuff, max_chunk_size) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate sendbuff (%zu bytes)\n", rank, max_chunk_size);
    PCCL_CHECK(pcclDestroy(comm));
    return;
  }
  if (devMalloc(&recvbuff, max_chunk_size * NUM_RANKS) != 0) {
    fprintf(stderr, "Rank %d: Failed to allocate recvbuff (%zu bytes)\n", rank, max_chunk_size * NUM_RANKS);
    devFree(sendbuff);
    PCCL_CHECK(pcclDestroy(comm));
    return;
  }

  // Allocate host buffers
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, max_chunk_size));
  DEV_CHECK(devMallocHost(&h_verify, max_chunk_size * NUM_RANKS));

  // Create stream
  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Rank %d: Failed to create stream\n", rank);
    devFree(sendbuff);
    devFree(recvbuff);
    DEV_CHECK(devFreeHost(h_init));
    DEV_CHECK(devFreeHost(h_verify));
    PCCL_CHECK(pcclDestroy(comm));
    return;
  }

  // Build IR program (same structure for all iterations; count varies per submit)
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

  bool all_iters_ok = true;

  for (int iter = 0; iter < phase.num_iters; iter++) {
    // Determine size for this iteration
    size_t this_size;
    if (phase.mixed) {
      this_size = MIXED_SIZES[iter % NUM_MIXED_SIZES];
    } else {
      this_size = phase.fixed_size;
    }
    size_t count = this_size / sizeof(float);

    // Fill send buffer with iteration-unique pattern
    uint8_t fill = fillByte(rank, iter);
    memset(h_init, fill, this_size);
    DEV_CHECK(devMemcpySync(sendbuff, h_init, this_size, MemcpyDirection::H2D));

    // Clear recvbuff to catch stale reads (no devMemset in HAL, use host zero + H2D)
    memset(h_verify, 0, this_size * NUM_RANKS);
    DEV_CHECK(devMemcpySync(recvbuff, h_verify, this_size * NUM_RANKS, MemcpyDirection::H2D));

    // Staggered submission: rotate which rank is late
    if (phase.staggered) {
      int late_rank = iter % NUM_RANKS;
      if (rank == late_rank) {
        int delay_ms = (iter / NUM_RANKS) % 10 + 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      }
    }

    // Submit and wait
    PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
    if (devSynchronizeStream(stream) != 0) {
      fprintf(stderr, "Rank %d: Failed to synchronize stream at iter %d\n", rank, iter);
      all_iters_ok = false;
      break;
    }

    // Copy back for verification
    DEV_CHECK(devMemcpySync(h_verify, recvbuff, this_size * NUM_RANKS, MemcpyDirection::D2H));

    // Verify all chunks
    bool iter_ok = verifyChunks(h_verify, this_size, rank, iter, NUM_RANKS, abort_on_fail);
    if (!iter_ok) {
      all_iters_ok = false;
      if (abort_on_fail)
        break;
    }

    result->completed_iters = iter + 1;

    // Progress report (rank 0 only, every 10 iterations)
    if (rank == 0 && (iter + 1) % 10 == 0) {
      printf("  [progress] %d/%d iterations completed\n", iter + 1, phase.num_iters);
    }
  }

  // Sync internal streams before cleanup
  pcclSynchronizeInternalStreams(comm);

  // Cleanup
  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  DEV_CHECK(devFreeHost(h_init));
  DEV_CHECK(devFreeHost(h_verify));
  PCCL_CHECK(pcclDestroy(comm));

  result->passed = all_iters_ok;
}

// ---------------------------------------------------------------------------
// Run a single phase
// ---------------------------------------------------------------------------
static bool runPhase(const PhaseConfig& phase, bool abort_on_fail) {
  printf("\n--- Phase: %s ---\n", phase.name.c_str());
  if (phase.mixed) {
    printf("  Sizes: mixed {4KB, 1MB, 1MB+4KB, 64MB}\n");
  } else {
    if (phase.fixed_size >= 1048576) {
      printf("  Size: %.1f MB\n", phase.fixed_size / (1024.0 * 1024.0));
    } else {
      printf("  Size: %zu bytes\n", phase.fixed_size);
    }
  }
  printf("  Iterations: %d, Staggered: %s\n", phase.num_iters, phase.staggered ? "yes" : "no");

  std::vector<RankResult> results(NUM_RANKS);
  std::vector<std::thread> threads;

  for (int r = 0; r < NUM_RANKS; r++) {
    threads.emplace_back(rankThread, r, std::ref(phase), abort_on_fail, &results[r]);
  }
  for (auto& t : threads) {
    t.join();
  }

  // Print per-rank summary
  bool phase_passed = true;
  for (int r = 0; r < NUM_RANKS; r++) {
    printf("  Rank %d: %s (%d/%d iters)\n", r, results[r].passed ? "PASSED" : "FAILED", results[r].completed_iters,
           phase.num_iters);
    if (!results[r].passed)
      phase_passed = false;
  }
  printf("  Phase result: %s\n", phase_passed ? "PASSED" : "FAILED");
  return phase_passed;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char* prog) {
  printf("Usage: %s [iters] [size|\"mixed\"|\"staggered\"|\"mixed-staggered\"] [--no-abort]\n", prog);
  printf("\nExamples:\n");
  printf("  %s                          # 100 iters, all 7 phases\n", prog);
  printf("  %s 500                      # 500 iters per phase\n", prog);
  printf("  %s 200 64M                  # 200 iters, 64MB fixed only\n", prog);
  printf("  %s 200 mixed                # 200 iters, mixed sizes only\n", prog);
  printf("  %s 200 staggered            # 200 iters, staggered 64MB only\n", prog);
  printf("  %s 200 mixed-staggered      # 200 iters, mixed+staggered\n", prog);
  printf("  %s 100 1M --no-abort        # continue after failure\n", prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int num_iters = 100;
  bool abort_on_fail = true;
  bool have_size_arg = false;
  std::string size_arg;

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--no-abort") {
      abort_on_fail = false;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (!have_size_arg && (arg[0] >= '0' && arg[0] <= '9') && !have_size_arg && i == 1) {
      // First numeric arg is iterations
      num_iters = atoi(argv[i]);
    } else {
      // Second positional arg is size/mode
      have_size_arg = true;
      size_arg = arg;
    }
  }

  printf("\n=== Test: 4-GPU AllGather Stress (Single-Process Multi-Thread) ===\n");
  printf("Iterations per phase: %d\n", num_iters);
  printf("Abort on failure: %s\n", abort_on_fail ? "yes" : "no");

  // Build phase list
  std::vector<PhaseConfig> phases;

  if (have_size_arg) {
    // Single specific phase
    if (size_arg == "mixed") {
      phases.push_back({"Mixed sizes", 0, true, false, num_iters});
    } else if (size_arg == "staggered") {
      phases.push_back({"Staggered 64MB", MAX_CHUNK_SIZE, false, true, num_iters});
    } else if (size_arg == "mixed-staggered") {
      phases.push_back({"Mixed + staggered", 0, true, true, num_iters});
    } else {
      // Parse as a fixed size
      size_t sz = parseSize(size_arg.c_str());
      if (sz == 0 || sz % sizeof(float) != 0) {
        fprintf(stderr, "Invalid size: %s (must be >0 and multiple of %zu)\n", size_arg.c_str(), sizeof(float));
        return 1;
      }
      char name[64];
      if (sz >= 1048576) {
        snprintf(name, sizeof(name), "Fixed %.1f MB", sz / (1024.0 * 1024.0));
      } else if (sz >= 1024) {
        snprintf(name, sizeof(name), "Fixed %zu KB", sz / 1024);
      } else {
        snprintf(name, sizeof(name), "Fixed %zu B", sz);
      }
      phases.push_back({name, sz, false, false, num_iters});
    }
  } else {
    // Default: run all 7 phases
    phases.push_back({"4KB (sub-subchunk)", 4096, false, false, num_iters});
    phases.push_back({"1MB (subchunk boundary)", 1048576, false, false, num_iters});
    phases.push_back({"1MB+4KB (2 subchunks)", 1048576 + 4096, false, false, num_iters});
    phases.push_back({"64MB (full pipeline)", MAX_CHUNK_SIZE, false, false, num_iters});
    phases.push_back({"Mixed sizes", 0, true, false, num_iters});
    phases.push_back({"Staggered 64MB", MAX_CHUNK_SIZE, false, true, num_iters});
    phases.push_back({"Mixed + staggered", 0, true, true, num_iters});
  }

  printf("Phases: %zu\n", phases.size());

  // Run all phases
  bool overall_passed = true;
  int phases_passed = 0;
  int phases_failed = 0;

  for (size_t p = 0; p < phases.size(); p++) {
    bool ok = runPhase(phases[p], abort_on_fail);
    if (ok) {
      phases_passed++;
    } else {
      phases_failed++;
      overall_passed = false;
    }
  }

  // Final summary
  printf("\n=== Overall Summary ===\n");
  printf("Phases passed: %d/%zu\n", phases_passed, phases.size());
  if (phases_failed > 0)
    printf("Phases failed: %d\n", phases_failed);
  printf("Result: %s\n", overall_passed ? "PASSED" : "FAILED");

  return overall_passed ? 0 : 1;
}
