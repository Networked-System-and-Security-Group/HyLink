#ifndef PCIECCL_PIPELINE_REDUCE_MANAGER_HPP
#define PCIECCL_PIPELINE_REDUCE_MANAGER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "ir.hpp"

namespace pccl {

// Forward declaration
class StateManager;

// Constants for pipeline reduce
constexpr int MAX_PIPELINE_DEPTH = 8;  // Maximum concurrent reducers
constexpr int MAX_H2H_THREADS = 8;    // Maximum threads per pipeline slot

//==============================================================================
// Pipeline Reduce Manager
//==============================================================================

// PipelineReduceManager: Manages pipeline parallelism for H2H operations
//
// Key features:
// 1. Round-robin subchunk assignment across threads with barrier synchronization
// 2. Per-slot progress tracking for inter-slot dependency ordering
// 3. Final-version tracking: only the slot achieving final_version updates shm.progress
// 4. All slots check their source chunk's shm.progress before reading
class PipelineReduceManager {
 public:
  explicit PipelineReduceManager(StateManager* state_manager);
  ~PipelineReduceManager();

  // Delete copy/move constructors and assignment operators
  PipelineReduceManager(const PipelineReduceManager&) = delete;
  PipelineReduceManager& operator=(const PipelineReduceManager&) = delete;
  PipelineReduceManager(PipelineReduceManager&&) = delete;
  PipelineReduceManager& operator=(PipelineReduceManager&&) = delete;

  // Submit a chunk-level copy task (multi-threaded round-robin execution)
  void submitChunkCopy(float* dst, const float* src, size_t count, int dst_chunk_idx, int dst_numa, int num_threads,
                       int src_numa, int src_chunk_idx, uint64_t version_delta);

  // Submit a chunk-level reduce task (multi-threaded round-robin execution)
  void submitChunkReduce(float* dst, const float* src, size_t count, int dst_chunk_idx, int dst_numa, int num_threads,
                         int src_numa, int src_chunk_idx, uint64_t version_delta);

  // Set the final version for a chunk (computed from IR by scheduler)
  void setFinalVersion(int chunk_idx, uint64_t final_ver);

  // Reset all chunk states (called from post-barrier cleanup path)
  void resetAllChunkStates();

 private:
  // Per-chunk state
  struct ChunkState {
    // Per-thread round completion count: thread_rounds[slot][tid] = rounds completed
    alignas(64) std::atomic<uint64_t> thread_rounds[MAX_PIPELINE_DEPTH][MAX_H2H_THREADS];

    // Number of threads per slot (atomic for cross-slot visibility)
    std::atomic<int> slot_num_threads[MAX_PIPELINE_DEPTH];

    // Per-slot counter for tracking last thread to finish (for final progress guarantee)
    std::atomic<int> threads_finished[MAX_PIPELINE_DEPTH];

    // Current pipeline depth (number of slots allocated in current batch)
    std::atomic<int> pipeline_depth{0};

    // Initial version when first reducer was assigned
    std::atomic<uint64_t> initial_version{0};

    // Final version for this chunk (set by scheduler from IR)
    uint64_t final_version{1};

    // Accumulated version so far (tracks each slot's contribution)
    std::atomic<uint64_t> accumulated_version{0};

    ChunkState() {
      for (int i = 0; i < MAX_PIPELINE_DEPTH; i++) {
        for (int j = 0; j < MAX_H2H_THREADS; j++) {
          thread_rounds[i][j].store(0, std::memory_order_relaxed);
        }
        slot_num_threads[i].store(0, std::memory_order_relaxed);
        threads_finished[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  // Custom deleter for aligned ChunkState allocation
  struct ChunkStateDeleter {
    void operator()(ChunkState* p) const {
      if (p) {
        p->~ChunkState();
        free(p);
      }
    }
  };

  // State manager (non-owning pointer)
  StateManager* state_manager_;

  // Map from chunk_idx to ChunkState
  std::map<int, std::unique_ptr<ChunkState, ChunkStateDeleter>> chunk_states_;
  std::mutex states_mutex_;

  // Get or create chunk state for a given chunk index
  ChunkState* getOrCreateChunkState(int chunk_idx);

  // Reset a single chunk's pipeline state
  void resetChunkState(ChunkState* chunk_state);

  // Core round-robin subchunk processing function
  void processSubchunksRoundRobin(ChunkState* chunk_state, int thread_id, int num_threads, float* dst, const float* src,
                                  size_t count, int pipeline_slot, int src_numa, int src_chunk_idx, int dst_numa,
                                  int dst_chunk_idx, bool is_final_slot, const char* op_name);
};

}  // namespace pccl

#endif  // PCIECCL_PIPELINE_REDUCE_MANAGER_HPP
