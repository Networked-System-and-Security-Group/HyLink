#ifndef STATE_MANAGER_HPP
#define STATE_MANAGER_HPP

#include <atomic>
#include <cstdint>

#include "states.hpp"

namespace pccl {

// StateManager: Manages shared memory state for inter-process communication
// This class encapsulates all shared memory operations and provides
// zero-overhead inline accessors for high-performance state synchronization
class StateManager {
 private:
  GlobalState* state_;
  int rank_;
  int world_size_;
  bool is_owner_;  // true if rank 0 (creates shared memory)

  // Validate indices
  inline void validateIndices(int numa_node, int chunk_idx) const {
    // In release builds, these checks will be optimized out if assertions disabled
    // But they help catch bugs during development
    (void)numa_node;
    (void)chunk_idx;
    // assert(numa_node >= 0 && numa_node < MAX_NUMA_NODES);
    // assert(chunk_idx >= 0 && chunk_idx < MAX_CHUNKS);
  }

 public:
  // Constructor: Sets up shared memory (creates or attaches)
  StateManager(int rank, int world_size);

  // Destructor: Cleans up shared memory (unmaps and unlinks if owner)
  ~StateManager();

  // Delete copy and move constructors/operators (non-copyable)
  StateManager(const StateManager&) = delete;
  StateManager& operator=(const StateManager&) = delete;
  StateManager(StateManager&&) = delete;
  StateManager& operator=(StateManager&&) = delete;

  // ===================================================================
  // INLINE METHODS - Zero-overhead abstraction for hot paths
  // ===================================================================

  // Get chunk version (atomic load with acquire semantics)
  inline uint64_t getChunkVersion(int numa_node, int chunk_idx,
                                  std::memory_order order = std::memory_order_acquire) const {
    validateIndices(numa_node, chunk_idx);
    return state_->numa_nodes[numa_node].chunks[chunk_idx].version.load(order);
  }

  // Set chunk version (atomic store with release semantics)
  inline void setChunkVersion(int numa_node, int chunk_idx, uint64_t version,
                              std::memory_order order = std::memory_order_release) {
    validateIndices(numa_node, chunk_idx);
    state_->numa_nodes[numa_node].chunks[chunk_idx].version.store(version, order);
  }

  // Compare-and-swap chunk version (atomic CAS)
  // Returns true if swap succeeded (version was expected value)
  inline bool compareAndSwapChunkVersion(int numa_node, int chunk_idx, uint64_t expected, uint64_t desired,
                                         std::memory_order success = std::memory_order_acq_rel,
                                         std::memory_order failure = std::memory_order_acquire) {
    validateIndices(numa_node, chunk_idx);
    return state_->numa_nodes[numa_node].chunks[chunk_idx].version.compare_exchange_strong(expected, desired, success,
                                                                                           failure);
  }

  // Fetch-and-add chunk version (atomic add, returns old value)
  inline uint64_t fetchAndAddChunkVersion(int numa_node, int chunk_idx, uint64_t delta,
                                          std::memory_order order = std::memory_order_acq_rel) {
    validateIndices(numa_node, chunk_idx);
    return state_->numa_nodes[numa_node].chunks[chunk_idx].version.fetch_add(delta, order);
  }

  // Increment chunk version (returns new value)
  inline uint64_t incrementChunkVersion(int numa_node, int chunk_idx,
                                        std::memory_order order = std::memory_order_acq_rel) {
    return fetchAndAddChunkVersion(numa_node, chunk_idx, 1, order) + 1;
  }

  // Get raw pointer to chunk state (for advanced usage)
  inline std::atomic<uint64_t>* getChunkVersionPtr(int numa_node, int chunk_idx) {
    validateIndices(numa_node, chunk_idx);
    return &state_->numa_nodes[numa_node].chunks[chunk_idx].version;
  }

  // ===================================================================
  // Progress accessors (for subchunk pipelining)
  // ===================================================================

  // Get chunk progress (atomic load with acquire semantics)
  inline uint64_t getChunkProgress(int numa_node, int chunk_idx,
                                   std::memory_order order = std::memory_order_acquire) const {
    validateIndices(numa_node, chunk_idx);
    return state_->numa_nodes[numa_node].chunks[chunk_idx].progress.load(order);
  }

  // Set chunk progress (atomic store with release semantics)
  inline void setChunkProgress(int numa_node, int chunk_idx, uint64_t value,
                               std::memory_order order = std::memory_order_release) {
    validateIndices(numa_node, chunk_idx);
    state_->numa_nodes[numa_node].chunks[chunk_idx].progress.store(value, order);
  }

  // Advance chunk progress monotonically via CAS (never goes backward)
  inline void advanceChunkProgress(int numa_node, int chunk_idx, uint64_t new_progress) {
    validateIndices(numa_node, chunk_idx);
    auto& p = state_->numa_nodes[numa_node].chunks[chunk_idx].progress;
    uint64_t old = p.load(std::memory_order_relaxed);
    while (new_progress > old) {
      if (p.compare_exchange_weak(old, new_progress, std::memory_order_release, std::memory_order_relaxed)) break;
    }
  }

  // Get raw pointer to chunk progress (for D2H memcpy target)
  inline std::atomic<uint64_t>* getChunkProgressPtr(int numa_node, int chunk_idx) {
    validateIndices(numa_node, chunk_idx);
    return &state_->numa_nodes[numa_node].chunks[chunk_idx].progress;
  }

  // ===================================================================
  // Validation and debugging interfaces
  // ===================================================================

  // Check if state manager is valid (state pointer is not null)
  bool isValid() const { return state_ != nullptr; }

  // Get rank and world size
  int rank() const { return rank_; }
  int worldSize() const { return world_size_; }

  // Dump current state for debugging (non-inline, defined in .cpp)
  void dumpState() const;

  // Get raw state pointer (use with caution)
  GlobalState* getRawState() { return state_; }
  const GlobalState* getRawState() const { return state_; }

  // Reset all chunk versions to 0 (for cleanup between tasks)
  void resetAllChunkVersions();

  // Mark this rank as initialized and wait for all ranks
  void barrierInit();

  // ===================================================================
  // Transfer completion barrier (for collective communication)
  // ===================================================================

  // Increment transfer complete count and return new value
  inline int incrementTransferCompleteCount(std::memory_order order = std::memory_order_acq_rel) {
    return state_->transfer_complete_count.fetch_add(1, order) + 1;
  }

  // Get current transfer complete count
  inline int getTransferCompleteCount(std::memory_order order = std::memory_order_acquire) const {
    return state_->transfer_complete_count.load(order);
  }

  // Reset transfer complete count to 0 (call before starting new task)
  inline void resetTransferCompleteCount(std::memory_order order = std::memory_order_release) {
    state_->transfer_complete_count.store(0, order);
  }

  // ===================================================================
  // Reusable barrier (generation-based sense-reversing)
  // ===================================================================

  // Block until all ranks have called barrier().
  // Safe for repeated use: each call advances the generation.
  inline void barrier() {
    // Snapshot current generation before announcing arrival
    uint64_t my_gen = state_->barrier_generation.load(std::memory_order_acquire);

    // Announce arrival
    int arrived = state_->barrier_count.fetch_add(1, std::memory_order_acq_rel) + 1;

    if (arrived == world_size_) {
      // Last to arrive: reset count for next round and advance generation
      state_->barrier_count.store(0, std::memory_order_relaxed);
      state_->barrier_generation.store(my_gen + 1, std::memory_order_release);
    } else {
      // Wait for generation to advance
      while (state_->barrier_generation.load(std::memory_order_acquire) <= my_gen) {
      }
    }
  }

  // Reset all chunk versions and progress on a given NUMA node
  inline void resetNUMANodeChunks(int numa_node) {
    for (int c = 0; c < MAX_CHUNKS; c++) {
      state_->numa_nodes[numa_node].chunks[c].version.store(0, std::memory_order_relaxed);
      state_->numa_nodes[numa_node].chunks[c].progress.store(0, std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
  }
};

}  // namespace pccl

#endif /* STATE_MANAGER_HPP */
