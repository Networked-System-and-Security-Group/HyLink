#ifndef STATES_HPP
#define STATES_HPP
#include <atomic>

#include "ir.hpp"

namespace pccl {

// Maximum number of NUMA nodes supported
#define MAX_NUMA_NODES 8

struct alignas(64) ChunkState {
  // V0: Empty
  // V1: Data Copied from Somewhere
  // Vn: Data Reduced n-1 times
  std::atomic<uint64_t> version;
  // Subchunk progress: number of contiguous subchunks completed at final version
  std::atomic<uint64_t> progress;
};

// State for all chunks in a single NUMA node
// Each NUMA node has a shared chunk pool, so we track chunk versions per NUMA node
struct alignas(64) NumaNodeState {
  ChunkState chunks[MAX_CHUNKS];
};

// Global state tracking chunk versions for all NUMA nodes
// This represents the physical Host Relay Buffers (one per NUMA node)
struct alignas(64) GlobalState {
  // Initialization barrier - number of ranks that have initialized
  std::atomic<int> initialized_count;

  // Transfer completion counter - incremented by each rank when H2D complete
  // When this reaches world_size, all ranks have finished their transfers
  std::atomic<int> transfer_complete_count;

  // Reusable barrier for post-completion cleanup synchronization
  // Uses generation-based sense-reversing: barrier_count tracks arrivals,
  // barrier_generation advances when all ranks arrive.
  std::atomic<int> barrier_count;
  std::atomic<uint64_t> barrier_generation;

  NumaNodeState numa_nodes[MAX_NUMA_NODES];
};

}  // namespace pccl

#endif /* STATES_HPP */
