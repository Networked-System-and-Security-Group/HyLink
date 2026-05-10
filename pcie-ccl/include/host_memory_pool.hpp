#ifndef HOST_MEMORY_POOL_HPP
#define HOST_MEMORY_POOL_HPP

#include <atomic>
#include <cstddef>
#include <map>

#include "error.hpp"
#include "ir.hpp"

namespace pccl {

// HostMemoryPool: Manages per-NUMA-node chunk pools for collective operations
// Each NUMA node has a shared chunk pool, accessible by all ranks on that node
// This eliminates the need for inter-rank buffer access within the same NUMA node
class HostMemoryPool {
 private:
  struct PoolHeader {
    size_t chunk_size;                         // Size of each chunk
    int num_chunks;                            // Number of chunks (= MAX_CHUNKS)
    std::atomic<bool> initialized[MAX_RANKS];  // Track which ranks initialized
  };

  // Per-NUMA-node buffers
  std::map<int, void*> numa_buffers_;  // numa_node -> buffer base ptr
  std::map<int, int> numa_shm_fds_;    // numa_node -> shm fd (for cleanup)
  std::map<int, bool> numa_pinned_;    // numa_node -> is_pinned

  int my_rank_;
  int my_numa_node_;  // This rank's NUMA node
  int world_size_;
  size_t chunk_size_;  // Size of each chunk
  int num_chunks_;     // Number of chunks (= MAX_CHUNKS)
  bool is_owner_;      // true if first rank in this NUMA node

 public:
  // Constructor: Sets up per-NUMA-node chunk pools
  // chunk_size: Size of each chunk in the pool
  // num_chunks: Number of chunks per NUMA node (default: MAX_CHUNKS)
  HostMemoryPool(int rank, int world_size, size_t chunk_size, int num_chunks = MAX_CHUNKS);

  // Destructor: Cleans up shared memory
  ~HostMemoryPool();

  // Delete copy and move constructors/operators (non-copyable)
  HostMemoryPool(const HostMemoryPool&) = delete;
  HostMemoryPool& operator=(const HostMemoryPool&) = delete;
  HostMemoryPool(HostMemoryPool&&) = delete;
  HostMemoryPool& operator=(HostMemoryPool&&) = delete;

  // ===================================================================
  // Core Access Interfaces
  // ===================================================================

  // Get buffer for a specific chunk on a specific NUMA node
  // This is the lowest-level interface
  void* getChunkBuffer(int numa_node, int chunk_idx);

  // Get buffer for a specific chunk on this rank's NUMA node
  // Convenient for local operations
  void* getLocalChunk(int chunk_idx);

  // Get buffer for a specific chunk on the NUMA node where a rank resides
  // Convenient for peer operations
  void* getRankChunk(int rank, int chunk_idx);

  // Get the base pointer for a NUMA node's entire pool
  // Advanced usage: caller manages chunk offsets
  void* getNUMABuffer(int numa_node);

  // Get this rank's NUMA node buffer base
  void* getMyNUMABuffer();

  // ===================================================================
  // Query Interfaces
  // ===================================================================

  // Get chunk size
  size_t getChunkSize() const { return chunk_size_; }

  // Get number of chunks per NUMA node
  int getNumChunks() const { return num_chunks_; }

  // Get this rank's NUMA node
  int getMyNUMANode() const { return my_numa_node_; }

  // Get the NUMA node for a specific rank
  int getRankNUMANode(int rank) const;

  // Check if pool is valid
  bool isValid() const { return !numa_buffers_.empty(); }

  // Get rank information
  int rank() const { return my_rank_; }
  int worldSize() const { return world_size_; }

 private:
  // Helper: Compute 64-byte aligned offset for chunk data after PoolHeader
  // Required because SIMD (AVX-512) operations need 64-byte aligned addresses
  static size_t alignedDataOffset() {
    return (sizeof(PoolHeader) + 63) & ~static_cast<size_t>(63);
  }

  // Helper: Check if this rank is the first rank in its NUMA node
  // (responsible for creating the shared memory)
  bool isFirstRankInNUMA() const;

  // Helper: Get the number of unique NUMA nodes used by all ranks
  int getNumUniqueNUMANodes() const;
};

}  // namespace pccl

#endif  // HOST_MEMORY_POOL_HPP
