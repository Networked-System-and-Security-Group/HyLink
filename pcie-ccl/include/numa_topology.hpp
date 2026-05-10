#ifndef NUMA_TOPOLOGY_HPP
#define NUMA_TOPOLOGY_HPP

#include <cstddef>

#include "ir.hpp"

namespace pccl {

// NumaTopology: Manages NUMA topology detection and rank-to-NUMA mapping
// Provides utilities for NUMA-aware memory allocation and thread binding
class NumaTopology {
 public:
  // Initialize NUMA subsystem (must be called before other methods)
  static void initialize();

  // Check if NUMA is available and initialized
  static bool isAvailable();

  // Get the number of NUMA nodes in the system
  static int getNumNUMANodes();

  // Get the NUMA node for a given rank (uses hardcoded mapping)
  static int getRankNUMANode(int rank);

  // Bind memory region to a specific NUMA node
  // ptr: pointer to memory region (must be mmap'd memory)
  // size: size of memory region in bytes
  // node: target NUMA node ID
  static void bindMemoryToNode(void* ptr, size_t size, int node);

  // Bind current thread to run on a specific NUMA node
  // node: target NUMA node ID
  static void bindCurrentThreadToNode(int node);

  // Set preferred NUMA node for subsequent memory allocations
  // node: target NUMA node ID
  static void setPreferredNode(int node);

 private:
  // Hardcoded mapping from rank to NUMA node
  // This array defines which NUMA node each rank should use
  // Example: {0, 0, 1, 1, 2, 2, 3, 3} means:
  //   - Ranks 0, 1 use NUMA node 0
  //   - Ranks 2, 3 use NUMA node 1
  //   - Ranks 4, 5 use NUMA node 2
  //   - Ranks 6, 7 use NUMA node 3
  static const int RANK_TO_NUMA_NODE[MAX_RANKS];

  static bool initialized_;
  static bool numa_available_;
  static int num_numa_nodes_;
};

}  // namespace pccl

#endif /* NUMA_TOPOLOGY_HPP */
