#include "include/numa_topology.hpp"

#include <numa.h>
#include <numaif.h>
#include <sched.h>

#include <cstring>
#include <mutex>

#include "include/error.hpp"
#include "include/log.hpp"

namespace pccl {

// Static member initialization
bool NumaTopology::initialized_ = false;
bool NumaTopology::numa_available_ = false;
int NumaTopology::num_numa_nodes_ = 1;

// Hardcoded rank-to-NUMA mapping
// TODO: Customize this mapping based on your hardware configuration
const int NumaTopology::RANK_TO_NUMA_NODE[MAX_RANKS] = {
    0, 0,  // Rank 0, 1 → NUMA Node 0
    0, 0,  // Rank 2, 3 → NUMA Node 0
    0, 0,  // Rank 4, 5 → NUMA Node 0
    0, 0   // Rank 6, 7 → NUMA Node 0
};

void NumaTopology::initialize() {
  static std::once_flag s_numa_init_flag;
  std::call_once(s_numa_init_flag, []() {
    // Check if NUMA is available
    if (numa_available() == -1) {
      LOG_WARN("NUMA not available on this system, falling back to single-node mode");
      numa_available_ = false;
      num_numa_nodes_ = 1;
    } else {
      numa_available_ = true;
      num_numa_nodes_ = numa_max_node() + 1;  // numa_max_node() returns max node ID
      LOG_INFO("NUMA initialized: %d NUMA nodes detected", num_numa_nodes_);
    }

    initialized_ = true;
  });
}

bool NumaTopology::isAvailable() {
  if (!initialized_) {
    initialize();
  }
  return numa_available_;
}

int NumaTopology::getNumNUMANodes() {
  if (!initialized_) {
    initialize();
  }
  return num_numa_nodes_;
}

int NumaTopology::getRankNUMANode(int rank) {
  if (rank < 0 || rank >= MAX_RANKS) {
    LOG_ERROR("Invalid rank %d (must be 0-%d)", rank, MAX_RANKS - 1);
    return 0;  // Fallback to node 0
  }

  int numa_node = RANK_TO_NUMA_NODE[rank];

  // Validate that the NUMA node exists
  if (!initialized_) {
    initialize();
  }

  if (numa_node >= num_numa_nodes_) {
    LOG_WARN("Rank %d mapped to NUMA node %d, but only %d nodes available, using node 0", rank, numa_node,
             num_numa_nodes_);
    return 0;
  }

  return numa_node;
}

void NumaTopology::bindMemoryToNode(void* ptr, size_t size, int node) {
  if (!initialized_) {
    initialize();
  }

  if (!numa_available_) {
    LOG_WARN("NUMA not available, skipping memory binding");
    return;
  }

  if (node < 0 || node >= num_numa_nodes_) {
    LOG_ERROR("Invalid NUMA node %d (available: 0-%d)", node, num_numa_nodes_ - 1);
    return;
  }

  // Move memory pages to the specified NUMA node
  // This uses mbind() system call internally
  numa_tonode_memory(ptr, size, node);

  LOG_INFO("Bound memory region [%p, %p) to NUMA node %d", ptr, static_cast<char*>(ptr) + size, node);
}

void NumaTopology::bindCurrentThreadToNode(int node) {
  if (!initialized_) {
    initialize();
  }

  if (!numa_available_) {
    LOG_WARN("NUMA not available, skipping thread binding");
    return;
  }

  if (node < 0 || node >= num_numa_nodes_) {
    LOG_ERROR("Invalid NUMA node %d (available: 0-%d)", node, num_numa_nodes_ - 1);
    return;
  }

  // Run current thread on the specified NUMA node
  // This sets CPU affinity to CPUs on that NUMA node
  numa_run_on_node(node);

  LOG_INFO("Bound current thread to NUMA node %d", node);
}

void NumaTopology::setPreferredNode(int node) {
  if (!initialized_) {
    initialize();
  }

  if (!numa_available_) {
    LOG_WARN("NUMA not available, skipping preferred node setting");
    return;
  }

  if (node < 0 || node >= num_numa_nodes_) {
    LOG_ERROR("Invalid NUMA node %d (available: 0-%d)", node, num_numa_nodes_ - 1);
    return;
  }

  // Set preferred NUMA node for subsequent memory allocations
  numa_set_preferred(node);

  LOG_INFO("Set preferred NUMA node to %d", node);
}

}  // namespace pccl
