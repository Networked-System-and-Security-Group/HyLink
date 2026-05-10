#ifndef PCCL_NUMA_HPP
#define PCCL_NUMA_HPP

#include <cstddef>

namespace pccl {

// Pin current thread to a NUMA node (-1 to unpin)
void pinThreadToNuma(int numa_node);

// Pin current thread to a specific CPU core
void pinThreadToCpu(int cpu);

// Get the same-die sibling NUMA node (nearest neighbor by distance).
// Returns numa_node itself if no sibling exists (single-die or no libnuma).
int getNumaSibling(int numa_node);

// Get list of CPU cores belonging to a NUMA node.
// Returns count of CPUs found. Fills cpus[] up to max_cpus entries.
int getNumaCpus(int numa_node, int* cpus, int max_cpus);

}  // namespace pccl

#endif /* PCCL_NUMA_HPP */
