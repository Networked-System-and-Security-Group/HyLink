#include "numa.hpp"

#include <cerrno>
#include <cstring>
#include <mutex>
#include <sched.h>

#include "../util/log.hpp"

// Conditionally include numa.h
#if __has_include(<numa.h>)
#include <numa.h>
#define PCCL_HAS_NUMA 1
#else
#define PCCL_HAS_NUMA 0
#endif

namespace pccl {

static std::once_flag s_numa_init_flag;
static bool s_numa_available = false;
static int s_num_numa_nodes = 1;

static void numaInit() {
  std::call_once(s_numa_init_flag, []() {
#if PCCL_HAS_NUMA
    if (numa_available() != -1) {
      s_numa_available = true;
      s_num_numa_nodes = numa_max_node() + 1;
      LOG_INFO("NUMA initialized: %d nodes detected", s_num_numa_nodes);
    } else {
      LOG_WARN("NUMA not available on this system");
    }
#else
    LOG_WARN("NUMA support not compiled in (libnuma not found)");
#endif
  });
}

void pinThreadToNuma(int numa_node) {
  numaInit();
#if PCCL_HAS_NUMA
  if (!s_numa_available)
    return;
  if (numa_node < 0) {
    // Unpin: allow thread to run on all nodes
    if (numa_run_on_node(-1) != 0) {
      LOG_WARN("Failed to unpin thread from NUMA affinity: %s", strerror(errno));
    } else {
      LOG_DEBUG("Unpinned thread from NUMA affinity");
    }
  } else if (numa_node < s_num_numa_nodes) {
    if (numa_run_on_node(numa_node) != 0) {
      LOG_WARN("Failed to pin thread to NUMA node %d: %s", numa_node, strerror(errno));
    } else {
      LOG_DEBUG("Pinned thread to NUMA node %d", numa_node);
    }
  } else {
    LOG_WARN("Invalid NUMA node %d (max %d)", numa_node, s_num_numa_nodes - 1);
  }
#else
  (void)numa_node;
#endif
}

void pinThreadToCpu(int cpu) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
    LOG_WARN("Failed to pin thread to CPU %d: %s", cpu, strerror(errno));
  } else {
    LOG_DEBUG("Pinned thread to CPU %d", cpu);
  }
}

int getNumaSibling(int numa_node) {
  numaInit();
#if PCCL_HAS_NUMA
  if (!s_numa_available || numa_node < 0 || numa_node >= s_num_numa_nodes)
    return numa_node;

  // Find nearest neighbor (smallest distance > 10, i.e. not self)
  int best = numa_node;
  int best_dist = INT32_MAX;
  for (int i = 0; i < s_num_numa_nodes; i++) {
    if (i == numa_node) continue;
    int d = numa_distance(numa_node, i);
    if (d > 0 && d < best_dist) {
      best_dist = d;
      best = i;
    }
  }
  // Only use sibling if it's meaningfully closer than remote nodes
  // (same-die typically has distance 11, cross-die 24+)
  if (best_dist > 15) {
    LOG_DEBUG("NUMA %d has no close sibling (nearest=%d dist=%d), staying local",
              numa_node, best, best_dist);
    return numa_node;
  }
  LOG_DEBUG("NUMA %d sibling is %d (distance=%d)", numa_node, best, best_dist);
  return best;
#else
  return numa_node;
#endif
}

int getNumaCpus(int numa_node, int* cpus, int max_cpus) {
  numaInit();
  int count = 0;
#if PCCL_HAS_NUMA
  if (!s_numa_available || numa_node < 0 || numa_node >= s_num_numa_nodes)
    return 0;
  struct bitmask* bm = numa_allocate_cpumask();
  numa_node_to_cpus(numa_node, bm);
  for (int i = 0; i < numa_num_possible_cpus() && count < max_cpus; i++) {
    if (numa_bitmask_isbitset(bm, i))
      cpus[count++] = i;
  }
  numa_free_cpumask(bm);
#else
  (void)numa_node;
  (void)cpus;
  (void)max_cpus;
#endif
  return count;
}

}  // namespace pccl
