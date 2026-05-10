#ifndef PCCL_TOPOLOGY_HPP
#define PCCL_TOPOLOGY_HPP

#include <string>
#include <vector>

namespace pccl {

struct LocalTopology {
  int num_gpus;
  int num_numa_nodes;  // de-duplicated count (compact, 0-based)

  std::vector<int> gpu_to_numa;       // gpu_id → logical NUMA (0 .. num_numa_nodes-1)
  std::vector<int> numa_to_physical;  // logical NUMA → physical NUMA (for OS calls)
  std::vector<std::string> gpu_pci_bus_ids;
  std::vector<std::vector<bool>> has_nvlink;
  std::vector<std::vector<bool>> lca_is_pcie_switch;
  std::vector<std::vector<bool>> direct_peer_allowed;

  int getNUMAForGPU(int gpu) const {
    if (gpu < 0 || gpu >= static_cast<int>(gpu_to_numa.size()))
      return 0;
    return gpu_to_numa[gpu];
  }

  int getPhysicalNUMA(int logical) const {
    if (logical < 0 || logical >= static_cast<int>(numa_to_physical.size()))
      return 0;
    return numa_to_physical[logical];
  }

  int getNumNUMANodes() const { return num_numa_nodes; }

  bool pairHasNVLink(int a, int b) const {
    if (a < 0 || b < 0 || a >= static_cast<int>(has_nvlink.size()) ||
        b >= static_cast<int>(has_nvlink[a].size()))
      return false;
    return has_nvlink[a][b];
  }

  bool pairLcaIsPcieSwitch(int a, int b) const {
    if (a < 0 || b < 0 || a >= static_cast<int>(lca_is_pcie_switch.size()) ||
        b >= static_cast<int>(lca_is_pcie_switch[a].size()))
      return false;
    return lca_is_pcie_switch[a][b];
  }

  bool pairDirectPeerAllowed(int a, int b) const {
    if (a == b)
      return true;
    if (a < 0 || b < 0 || a >= static_cast<int>(direct_peer_allowed.size()) ||
        b >= static_cast<int>(direct_peer_allowed[a].size()))
      return false;
    return direct_peer_allowed[a][b];
  }
};

// Auto-discover local topology from sysfs
LocalTopology discoverLocalTopology(int num_gpus);

// Parse PCCL_GPU_NUMA_MAP environment variable or config string
// Format: "0,0,1,1" (comma-separated NUMA node per GPU)
LocalTopology parseGpuNumaMap(const std::string& map_str, int num_gpus);

}  // namespace pccl

#endif /* PCCL_TOPOLOGY_HPP */
