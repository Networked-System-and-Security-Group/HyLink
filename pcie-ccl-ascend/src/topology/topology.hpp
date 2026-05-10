#ifndef PCCL_TOPOLOGY_HPP
#define PCCL_TOPOLOGY_HPP

#include <string>
#include <vector>

namespace pccl {

struct LocalTopology {
  int num_devices;
  int num_numa_nodes;  // de-duplicated count (compact, 0-based)

  std::vector<int> device_to_numa;       // device id -> logical NUMA (0 .. num_numa_nodes-1)
  std::vector<int> numa_to_physical;     // logical NUMA -> physical NUMA (for OS calls)
  std::vector<std::string> device_pci_bus_ids;

  int getNUMAForDevice(int device) const {
    if (device < 0 || device >= static_cast<int>(device_to_numa.size()))
      return 0;
    return device_to_numa[device];
  }

  int getPhysicalNUMA(int logical) const {
    if (logical < 0 || logical >= static_cast<int>(numa_to_physical.size()))
      return 0;
    return numa_to_physical[logical];
  }

  int getNumNUMANodes() const { return num_numa_nodes; }
};

// Auto-discover local topology from sysfs
LocalTopology discoverLocalTopology(int num_devices);

// Parse a device-to-NUMA map string, e.g. "0,0,1,1".
LocalTopology parseDeviceNumaMap(const std::string& map_str, int num_devices);

}  // namespace pccl

#endif /* PCCL_TOPOLOGY_HPP */
