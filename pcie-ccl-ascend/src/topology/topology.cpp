#include "topology.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../runtime/ascend_runtime.hpp"
#include "../util/log.hpp"

namespace pccl {

// Query NUMA node for a device via its PCI Bus ID.
// Throws on failure — no silent fallback.
static int queryNumaForDevice(int device) {
  char bus_id[32] = {};
  if (devGetPciBusId(device, bus_id, sizeof(bus_id)) != devSuccess || bus_id[0] == '\0') {
    throw std::runtime_error("NPU " + std::to_string(device) + ": failed to get PCI Bus ID (is the device visible?)");
  }

  // sysfs path: /sys/bus/pci/devices/<bus_id>/numa_node
  std::string sysfs_path = std::string("/sys/bus/pci/devices/") + bus_id + "/numa_node";
  std::ifstream f(sysfs_path);
  if (!f.is_open()) {
    throw std::runtime_error("NPU " + std::to_string(device) + " (PCI " + bus_id + "): sysfs numa_node not found at " +
                             sysfs_path);
  }
  int numa_node;
  f >> numa_node;
  if (numa_node < 0) {
    throw std::runtime_error("NPU " + std::to_string(device) + " (PCI " + bus_id +
                             "): sysfs reports invalid NUMA node " + std::to_string(numa_node));
  }

  LOG_INFO("NPU %d (PCI %s) -> physical NUMA %d (sysfs)", device, bus_id, numa_node);
  return numa_node;
}

static std::string queryPciBusIdForDevice(int device) {
  char bus_id[32] = {};
  if (devGetPciBusId(device, bus_id, sizeof(bus_id)) != devSuccess || bus_id[0] == '\0') {
    throw std::runtime_error("NPU " + std::to_string(device) + ": failed to get PCI Bus ID for topology");
  }
  return std::string(bus_id);
}

static void populatePairwiseTopology(LocalTopology& topo) {
  topo.device_pci_bus_ids.resize(topo.num_devices);
  for (int device = 0; device < topo.num_devices; ++device) {
    topo.device_pci_bus_ids[device] = queryPciBusIdForDevice(device);
  }
}

// Build logical NUMA mapping from a vector of physical NUMA IDs per device.
// Returns a LocalTopology with compact logical NUMA IDs.
static LocalTopology buildLogicalMappingFromPhysical(int num_devices, const std::vector<int>& physical_numas) {
  LocalTopology topo;
  topo.num_devices = num_devices;

  // Collect unique physical NUMA nodes in order of first appearance
  std::vector<int> seen;
  for (int phys : physical_numas) {
    bool found = false;
    for (int s : seen) {
      if (s == phys) {
        found = true;
        break;
      }
    }
    if (!found)
      seen.push_back(phys);
  }
  topo.numa_to_physical = seen;
  topo.num_numa_nodes = static_cast<int>(seen.size());

  // Build reverse mapping: physical -> logical.
  topo.device_to_numa.resize(num_devices);
  for (int device = 0; device < num_devices; device++) {
    int phys = physical_numas[device];
    for (int l = 0; l < topo.num_numa_nodes; l++) {
      if (topo.numa_to_physical[l] == phys) {
        topo.device_to_numa[device] = l;
        break;
      }
    }
  }

  populatePairwiseTopology(topo);
  return topo;
}

static LocalTopology buildTopologyFromLogicalMap(int num_devices, const std::vector<int>& logical_numas) {
  LocalTopology topo;
  topo.num_devices = num_devices;
  topo.device_to_numa = logical_numas;
  topo.num_numa_nodes = 0;
  for (int logical : logical_numas) {
    if (logical < 0)
      throw std::runtime_error("PCCL_NPU_NUMA_MAP contains a negative NUMA id");
    topo.num_numa_nodes = std::max(topo.num_numa_nodes, logical + 1);
  }
  topo.numa_to_physical.assign(topo.num_numa_nodes, -1);
  for (int device = 0; device < num_devices; ++device) {
    int logical = logical_numas[device];
    int physical = queryNumaForDevice(device);
    if (topo.numa_to_physical[logical] < 0) {
      topo.numa_to_physical[logical] = physical;
    } else if (topo.numa_to_physical[logical] != physical) {
      LOG_WARN("Logical NUMA %d includes devices on physical NUMA %d and %d; using first physical node",
               logical, topo.numa_to_physical[logical], physical);
    }
  }
  for (int logical = 0; logical < topo.num_numa_nodes; ++logical) {
    if (topo.numa_to_physical[logical] < 0)
      topo.numa_to_physical[logical] = logical;
  }
  populatePairwiseTopology(topo);
  return topo;
}

LocalTopology discoverLocalTopology(int num_devices) {
  // Check for manual override first
  const char* map_env = getenv("PCCL_NPU_NUMA_MAP");
  if (!map_env)
    map_env = getenv("PCCL_GPU_NUMA_MAP");
  if (map_env) {
    LOG_INFO("Using device NUMA map: %s", map_env);
    return parseDeviceNumaMap(map_env, num_devices);
  }

  // Collect physical NUMA for each NPU (throws on sysfs failure).
  std::vector<int> physical_numas(num_devices);
  for (int device = 0; device < num_devices; device++) {
    physical_numas[device] = queryNumaForDevice(device);
  }

  LocalTopology topo = buildLogicalMappingFromPhysical(num_devices, physical_numas);
  LOG_INFO("Discovered topology: %d devices, %d NUMA nodes", topo.num_devices, topo.num_numa_nodes);
  return topo;
}

LocalTopology parseDeviceNumaMap(const std::string& map_str, int num_devices) {
  std::vector<int> logical_numas(num_devices, 0);

  std::istringstream ss(map_str);
  std::string token;
  int idx = 0;

  while (std::getline(ss, token, ',') && idx < num_devices) {
    logical_numas[idx] = std::stoi(token);
    idx++;
  }

  return buildTopologyFromLogicalMap(num_devices, logical_numas);
}

}  // namespace pccl
