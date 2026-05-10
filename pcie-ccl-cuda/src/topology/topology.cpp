#include "topology.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../hal/device_rt.hpp"
#include "../util/log.hpp"

namespace pccl {

namespace {

#if defined(PCCL_DEVICE_CUDA)
using nvmlReturn_t = int;
using nvmlDevice_t = void*;

constexpr nvmlReturn_t kNvmlSuccess = 0;
constexpr int kNvmlTopologySingle = 10;  // NVML_TOPOLOGY_SINGLE: traverse a single PCIe switch.

struct NvmlApi {
  void* handle = nullptr;
  nvmlReturn_t (*init_v2)() = nullptr;
  nvmlReturn_t (*shutdown)() = nullptr;
  nvmlReturn_t (*device_get_handle_by_pci_bus_id_v2)(const char*, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*device_get_topology_common_ancestor)(nvmlDevice_t, nvmlDevice_t, int*) = nullptr;
  bool ready = false;

  NvmlApi() {
    handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      return;

    init_v2 = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(handle, "nvmlInit_v2"));
    shutdown = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(handle, "nvmlShutdown"));
    device_get_handle_by_pci_bus_id_v2 =
        reinterpret_cast<nvmlReturn_t (*)(const char*, nvmlDevice_t*)>(
            dlsym(handle, "nvmlDeviceGetHandleByPciBusId_v2"));
    device_get_topology_common_ancestor =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, int*)>(
            dlsym(handle, "nvmlDeviceGetTopologyCommonAncestor"));

    if (!init_v2 || !shutdown || !device_get_handle_by_pci_bus_id_v2 ||
        !device_get_topology_common_ancestor) {
      dlclose(handle);
      handle = nullptr;
      return;
    }

    ready = (init_v2() == kNvmlSuccess);
  }

  ~NvmlApi() {
    if (ready && shutdown)
      shutdown();
    if (handle)
      dlclose(handle);
  }

  bool pairIsPix(const std::string& a_bus_id, const std::string& b_bus_id) const {
    if (!ready)
      return false;

    nvmlDevice_t a = nullptr;
    nvmlDevice_t b = nullptr;
    if (device_get_handle_by_pci_bus_id_v2(a_bus_id.c_str(), &a) != kNvmlSuccess ||
        device_get_handle_by_pci_bus_id_v2(b_bus_id.c_str(), &b) != kNvmlSuccess) {
      return false;
    }

    int ancestor = -1;
    if (device_get_topology_common_ancestor(a, b, &ancestor) != kNvmlSuccess)
      return false;
    return ancestor == kNvmlTopologySingle;
  }
};

#endif

}  // namespace

// Query NUMA node for a device via its PCI Bus ID.
// Uses HAL to get the PCI address, then reads sysfs.
// Throws on failure — no silent fallback.
static int queryNumaForDevice(int device) {
  char bus_id[32] = {};
  if (devGetPciBusId(device, bus_id, sizeof(bus_id)) != devSuccess || bus_id[0] == '\0') {
    throw std::runtime_error("GPU " + std::to_string(device) + ": failed to get PCI Bus ID (is the device visible?)");
  }

  // sysfs path: /sys/bus/pci/devices/<bus_id>/numa_node
  std::string sysfs_path = std::string("/sys/bus/pci/devices/") + bus_id + "/numa_node";
  std::ifstream f(sysfs_path);
  if (!f.is_open()) {
    throw std::runtime_error("GPU " + std::to_string(device) + " (PCI " + bus_id + "): sysfs numa_node not found at " +
                             sysfs_path);
  }
  int numa_node;
  f >> numa_node;
  if (numa_node < 0) {
    throw std::runtime_error("GPU " + std::to_string(device) + " (PCI " + bus_id +
                             "): sysfs reports invalid NUMA node " + std::to_string(numa_node));
  }

  LOG_INFO("GPU %d (PCI %s) -> physical NUMA %d (sysfs)", device, bus_id, numa_node);
  return numa_node;
}

static std::string queryPciBusIdForDevice(int device) {
  char bus_id[32] = {};
  if (devGetPciBusId(device, bus_id, sizeof(bus_id)) != devSuccess || bus_id[0] == '\0') {
    throw std::runtime_error("GPU " + std::to_string(device) + ": failed to get PCI Bus ID for topology");
  }
  return std::string(bus_id);
}

static void populatePairwiseTopology(LocalTopology& topo) {
  topo.gpu_pci_bus_ids.resize(topo.num_gpus);
  topo.has_nvlink.assign(topo.num_gpus, std::vector<bool>(topo.num_gpus, false));
  topo.lca_is_pcie_switch.assign(topo.num_gpus, std::vector<bool>(topo.num_gpus, false));
  topo.direct_peer_allowed.assign(topo.num_gpus, std::vector<bool>(topo.num_gpus, false));

  for (int g = 0; g < topo.num_gpus; ++g) {
    topo.gpu_pci_bus_ids[g] = queryPciBusIdForDevice(g);
  }

  const char* force_switch = std::getenv("PCCL_FORCE_PCIE_SWITCH");
  bool forced_switch = force_switch && std::atoi(force_switch) != 0;
  const char* force_nvlink = std::getenv("PCCL_FORCE_NVLINK");
  bool forced_nvlink = force_nvlink && std::atoi(force_nvlink) != 0;

#if defined(PCCL_DEVICE_CUDA)
  NvmlApi nvml;
  if (!nvml.ready && !forced_switch) {
    LOG_WARN("NVML topology query unavailable; peer D2D requires explicit PCCL_FORCE_PCIE_SWITCH=1");
  }
#endif

  for (int i = 0; i < topo.num_gpus; ++i) {
    topo.direct_peer_allowed[i][i] = true;
    for (int j = 0; j < topo.num_gpus; ++j) {
      if (i == j)
        continue;

      // NVML detection is intentionally not linked in this first pass. Tests can
      // force relay behavior with PCCL_FORCE_NVLINK=1.
      topo.has_nvlink[i][j] = forced_nvlink;

      bool is_pix = forced_switch;
#if defined(PCCL_DEVICE_CUDA)
      if (!is_pix)
        is_pix = nvml.pairIsPix(topo.gpu_pci_bus_ids[i], topo.gpu_pci_bus_ids[j]);
#endif
      topo.lca_is_pcie_switch[i][j] = is_pix;

      int can = 0;
      devStatus status = devCanAccessPeer(i, j, &can);
      topo.direct_peer_allowed[i][j] = is_pix && (status == devSuccess && can != 0);
      LOG_DEBUG("Topology pair %d-%d: pix=%d peer_access=%d", i, j, is_pix ? 1 : 0,
                topo.direct_peer_allowed[i][j] ? 1 : 0);
    }
  }
}

// Build logical NUMA mapping from a vector of physical NUMA IDs per GPU.
// Returns a LocalTopology with compact logical NUMA IDs.
static LocalTopology buildLogicalMapping(int num_gpus, const std::vector<int>& physical_numas) {
  LocalTopology topo;
  topo.num_gpus = num_gpus;

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

  // Build reverse mapping: physical → logical
  // (only needed temporarily for gpu_to_numa construction)
  topo.gpu_to_numa.resize(num_gpus);
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    int phys = physical_numas[gpu];
    for (int l = 0; l < topo.num_numa_nodes; l++) {
      if (topo.numa_to_physical[l] == phys) {
        topo.gpu_to_numa[gpu] = l;
        break;
      }
    }
  }

  populatePairwiseTopology(topo);
  return topo;
}

LocalTopology discoverLocalTopology(int num_gpus) {
  // Check for manual override first
  const char* map_env = getenv("PCCL_GPU_NUMA_MAP");
  if (map_env) {
    LOG_INFO("Using PCCL_GPU_NUMA_MAP: %s", map_env);
    return parseGpuNumaMap(map_env, num_gpus);
  }

  // Collect physical NUMA for each GPU (throws on sysfs failure)
  std::vector<int> physical_numas(num_gpus);
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    physical_numas[gpu] = queryNumaForDevice(gpu);
  }

  LocalTopology topo = buildLogicalMapping(num_gpus, physical_numas);
  LOG_INFO("Discovered topology: %d GPUs, %d NUMA nodes", topo.num_gpus, topo.num_numa_nodes);
  return topo;
}

LocalTopology parseGpuNumaMap(const std::string& map_str, int num_gpus) {
  std::vector<int> physical_numas(num_gpus, 0);

  std::istringstream ss(map_str);
  std::string token;
  int idx = 0;

  while (std::getline(ss, token, ',') && idx < num_gpus) {
    physical_numas[idx] = std::stoi(token);
    idx++;
  }

  return buildLogicalMapping(num_gpus, physical_numas);
}

}  // namespace pccl
