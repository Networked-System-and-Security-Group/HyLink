#ifndef PCCL_COMM_TOPOLOGY_HPP
#define PCCL_COMM_TOPOLOGY_HPP

#include <algorithm>
#include <vector>

#include "topology.hpp"

namespace pccl {

// Build the compact topology visible to a communicator.  PCCL keeps one
// process-wide topology for all devices, while a subgroup IR is described in
// local rank/NUMA coordinates.  For the contiguous subgroups created by the
// current HCCL adapter, device_id - local_rank identifies the first device in
// the communicator.
inline LocalTopology projectCommunicatorTopology(const LocalTopology& global,
                                                 int communicator_size,
                                                 int local_device_id,
                                                 int local_rank) {
  if (communicator_size <= 0 || communicator_size >= global.num_devices)
    return global;

  int first_device = local_device_id - local_rank;
  if (first_device < 0 || first_device + communicator_size > global.num_devices)
    first_device = 0;

  LocalTopology projected;
  projected.num_devices = communicator_size;
  std::vector<int> global_numas;
  projected.device_to_numa.reserve(communicator_size);

  for (int local_device = 0; local_device < communicator_size; ++local_device) {
    const int global_device = first_device + local_device;
    const int global_numa = global.getNUMAForDevice(global_device);
    auto it = std::find(global_numas.begin(), global_numas.end(), global_numa);
    if (it == global_numas.end()) {
      global_numas.push_back(global_numa);
      it = global_numas.end() - 1;
    }
    projected.device_to_numa.push_back(static_cast<int>(it - global_numas.begin()));
  }

  projected.num_numa_nodes = static_cast<int>(global_numas.size());
  projected.numa_to_physical.reserve(global_numas.size());
  for (int global_numa : global_numas)
    projected.numa_to_physical.push_back(global.getPhysicalNUMA(global_numa));

  if (static_cast<int>(global.device_pci_bus_ids.size()) >= first_device + communicator_size) {
    projected.device_pci_bus_ids.assign(global.device_pci_bus_ids.begin() + first_device,
                                        global.device_pci_bus_ids.begin() + first_device + communicator_size);
  }
  return projected;
}

}  // namespace pccl

#endif /* PCCL_COMM_TOPOLOGY_HPP */
