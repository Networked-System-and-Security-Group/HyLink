#include <cassert>

#include "topology/comm_topology.hpp"

int main() {
  pccl::LocalTopology global;
  global.num_devices = 8;
  global.num_numa_nodes = 4;
  global.device_to_numa = {0, 0, 1, 1, 2, 2, 3, 3};
  global.numa_to_physical = {10, 11, 12, 13};

  const auto child = pccl::projectCommunicatorTopology(global, 4, 4, 0);
  assert(child.num_devices == 4);
  assert(child.num_numa_nodes == 2);
  assert((child.device_to_numa == std::vector<int>{0, 0, 1, 1}));
  assert((child.numa_to_physical == std::vector<int>{12, 13}));

  const auto world = pccl::projectCommunicatorTopology(global, 8, 0, 0);
  assert(world.num_devices == 8);
  assert(world.device_to_numa == global.device_to_numa);
  assert(world.numa_to_physical == global.numa_to_physical);

  return 0;
}
