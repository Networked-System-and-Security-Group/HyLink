#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "memory/shared_state.hpp"

int main() {
  pccl::SharedStateManager manager(1);
  std::atomic<int> active{0};
  std::atomic<int> max_active{0};
  std::atomic<int> failures{0};
  std::array<std::atomic<size_t>, 2> offsets{};
  std::vector<std::thread> workers;

  int comm_index = 0;
  for (int comm_id : {2, 14}) {
    const int index = comm_index++;
    for (int rank = 0; rank < 2; ++rank) {
      workers.emplace_back([&, comm_id, index, rank] {
        size_t required[pccl::MAX_NUMA_NODES] = {};
        size_t capacity[pccl::MAX_NUMA_NODES] = {};
        size_t region_offsets[pccl::MAX_NUMA_NODES] = {};
        required[0] = 2560ULL * 1024ULL * 1024ULL;
        capacity[0] = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        if (!manager.acquireHostStagingRegion(
                comm_id, 2, rank, required, capacity, region_offsets)) {
          failures.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        if (rank == 0)
          offsets[index].store(region_offsets[0], std::memory_order_release);
        const int now = active.fetch_add(1, std::memory_order_acq_rel) + 1;
        int observed = max_active.load(std::memory_order_relaxed);
        while (now > observed &&
               !max_active.compare_exchange_weak(observed, now, std::memory_order_relaxed)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        active.fetch_sub(1, std::memory_order_acq_rel);
        if (!manager.releaseHostStagingRegion(comm_id, 2, rank))
          failures.fetch_add(1, std::memory_order_relaxed);
      });
    }
  }

  for (auto& worker : workers)
    worker.join();

  if (failures.load(std::memory_order_relaxed) != 0)
    return 1;
  if (max_active.load(std::memory_order_relaxed) != 4)
    return 2;
  const size_t first = offsets[0].load(std::memory_order_acquire);
  const size_t second = offsets[1].load(std::memory_order_acquire);
  const size_t bytes = 2560ULL * 1024ULL * 1024ULL;
  if (!(first + bytes <= second || second + bytes <= first))
    return 3;
  return 0;
}
