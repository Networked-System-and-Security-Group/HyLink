#include "host_pool.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include "../runtime/ascend_runtime.hpp"
#include "../topology/numa.hpp"
#include "../topology/topology.hpp"
#include "../util/log.hpp"
#include "shm_allocator.hpp"

namespace pccl {

// ---- Single-process constructor (unchanged behavior) ----

HostMemoryPool::HostMemoryPool(int nranks, size_t buffer_size, const LocalTopology* topo)
    : buffer_size_(buffer_size), nranks_(nranks), topo_(topo) {
  int num_numa = (topo && topo->num_numa_nodes > 0) ? topo->num_numa_nodes : 1;

  size_t pool_size = alignedDataOffset() + buffer_size;

  for (int numa = 0; numa < num_numa; numa++) {
    // Pin current thread to the physical NUMA node so that devMallocHost
    // allocates physical pages on it (Linux first-touch policy).
    int physical = topo ? topo->getPhysicalNUMA(numa) : 0;
    pinThreadToNuma(physical);

    void* ptr = nullptr;
    devStatus status = devMallocHost(&ptr, pool_size);
    if (status != devSuccess || !ptr) {
      LOG_ERROR("HostMemoryPool: devMallocHost failed for logical NUMA %d (physical %d), size=%zu, status=%d", numa,
                physical, pool_size, static_cast<int>(status));
      continue;
    }

    numa_buffers_[numa] = ptr;
    numa_sizes_[numa] = pool_size;

    LOG_INFO("HostMemoryPool: Allocated %zu bytes (pinned) for logical NUMA %d (physical %d) at %p", pool_size, numa,
             physical, ptr);
  }

  // Restore: unpin thread from any specific NUMA node
  pinThreadToNuma(-1);
}

// ---- Multi-process constructor ----
//
// Multi-process host staging uses memfd_create(MFD_HUGETLB) + mbind +
// devHostRegister.  The BootstrapShm region shares the creator's memfd numbers
// with joiners.

HostMemoryPool::HostMemoryPool(int nranks, size_t buffer_size, const LocalTopology* topo,
                               const std::string& unique_id, bool is_creator)
    : HostMemoryPool(nranks, buffer_size, topo, unique_id, is_creator, std::string()) {}

HostMemoryPool::HostMemoryPool(int nranks, size_t buffer_size, const LocalTopology* topo,
                               const std::string& unique_id, bool is_creator,
                               const std::string& namespace_suffix)
    : buffer_size_(buffer_size),
      nranks_(nranks),
      topo_(topo),
      multi_process_(true),
      is_creator_(is_creator),
      unique_id_(unique_id + namespace_suffix) {
  int num_numa = (topo && topo->num_numa_nodes > 0) ? topo->num_numa_nodes : 1;

  // Open or create bootstrap shm for fd exchange (shared by both platforms)
  BootstrapShm* bs = nullptr;
  if (is_creator) {
    bs = ShmAllocator::createBootstrap(unique_id_, nranks);
    if (!bs) {
      LOG_ERROR("HostMemoryPool: Failed to create bootstrap shm");
      return;
    }
  } else {
    bs = ShmAllocator::openBootstrap(unique_id_);
    if (!bs) {
      LOG_ERROR("HostMemoryPool: Failed to open bootstrap shm");
      return;
    }
  }

  // ---- Ascend path: memfd hugepages + mbind + devHostRegister ----

  static constexpr size_t HUGEPAGE_SIZE = 2UL * 1024 * 1024;
  size_t pool_size = (buffer_size + HUGEPAGE_SIZE - 1) & ~(HUGEPAGE_SIZE - 1);

  for (int numa = 0; numa < num_numa; numa++) {
    int physical = topo ? topo->getPhysicalNUMA(numa) : 0;

    void* ptr = nullptr;

    if (is_creator) {
      char name[128];
      std::snprintf(name, sizeof(name), "pccl_%s_host_numa%d", unique_id_.c_str(), physical);
      int fd = -1;
      ptr = ShmAllocator::allocHugepage(name, pool_size, physical, &fd);
      if (!ptr) {
        LOG_ERROR("HostMemoryPool: ShmAllocator::allocHugepage failed for NUMA %d. "
                  "Ensure sufficient 2MB hugepages are available (see README).", physical);
        return;
      }

      ShmAllocator::shareFd(bs, numa, 0, fd);
      bs->host_sizes[numa] = pool_size;
      numa_fds_[numa] = fd;

    } else {
      int fd = ShmAllocator::openSharedFd(bs, numa, 0);
      if (fd < 0) {
        LOG_ERROR("HostMemoryPool: Failed to open shared fd for NUMA %d", numa);
        return;
      }
      ptr = mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      close(fd);
      if (ptr == MAP_FAILED) {
        LOG_ERROR("HostMemoryPool: mmap shared fd failed for NUMA %d: %s", numa, strerror(errno));
        return;
      }
    }

    devStatus status = devHostRegister(ptr, pool_size);
    if (status != devSuccess) {
      LOG_ERROR("HostMemoryPool: devHostRegister failed for NUMA %d, status=%d", numa, static_cast<int>(status));
      return;
    }

    numa_buffers_[numa] = ptr;
    numa_sizes_[numa] = pool_size;

    LOG_INFO("HostMemoryPool: %s %zu bytes (hugepage+registered) for logical NUMA %d (physical %d) at %p",
             is_creator ? "Created" : "Joined", pool_size, numa, physical, ptr);
  }

  if (is_creator) {
    bs->ready.store(1, std::memory_order_release);
  } else {
    bs->joined_count.fetch_add(1, std::memory_order_acq_rel);
  }

  pinThreadToNuma(-1);
}

HostMemoryPool::~HostMemoryPool() {
  if (!multi_process_) {
    // Single-process: simple devFreeHost
    for (auto& kv : numa_buffers_) {
      devFreeHost(kv.second);
    }
  } else {
    // Ascend cleanup: unregister from device, unmap memory
    for (auto& kv : numa_buffers_) {
      devHostUnregister(kv.second);
      if (is_creator_) {
        ShmAllocator::freeHugepage(kv.second, numa_sizes_[kv.first]);
      } else {
        munmap(kv.second, numa_sizes_[kv.first]);
      }
    }

    // Close shared memfd descriptors (creator only)
    for (auto& kv : numa_fds_) {
      if (kv.second >= 0) close(kv.second);
    }

    if (is_creator_) {
      ShmAllocator::destroyBootstrap(unique_id_, nullptr);
    }
  }

  numa_buffers_.clear();
  numa_sizes_.clear();
  numa_fds_.clear();
  LOG_INFO("HostMemoryPool: Released all buffers (multi_process=%d)", multi_process_);
}

void* HostMemoryPool::getChunkBuffer(int numa_node, int chunk_idx, size_t chunk_size) {
  auto it = numa_buffers_.find(numa_node);
  if (it == numa_buffers_.end()) {
    LOG_ERROR("HostMemoryPool: No buffer for NUMA %d", numa_node);
    return nullptr;
  }
  size_t offset = alignedDataOffset() + chunk_idx * chunk_size;
  if (offset + chunk_size > numa_sizes_[numa_node]) {
    LOG_ERROR("HostMemoryPool: Chunk %d (size=%zu) exceeds buffer for NUMA %d", chunk_idx, chunk_size, numa_node);
    return nullptr;
  }
  return static_cast<char*>(it->second) + offset;
}

size_t HostMemoryPool::getAllocatedSize(int numa_node) const {
  auto it = numa_sizes_.find(numa_node);
  return it == numa_sizes_.end() ? 0 : it->second;
}

int HostMemoryPool::getRankNUMANode(int rank) const {
  if (topo_)
    return topo_->getNUMAForDevice(rank);
  return 0;
}

}  // namespace pccl
