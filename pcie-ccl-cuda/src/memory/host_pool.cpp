#include "host_pool.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include "../hal/device_rt.hpp"
#include "../topology/numa.hpp"
#include "../topology/topology.hpp"
#include "../util/log.hpp"
#include "shm_allocator.hpp"

#ifdef PCCL_DEVICE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define CU_CHECK(call)                                                                        \
  do {                                                                                        \
    CUresult err = (call);                                                                    \
    if (err != CUDA_SUCCESS) {                                                                \
      const char* errStr = nullptr;                                                           \
      cuGetErrorString(err, &errStr);                                                         \
      LOG_ERROR("HostMemoryPool: %s failed: %s (CUresult %d)", #call, errStr ? errStr : "?", \
                static_cast<int>(err));                                                       \
      std::abort();                                                                           \
    }                                                                                         \
  } while (0)
#endif

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

#if defined(PCCL_DEVICE_CUDA)
// PCCL_HOST_POOL_SHM=1: back the host pool with a memfd (4K pages, NUMA placed by
// first touch) registered via cudaHostRegister, instead of cuMem host-NUMA
// allocations. On an IOMMU-translated EPYC + PCIe Gen4 host, cuMem's 2MB-granular
// pool made host DMA bandwidth bimodal across processes (RS 1G busbw 6.3 vs 7.1
// GB/s); the memfd pool was a stable 7.6-7.9. The fd is still shared via UDS.
static bool hostPoolShmMode() {
  static const bool v = [] {
    const char* e = std::getenv("PCCL_HOST_POOL_SHM");
    return e && e[0] == '1';
  }();
  return v;
}
#endif

// ---- Multi-process constructor ----
//
// Platform dispatch:
//   CUDA:   cuMem VMM API — explicit NUMA binding, native GPU access, IPC via fd export
//   Ascend: memfd_create(MFD_HUGETLB) + mbind + devHostRegister
//
// Both paths reuse the BootstrapShm mechanism for cross-process fd exchange.

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

#if defined(PCCL_DEVICE_CUDA)
  // ---- CUDA path: cuMem VMM API with NUMA binding ----
  //
  // IPC fd exchange uses UDS + SCM_RIGHTS (matching the NCCL pattern) because
  // cuMemExportToShareableHandle fds must be *duplicated* across processes —
  // reopening via /proc/<pid>/fd/ creates a new file description that the CUDA
  // driver does not recognise as a valid shareable handle.

  // Query current GPU and its NUMA node
  int cudaDev;
  cudaGetDevice(&cudaDev);
  CUdevice cuDev;
  CU_CHECK(cuDeviceGet(&cuDev, cudaDev));
  int cpuNumaNodeId = 0;
  cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cuDev);
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;

  // Determine allocation granularity for cuMem (typically 2MB)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = cpuNumaNodeId;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t pool_size = (buffer_size + granularity - 1) & ~(granularity - 1);

  if (is_creator) {
    // ---- Creator: allocate all NUMA buffers, then distribute fds via UDS ----

    for (int numa = 0; numa < num_numa; numa++) {
      int physical = topo ? topo->getPhysicalNUMA(numa) : 0;

      CUmemAllocationProp alloc_prop = {};
      alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      alloc_prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      alloc_prop.location.id = physical;
      alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

      if (hostPoolShmMode()) {
        char name[128];
        std::snprintf(name, sizeof(name), "pccl_%s_host_numa%d", unique_id_.c_str(), physical);
        int fd = memfd_create(name, MFD_CLOEXEC);
        if (fd < 0 || ftruncate(fd, static_cast<off_t>(pool_size)) != 0) {
          LOG_ERROR("HostMemoryPool: memfd_create/ftruncate failed: %s", strerror(errno));
          std::abort();
        }
        void* ptr = mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
          LOG_ERROR("HostMemoryPool: mmap memfd failed: %s", strerror(errno));
          std::abort();
        }
        pinThreadToNuma(physical);
        std::memset(ptr, 0, pool_size);
        if (cudaHostRegister(ptr, pool_size, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
          LOG_ERROR("HostMemoryPool: cudaHostRegister failed for NUMA %d", physical);
          std::abort();
        }
        bs->host_sizes[numa] = pool_size;
        numa_buffers_[numa] = ptr;
        numa_sizes_[numa] = pool_size;
        numa_fds_[numa] = fd;
        LOG_INFO("HostMemoryPool: Created %zu bytes (memfd+registered, NUMA %d) for logical NUMA %d at %p",
                 pool_size, physical, numa, ptr);
        continue;
      }

      CUmemGenericAllocationHandle handle;
      CU_CHECK(cuMemCreate(&handle, pool_size, &alloc_prop, 0));

      CUdeviceptr dptr = 0;
      CU_CHECK(cuMemAddressReserve(&dptr, pool_size, granularity, 0, 0));
      CU_CHECK(cuMemMap(dptr, pool_size, 0, handle, 0));

      // Set access: GPU device
      CUmemAccessDesc accessDesc;
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = cudaDev;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CU_CHECK(cuMemSetAccess(dptr, pool_size, &accessDesc, 1));

      // Set access: host NUMA
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      accessDesc.location.id = cpuNumaNodeId;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CU_CHECK(cuMemSetAccess(dptr, pool_size, &accessDesc, 1));

      // Export handle as POSIX fd for cross-process sharing
      int shareable_fd = -1;
      CU_CHECK(cuMemExportToShareableHandle(&shareable_fd, handle,
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

      bs->host_sizes[numa] = pool_size;
      numa_buffers_[numa] = reinterpret_cast<void*>(dptr);
      numa_sizes_[numa] = pool_size;
      numa_fds_[numa] = shareable_fd;
      ipc_handles_[numa] = static_cast<uint64_t>(handle);

      // First-touch to ensure physical pages are allocated
      std::memset(reinterpret_cast<void*>(dptr), 0, pool_size);

      LOG_INFO("HostMemoryPool: Created %zu bytes (cuMem, NUMA %d) for logical NUMA %d at %p",
               pool_size, physical, numa, reinterpret_cast<void*>(dptr));
    }

    // Distribute exported fds to joiners via UDS SCM_RIGHTS
    if (bs->expected_joiners > 0) {
      int uds_server = ShmAllocator::createFdServer(unique_id_, bs->expected_joiners);
      if (uds_server < 0) {
        LOG_ERROR("HostMemoryPool: Failed to create UDS fd-server");
        std::abort();
      }
      // Signal ready — joiners may now connect to the UDS server
      bs->ready.store(1, std::memory_order_release);

      int fds[MAX_NUMA_NODES];
      for (int n = 0; n < num_numa; n++) fds[n] = numa_fds_[n];
      for (int j = 0; j < bs->expected_joiners; j++) {
        if (ShmAllocator::serveFds(uds_server, fds, num_numa) < 0) {
          LOG_ERROR("HostMemoryPool: Failed to serve fds to joiner %d", j);
          std::abort();
        }
      }
      close(uds_server);
    } else {
      bs->ready.store(1, std::memory_order_release);
    }

  } else {
    // ---- Joiner: receive fds via UDS, then import and map ----

    int received_fds[MAX_NUMA_NODES];
    if (ShmAllocator::receiveFds(unique_id_, received_fds, num_numa) < 0) {
      LOG_ERROR("HostMemoryPool: Failed to receive fds from creator");
      std::abort();
    }

    for (int numa = 0; numa < num_numa; numa++) {
      int physical = topo ? topo->getPhysicalNUMA(numa) : 0;
      int fd = received_fds[numa];

      // Use the creator's actual allocation size for consistency
      size_t remote_size = bs->host_sizes[numa];
      if (remote_size == 0) remote_size = pool_size;

      if (hostPoolShmMode()) {
        void* ptr = mmap(nullptr, remote_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) {
          LOG_ERROR("HostMemoryPool: mmap received memfd failed: %s", strerror(errno));
          std::abort();
        }
        if (cudaHostRegister(ptr, remote_size, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
          LOG_ERROR("HostMemoryPool: cudaHostRegister failed for NUMA %d", physical);
          std::abort();
        }
        numa_buffers_[numa] = ptr;
        numa_sizes_[numa] = remote_size;
        LOG_INFO("HostMemoryPool: Joined %zu bytes (memfd+registered, NUMA %d) for logical NUMA %d at %p",
                 remote_size, physical, numa, ptr);
        continue;
      }

      CUmemGenericAllocationHandle handle;
      CU_CHECK(cuMemImportFromShareableHandle(
          &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(fd);

      CUdeviceptr dptr = 0;
      CU_CHECK(cuMemAddressReserve(&dptr, remote_size, 0, 0, 0));
      CU_CHECK(cuMemMap(dptr, remote_size, 0, handle, 0));

      // Set access: GPU device
      CUmemAccessDesc accessDesc;
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = cudaDev;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CU_CHECK(cuMemSetAccess(dptr, remote_size, &accessDesc, 1));

      // Set access: host NUMA
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      accessDesc.location.id = cpuNumaNodeId;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CU_CHECK(cuMemSetAccess(dptr, remote_size, &accessDesc, 1));

      numa_buffers_[numa] = reinterpret_cast<void*>(dptr);
      numa_sizes_[numa] = remote_size;
      ipc_handles_[numa] = static_cast<uint64_t>(handle);

      LOG_INFO("HostMemoryPool: Joined %zu bytes (cuMem, NUMA %d) for logical NUMA %d at %p",
               remote_size, physical, numa, reinterpret_cast<void*>(dptr));
    }

    bs->joined_count.fetch_add(1, std::memory_order_acq_rel);
  }

#elif defined(PCCL_DEVICE_ASCEND)
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
        std::abort();
      }

      ShmAllocator::shareFd(bs, numa, 0, fd);
      bs->host_sizes[numa] = pool_size;
      numa_fds_[numa] = fd;

    } else {
      int fd = ShmAllocator::openSharedFd(bs, numa, 0);
      if (fd < 0) {
        LOG_ERROR("HostMemoryPool: Failed to open shared fd for NUMA %d", numa);
        std::abort();
      }
      ptr = mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      close(fd);
      if (ptr == MAP_FAILED) {
        LOG_ERROR("HostMemoryPool: mmap shared fd failed for NUMA %d: %s", numa, strerror(errno));
        std::abort();
      }
    }

    devStatus status = devHostRegister(ptr, pool_size);
    if (status != devSuccess) {
      LOG_ERROR("HostMemoryPool: devHostRegister failed for NUMA %d, status=%d", numa, static_cast<int>(status));
      std::abort();
    }

    numa_buffers_[numa] = ptr;
    numa_sizes_[numa] = pool_size;

    LOG_INFO("HostMemoryPool: %s %zu bytes (hugepage+registered) for logical NUMA %d (physical %d) at %p",
             is_creator ? "Created" : "Joined", pool_size, numa, physical, ptr);
  }
#endif  // PCCL_DEVICE_CUDA / PCCL_DEVICE_ASCEND

  // CUDA path handles ready/joined_count internally (UDS exchange acts as
  // an implicit barrier).  Ascend path still needs the original logic.
#if !defined(PCCL_DEVICE_CUDA)
  if (is_creator) {
    bs->ready.store(1, std::memory_order_release);
  } else {
    bs->joined_count.fetch_add(1, std::memory_order_acq_rel);
  }
#endif

  pinThreadToNuma(-1);
}

HostMemoryPool::~HostMemoryPool() {
  if (!multi_process_) {
    // Single-process: simple devFreeHost
    for (auto& kv : numa_buffers_) {
      devFreeHost(kv.second);
    }
  } else {
#if defined(PCCL_DEVICE_CUDA)
    // CUDA cuMem cleanup: unmap, free virtual range, release physical allocation
    for (auto& kv : numa_buffers_) {
      if (hostPoolShmMode()) {
        cudaHostUnregister(kv.second);
        munmap(kv.second, numa_sizes_[kv.first]);
        continue;
      }
      int numa = kv.first;
      CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(kv.second);
      size_t size = numa_sizes_[numa];
      cuMemUnmap(dptr, size);
      cuMemAddressFree(dptr, size);
      auto hit = ipc_handles_.find(numa);
      if (hit != ipc_handles_.end()) {
        cuMemRelease(static_cast<CUmemGenericAllocationHandle>(hit->second));
      }
    }
#elif defined(PCCL_DEVICE_ASCEND)
    // Ascend cleanup: unregister from device, unmap memory
    for (auto& kv : numa_buffers_) {
      devHostUnregister(kv.second);
      if (is_creator_) {
        ShmAllocator::freeHugepage(kv.second, numa_sizes_[kv.first]);
      } else {
        munmap(kv.second, numa_sizes_[kv.first]);
      }
    }
#endif

    // Close IPC fds (creator only)
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
  ipc_handles_.clear();
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

int HostMemoryPool::getRankNUMANode(int rank) const {
  if (topo_)
    return topo_->getNUMAForGPU(rank);
  return 0;
}

}  // namespace pccl
