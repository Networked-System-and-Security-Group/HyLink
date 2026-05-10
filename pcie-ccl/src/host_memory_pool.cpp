#include "include/host_memory_pool.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>

#include "include/hal/device_rt.hpp"
#include "include/log.hpp"
#include "include/numa_topology.hpp"

#ifndef PCCL_SINGLE_PROCESS
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pccl {

#ifdef PCCL_SINGLE_PROCESS
static std::mutex s_pool_mtx;
static std::map<int, void*> s_pool_buffers;    // numa_node -> devMallocHost buffer
static std::map<int, int> s_pool_ref_counts;   // numa_node -> ref count

HostMemoryPool::HostMemoryPool(int rank, int world_size, size_t chunk_size, int num_chunks)
    : my_rank_(rank), world_size_(world_size), chunk_size_(chunk_size), num_chunks_(num_chunks) {
  // Initialize NUMA topology
  NumaTopology::initialize();

  // Determine this rank's NUMA node
  my_numa_node_ = NumaTopology::getRankNUMANode(my_rank_);
  LOG_INFO("Rank %d: Assigned to NUMA node %d", my_rank_, my_numa_node_);

  is_owner_ = isFirstRankInNUMA();

  // Calculate memory size for each NUMA pool (data offset aligned to 64 bytes for SIMD)
  const size_t pool_size = alignedDataOffset() + (chunk_size_ * num_chunks_);

  // Determine all unique NUMA nodes used by ranks
  std::set<int> unique_numa_nodes;
  for (int r = 0; r < world_size_; r++) {
    unique_numa_nodes.insert(NumaTopology::getRankNUMANode(r));
  }

  LOG_INFO("Rank %d: Total %zu unique NUMA nodes in use", my_rank_, unique_numa_nodes.size());

  // Allocate or share devMallocHost buffers for each NUMA node
  {
    std::lock_guard<std::mutex> lock(s_pool_mtx);

    for (int numa_node : unique_numa_nodes) {
      if (s_pool_buffers.find(numa_node) == s_pool_buffers.end()) {
        // First thread for this NUMA node: allocate via devMallocHost
        LOG_INFO("Rank %d: Allocating NUMA pool %d via devMallocHost (%zu bytes)", my_rank_, numa_node, pool_size);

        void* buffer_ptr = nullptr;
        devStatus status = devMallocHost(&buffer_ptr, pool_size);
        if (status != devSuccess || buffer_ptr == nullptr) {
          LOG_ERROR("Rank %d: devMallocHost failed for NUMA pool %d (status=%d)", my_rank_, numa_node, status);
          exit(EXIT_FAILURE);
        }

        // Initialize header
        PoolHeader* header = static_cast<PoolHeader*>(buffer_ptr);
        header->chunk_size = chunk_size_;
        header->num_chunks = num_chunks_;
        for (int i = 0; i < MAX_RANKS; i++) {
          header->initialized[i].store(false, std::memory_order_release);
        }

        s_pool_buffers[numa_node] = buffer_ptr;
        s_pool_ref_counts[numa_node] = 0;

        LOG_INFO("Rank %d: NUMA pool %d allocated at %p", my_rank_, numa_node, buffer_ptr);
      }

      numa_buffers_[numa_node] = s_pool_buffers[numa_node];
      numa_pinned_[numa_node] = true;  // devMallocHost memory is already pinned
      s_pool_ref_counts[numa_node]++;

      LOG_INFO("Rank %d: Attached to NUMA pool %d at %p (ref_count=%d)", my_rank_, numa_node,
               numa_buffers_[numa_node], s_pool_ref_counts[numa_node]);
    }
  }

  // Mark this rank as initialized in its NUMA pool's header
  PoolHeader* my_header = static_cast<PoolHeader*>(numa_buffers_[my_numa_node_]);
  my_header->initialized[my_rank_].store(true, std::memory_order_release);

  LOG_INFO("Rank %d: Host memory pool initialization completed", my_rank_);
}

HostMemoryPool::~HostMemoryPool() {
  LOG_INFO("Rank %d: Cleaning up host memory pool", my_rank_);

  std::lock_guard<std::mutex> lock(s_pool_mtx);

  for (auto& entry : numa_buffers_) {
    int numa_node = entry.first;

    auto ref_it = s_pool_ref_counts.find(numa_node);
    if (ref_it != s_pool_ref_counts.end()) {
      ref_it->second--;
      LOG_DEBUG("Rank %d: NUMA pool %d ref_count=%d", my_rank_, numa_node, ref_it->second);

      if (ref_it->second == 0) {
        LOG_INFO("Rank %d: Last reference to NUMA pool %d, freeing", my_rank_, numa_node);
        auto buf_it = s_pool_buffers.find(numa_node);
        if (buf_it != s_pool_buffers.end()) {
          devFreeHost(buf_it->second);
          s_pool_buffers.erase(buf_it);
        }
        s_pool_ref_counts.erase(ref_it);
      }
    }
  }

  numa_buffers_.clear();

  LOG_INFO("Rank %d: Host memory pool cleanup completed", my_rank_);
}

#else  // Multi-process mode (original implementation)

HostMemoryPool::HostMemoryPool(int rank, int world_size, size_t chunk_size, int num_chunks)
    : my_rank_(rank), world_size_(world_size), chunk_size_(chunk_size), num_chunks_(num_chunks) {
  // Initialize NUMA topology
  NumaTopology::initialize();

  // Determine this rank's NUMA node
  my_numa_node_ = NumaTopology::getRankNUMANode(my_rank_);
  LOG_INFO("Rank %d: Assigned to NUMA node %d", my_rank_, my_numa_node_);

  // Check if this rank is the first in its NUMA node (owner/creator)
  is_owner_ = isFirstRankInNUMA();

  // Calculate memory size for each NUMA pool (data offset aligned to 64 bytes for SIMD)
  const size_t pool_size = alignedDataOffset() + (chunk_size_ * num_chunks_);

  // Determine all unique NUMA nodes used by ranks
  std::set<int> unique_numa_nodes;
  for (int r = 0; r < world_size_; r++) {
    unique_numa_nodes.insert(NumaTopology::getRankNUMANode(r));
  }

  LOG_INFO("Rank %d: Total %zu unique NUMA nodes in use", my_rank_, unique_numa_nodes.size());

  // Create or attach to shared memory for each NUMA node
  for (int numa_node : unique_numa_nodes) {
    char shm_name[64];
    snprintf(shm_name, sizeof(shm_name), "/pccl_numa_pool_%d", numa_node);

    int shm_fd;
    void* buffer_ptr;

    // Check if this rank should create the pool for this NUMA node
    bool should_create = (numa_node == my_numa_node_) && is_owner_;

    if (should_create) {
      LOG_INFO("Rank %d: Creating NUMA pool %d (%zu bytes)", my_rank_, numa_node, pool_size);

      // Remove any existing shared memory
      shm_unlink(shm_name);

      // Create shared memory object
      shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
      if (shm_fd == -1) {
        LOG_ERROR("Rank %d: Failed to create shared memory '%s': %s", my_rank_, shm_name, strerror(errno));
        exit(EXIT_FAILURE);
      }

      // Set size
      if (ftruncate(shm_fd, pool_size) == -1) {
        LOG_ERROR("Rank %d: Failed to set shared memory size: %s", my_rank_, strerror(errno));
        close(shm_fd);
        shm_unlink(shm_name);
        exit(EXIT_FAILURE);
      }

      // Map into address space
      buffer_ptr = mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

      if (buffer_ptr == MAP_FAILED) {
        LOG_ERROR("Rank %d: Failed to map shared memory: %s", my_rank_, strerror(errno));
        close(shm_fd);
        shm_unlink(shm_name);
        exit(EXIT_FAILURE);
      }

      // Bind memory to the NUMA node
      NumaTopology::bindMemoryToNode(buffer_ptr, pool_size, numa_node);

      // Register memory as pinned for better PCIe transfer performance
      devStatus status = devHostRegister(buffer_ptr, pool_size);
      if (status != devSuccess) {
        LOG_WARN(
            "Rank %d: Failed to register NUMA pool %d as pinned memory (status=%d), "
            "continuing with regular memory",
            my_rank_, numa_node, status);
        numa_pinned_[numa_node] = false;
      } else {
        LOG_INFO("Rank %d: Registered NUMA pool %d as pinned memory", my_rank_, numa_node);
        numa_pinned_[numa_node] = true;
      }

      // Initialize header
      PoolHeader* header = static_cast<PoolHeader*>(buffer_ptr);
      header->chunk_size = chunk_size_;
      header->num_chunks = num_chunks_;
      for (int i = 0; i < MAX_RANKS; i++) {
        header->initialized[i].store(false, std::memory_order_release);
      }

      LOG_INFO("Rank %d: NUMA pool %d created and initialized at %p", my_rank_, numa_node, buffer_ptr);
    } else {
      // Attach to existing shared memory
      LOG_INFO("Rank %d: Attaching to NUMA pool %d", my_rank_, numa_node);

      const int kMaxRetries = 300;  // 30 seconds total (100ms * 300)
      int retry_count = 0;

      while (true) {
        // Try to open shared memory (owner may not have created it yet)
        shm_fd = shm_open(shm_name, O_RDWR, 0666);
        if (shm_fd == -1) {
          if (errno != ENOENT || retry_count >= kMaxRetries) {
            LOG_ERROR("Rank %d: Failed to open shared memory '%s': %s (retries=%d)", my_rank_, shm_name,
                      strerror(errno), retry_count);
            exit(EXIT_FAILURE);
          }
          retry_count++;
          usleep(100000);  // 100ms between retries
          continue;
        }

        // Map into address space
        buffer_ptr = mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (buffer_ptr == MAP_FAILED) {
          LOG_ERROR("Rank %d: Failed to map shared memory: %s", my_rank_, strerror(errno));
          close(shm_fd);
          exit(EXIT_FAILURE);
        }

        // --- Detect stale shared memory from a previous run ---
        bool is_stale = false;

        // Check 1: If our rank is already marked initialized, this is stale.
        if (numa_node == my_numa_node_) {
          PoolHeader* header = static_cast<PoolHeader*>(buffer_ptr);
          if (header->initialized[my_rank_].load(std::memory_order_acquire)) {
            LOG_WARN("Rank %d: Stale NUMA pool %d (rank already initialized)", my_rank_, numa_node);
            is_stale = true;
          }
        }

        // Check 2: Compare inode to detect if owner has replaced the shm object.
        if (!is_stale) {
          struct stat my_stat;
          if (fstat(shm_fd, &my_stat) == 0) {
            int check_fd = shm_open(shm_name, O_RDWR, 0666);
            if (check_fd == -1) {
              LOG_WARN("Rank %d: NUMA pool %d name removed, retrying", my_rank_, numa_node);
              is_stale = true;
            } else {
              struct stat check_stat;
              if (fstat(check_fd, &check_stat) == 0 && my_stat.st_ino != check_stat.st_ino) {
                LOG_WARN("Rank %d: NUMA pool %d inode changed, retrying", my_rank_, numa_node);
                is_stale = true;
              }
              close(check_fd);
            }
          }
        }

        if (is_stale) {
          munmap(buffer_ptr, pool_size);
          buffer_ptr = nullptr;
          close(shm_fd);
          if (retry_count >= kMaxRetries) {
            LOG_ERROR("Rank %d: Failed to attach to fresh NUMA pool %d after %d retries", my_rank_, numa_node,
                      retry_count);
            exit(EXIT_FAILURE);
          }
          retry_count++;
          usleep(100000);  // 100ms between retries
          continue;
        }

        close(shm_fd);
        break;
      }

      if (retry_count > 0) {
        LOG_INFO("Rank %d: Attached to NUMA pool %d after %d retries", my_rank_, numa_node, retry_count);
      }

      // Register memory as pinned for better PCIe transfer performance
      devStatus status = devHostRegister(buffer_ptr, pool_size);
      if (status != devSuccess) {
        LOG_WARN(
            "Rank %d: Failed to register NUMA pool %d as pinned memory (status=%d), "
            "continuing with regular memory",
            my_rank_, numa_node, status);
        numa_pinned_[numa_node] = false;
      } else {
        LOG_INFO("Rank %d: Registered NUMA pool %d as pinned memory", my_rank_, numa_node);
        numa_pinned_[numa_node] = true;
      }

      LOG_INFO("Rank %d: Attached to NUMA pool %d at %p", my_rank_, numa_node, buffer_ptr);
    }

    // Store the buffer pointer and fd
    numa_buffers_[numa_node] = buffer_ptr;
    numa_shm_fds_[numa_node] = shm_fd;

    close(shm_fd);
  }

  // Mark this rank as initialized in its NUMA pool's header
  PoolHeader* my_header = static_cast<PoolHeader*>(numa_buffers_[my_numa_node_]);
  my_header->initialized[my_rank_].store(true, std::memory_order_release);

  LOG_INFO("Rank %d: Host memory pool initialization completed", my_rank_);
}

HostMemoryPool::~HostMemoryPool() {
  const size_t pool_size = alignedDataOffset() + (chunk_size_ * num_chunks_);

  LOG_INFO("Rank %d: Cleaning up host memory pool", my_rank_);

  // Unmap all NUMA buffers
  for (auto& entry : numa_buffers_) {
    int numa_node = entry.first;
    void* buffer_ptr = entry.second;

    if (buffer_ptr) {
      // Unregister pinned memory before unmapping
      auto pinned_it = numa_pinned_.find(numa_node);
      if (pinned_it != numa_pinned_.end() && pinned_it->second) {
        devStatus status = devHostUnregister(buffer_ptr);
        if (status != devSuccess) {
          LOG_WARN("Rank %d: Failed to unregister NUMA pool %d (status=%d)", my_rank_, numa_node, status);
        } else {
          LOG_DEBUG("Rank %d: Unregistered NUMA pool %d", my_rank_, numa_node);
        }
      }

      if (munmap(buffer_ptr, pool_size) == -1) {
        LOG_ERROR("Rank %d: Failed to unmap NUMA pool %d: %s", my_rank_, numa_node, strerror(errno));
      } else {
        LOG_DEBUG("Rank %d: Unmapped NUMA pool %d", my_rank_, numa_node);
      }
    }
  }

  // Only the owner (first rank in NUMA node) removes the shared memory object
  if (is_owner_) {
    char shm_name[64];
    snprintf(shm_name, sizeof(shm_name), "/pccl_numa_pool_%d", my_numa_node_);

    if (shm_unlink(shm_name) == -1) {
      LOG_ERROR("Rank %d: Failed to unlink NUMA pool %d: %s", my_rank_, my_numa_node_, strerror(errno));
    } else {
      LOG_INFO("Rank %d: Removed NUMA pool %d object", my_rank_, my_numa_node_);
    }
  }

  LOG_INFO("Rank %d: Host memory pool cleanup completed", my_rank_);
}

#endif  // PCCL_SINGLE_PROCESS

void* HostMemoryPool::getChunkBuffer(int numa_node, int chunk_idx) {
  if (!isValid()) {
    LOG_ERROR("HostMemoryPool::getChunkBuffer called on invalid pool");
    return nullptr;
  }

  if (chunk_idx < 0 || chunk_idx >= num_chunks_) {
    LOG_ERROR("HostMemoryPool::getChunkBuffer: invalid chunk_idx %d (num_chunks=%d)", chunk_idx, num_chunks_);
    return nullptr;
  }

  auto it = numa_buffers_.find(numa_node);
  if (it == numa_buffers_.end()) {
    LOG_ERROR("HostMemoryPool::getChunkBuffer: NUMA node %d not found", numa_node);
    return nullptr;
  }

  void* base = it->second;
  size_t offset = alignedDataOffset() + (chunk_idx * chunk_size_);
  return static_cast<char*>(base) + offset;
}

void* HostMemoryPool::getLocalChunk(int chunk_idx) {
  return getChunkBuffer(my_numa_node_, chunk_idx);
}

void* HostMemoryPool::getRankChunk(int rank, int chunk_idx) {
  int numa_node = NumaTopology::getRankNUMANode(rank);
  return getChunkBuffer(numa_node, chunk_idx);
}

void* HostMemoryPool::getNUMABuffer(int numa_node) {
  if (!isValid()) {
    LOG_ERROR("HostMemoryPool::getNUMABuffer called on invalid pool");
    return nullptr;
  }

  auto it = numa_buffers_.find(numa_node);
  if (it == numa_buffers_.end()) {
    LOG_ERROR("HostMemoryPool::getNUMABuffer: NUMA node %d not found", numa_node);
    return nullptr;
  }

  return it->second;
}

void* HostMemoryPool::getMyNUMABuffer() {
  return getNUMABuffer(my_numa_node_);
}

int HostMemoryPool::getRankNUMANode(int rank) const {
  return NumaTopology::getRankNUMANode(rank);
}

bool HostMemoryPool::isFirstRankInNUMA() const {
  int my_numa = my_numa_node_;

  // Check if any rank with smaller ID is on the same NUMA node
  for (int r = 0; r < my_rank_; r++) {
    if (NumaTopology::getRankNUMANode(r) == my_numa) {
      return false;  // Not the first
    }
  }

  return true;  // This is the first rank in this NUMA node
}

int HostMemoryPool::getNumUniqueNUMANodes() const {
  std::set<int> unique_nodes;
  for (int r = 0; r < world_size_; r++) {
    unique_nodes.insert(NumaTopology::getRankNUMANode(r));
  }
  return unique_nodes.size();
}

}  // namespace pccl
