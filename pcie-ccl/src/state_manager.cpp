#include "include/state_manager.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "include/log.hpp"
#include "include/numa_topology.hpp"

#ifdef PCCL_SINGLE_PROCESS
#include "include/hal/device_rt.hpp"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pccl {

#ifdef PCCL_SINGLE_PROCESS
static std::mutex s_state_mtx;
static GlobalState* s_shared_state = nullptr;
static int s_state_ref_count = 0;

StateManager::StateManager(int rank, int world_size)
    : state_(nullptr), rank_(rank), world_size_(world_size), is_owner_(rank == 0) {
  std::lock_guard<std::mutex> lock(s_state_mtx);

  if (s_shared_state == nullptr) {
    LOG_INFO("Rank %d: Allocating GlobalState via devMallocHost (%zu bytes)", rank_, sizeof(GlobalState));

    NumaTopology::initialize();

    void* ptr = nullptr;
    devStatus status = devMallocHost(&ptr, sizeof(GlobalState));
    if (status != devSuccess || ptr == nullptr) {
      LOG_ERROR("Rank %d: devMallocHost failed for GlobalState (status=%d)", rank_, status);
      exit(EXIT_FAILURE);
    }

    std::memset(ptr, 0, sizeof(GlobalState));
    s_shared_state = static_cast<GlobalState*>(ptr);
    s_shared_state->initialized_count.store(0, std::memory_order_release);

    LOG_INFO("Rank %d: GlobalState allocated at %p", rank_, (void*)s_shared_state);
  }

  state_ = s_shared_state;
  s_state_ref_count++;
  LOG_INFO("Rank %d: Attached to GlobalState at %p (ref_count=%d)", rank_, (void*)state_, s_state_ref_count);
}

StateManager::~StateManager() {
  if (!state_) {
    LOG_WARN("Rank %d: State is already null, nothing to cleanup", rank_);
    return;
  }

  LOG_INFO("Rank %d: Cleaning up state", rank_);

  std::lock_guard<std::mutex> lock(s_state_mtx);
  state_ = nullptr;
  s_state_ref_count--;

  if (s_state_ref_count == 0) {
    LOG_INFO("Rank %d: Last reference, freeing GlobalState", rank_);
    devFreeHost(s_shared_state);
    s_shared_state = nullptr;
  }

  LOG_INFO("Rank %d: State cleanup completed (ref_count=%d)", rank_, s_state_ref_count);
}

#else  // Multi-process mode (original implementation)

StateManager::StateManager(int rank, int world_size)
    : state_(nullptr), rank_(rank), world_size_(world_size), is_owner_(rank == 0) {
  const char* shm_name = "/pccl_global_state";
  const size_t shm_size = sizeof(GlobalState);

  int shm_fd;

  if (is_owner_) {
    LOG_INFO("Rank 0: Initializing shared memory (%zu bytes)", shm_size);

    // Initialize NUMA topology
    NumaTopology::initialize();

    // Set preferred NUMA node to 0 for GlobalState allocation
    NumaTopology::setPreferredNode(0);

    // Remove any existing shared memory object
    shm_unlink(shm_name);

    // Create new shared memory object
    shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
      LOG_ERROR("Failed to create shared memory '%s': %s", shm_name, strerror(errno));
      exit(EXIT_FAILURE);
    }

    // Set the size of the shared memory
    if (ftruncate(shm_fd, shm_size) == -1) {
      LOG_ERROR("Failed to set shared memory size: %s", strerror(errno));
      close(shm_fd);
      shm_unlink(shm_name);
      exit(EXIT_FAILURE);
    }

    // Map the shared memory into address space
    state_ = static_cast<GlobalState*>(mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));

    if (state_ == MAP_FAILED) {
      LOG_ERROR("Failed to map shared memory: %s", strerror(errno));
      close(shm_fd);
      shm_unlink(shm_name);
      exit(EXIT_FAILURE);
    }

    // Bind GlobalState memory to NUMA node 0
    NumaTopology::bindMemoryToNode(state_, shm_size, 0);

    // Initialize the global state (zero out all atomic versions)
    std::memset(static_cast<void*>(state_), 0, shm_size);

    // Explicitly initialize initialized_count to 0
    state_->initialized_count.store(0, std::memory_order_release);

    close(shm_fd);
    LOG_INFO("Rank 0: Shared memory initialized successfully at %p (bound to NUMA node 0)", (void*)state_);
  } else {
    LOG_INFO("Rank %d: Attaching to shared memory", rank_);

    const int kMaxRetries = 300;  // 30 seconds total (100ms * 300)
    int retry_count = 0;

    while (true) {
      // Try to open shared memory (Rank 0 may not have created it yet)
      shm_fd = shm_open(shm_name, O_RDWR, 0666);
      if (shm_fd == -1) {
        if (errno != ENOENT || retry_count >= kMaxRetries) {
          LOG_ERROR("Rank %d: Failed to open shared memory '%s': %s (retries=%d)", rank_, shm_name, strerror(errno),
                    retry_count);
          exit(EXIT_FAILURE);
        }
        retry_count++;
        usleep(100000);  // 100ms between retries
        continue;
      }

      // Map the shared memory into address space
      state_ = static_cast<GlobalState*>(mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
      if (state_ == MAP_FAILED) {
        LOG_ERROR("Failed to map shared memory: %s", strerror(errno));
        close(shm_fd);
        exit(EXIT_FAILURE);
      }

      // --- Detect stale shared memory from a previous run ---
      bool is_stale = false;

      // Check 1: initialized_count >= world_size means a previous run's barrier
      // completed. This shm is definitely stale.
      int current_init = state_->initialized_count.load(std::memory_order_acquire);
      if (current_init >= world_size_) {
        LOG_WARN("Rank %d: Stale shared memory (initialized_count=%d >= world_size=%d)", rank_, current_init,
                 world_size_);
        is_stale = true;
      }

      // Check 2: Compare inode of our fd with the current shm name to detect if
      // Rank 0 has unlinked+recreated the shm object since we opened it.
      if (!is_stale) {
        struct stat my_stat;
        if (fstat(shm_fd, &my_stat) == 0) {
          int check_fd = shm_open(shm_name, O_RDWR, 0666);
          if (check_fd == -1) {
            // Name was removed (Rank 0 unlinked it) - our mapping is stale
            LOG_WARN("Rank %d: Shared memory name removed, retrying", rank_);
            is_stale = true;
          } else {
            struct stat check_stat;
            if (fstat(check_fd, &check_stat) == 0 && my_stat.st_ino != check_stat.st_ino) {
              LOG_WARN("Rank %d: Shared memory inode changed (%lu -> %lu), retrying", rank_,
                       (unsigned long)my_stat.st_ino, (unsigned long)check_stat.st_ino);
              is_stale = true;
            }
            close(check_fd);
          }
        }
      }

      if (is_stale) {
        munmap(state_, shm_size);
        state_ = nullptr;
        close(shm_fd);
        if (retry_count >= kMaxRetries) {
          LOG_ERROR("Rank %d: Failed to attach to fresh shared memory after %d retries", rank_, retry_count);
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
      LOG_INFO("Rank %d: Attached to shared memory after %d retries", rank_, retry_count);
    }

    LOG_INFO("Rank %d: Shared memory attached successfully at %p", rank_, (void*)state_);
  }
}

StateManager::~StateManager() {
  const char* shm_name = "/pccl_global_state";
  const size_t shm_size = sizeof(GlobalState);

  if (!state_) {
    LOG_WARN("Rank %d: State is already null, nothing to cleanup", rank_);
    return;
  }

  LOG_INFO("Rank %d: Cleaning up shared memory", rank_);

  // Unmap the shared memory
  if (munmap(state_, shm_size) == -1) {
    LOG_ERROR("Rank %d: Failed to unmap shared memory: %s", rank_, strerror(errno));
  } else {
    LOG_DEBUG("Rank %d: Shared memory unmapped successfully", rank_);
  }

  state_ = nullptr;

  // Only rank 0 removes the shared memory object
  if (is_owner_) {
    if (shm_unlink(shm_name) == -1) {
      LOG_ERROR("Rank 0: Failed to unlink shared memory '%s': %s", shm_name, strerror(errno));
    } else {
      LOG_INFO("Rank 0: Shared memory object removed successfully");
    }
  }

  LOG_INFO("Rank %d: Shared memory cleanup completed", rank_);
}

#endif  // PCCL_SINGLE_PROCESS

void StateManager::dumpState() const {
  if (!isValid()) {
    LOG_WARN("StateManager::dumpState called on invalid state");
    return;
  }

  LOG_INFO("=== StateManager Dump (Rank %d/%d) ===", rank_, world_size_);

  int num_numa_nodes = NumaTopology::getNumNUMANodes();
  for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
    LOG_INFO("NUMA Node %d:", numa_node);
    for (int c = 0; c < MAX_CHUNKS; c++) {
      uint64_t version = getChunkVersion(numa_node, c, std::memory_order_relaxed);
      if (version != 0) {  // Only print non-zero versions
        LOG_INFO("  Chunk %d: version=%lu", c, version);
      }
    }
  }

  LOG_INFO("=== End StateManager Dump ===");
}

void StateManager::resetAllChunkVersions() {
  if (!isValid()) {
    LOG_WARN("StateManager::resetAllChunkVersions called on invalid state");
    return;
  }

  LOG_DEBUG("Rank %d: Resetting all chunk versions to 0", rank_);

  int num_numa_nodes = NumaTopology::getNumNUMANodes();
  for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
    for (int c = 0; c < MAX_CHUNKS; c++) {
      state_->numa_nodes[numa_node].chunks[c].version.store(0, std::memory_order_release);
    }
  }

  LOG_DEBUG("Rank %d: All chunk versions reset successfully", rank_);
}

void StateManager::barrierInit() {
  // Increment initialized count
  int count = state_->initialized_count.fetch_add(1, std::memory_order_acq_rel) + 1;
  LOG_INFO("Rank %d: Initialized (%d/%d ranks ready)", rank_, count, world_size_);

  // Wait for all ranks to initialize
  while (state_->initialized_count.load(std::memory_order_acquire) < world_size_) {
  }
  LOG_INFO("Rank %d: All ranks initialized, proceeding", rank_);
}

}  // namespace pccl
