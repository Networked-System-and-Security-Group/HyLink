#include "shared_state.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "../util/debug_flags.hpp"
#include "../util/log.hpp"

namespace pccl {

namespace {

int barrierSlotForWhere(const char* where) {
  if (!where)
    return 0;
  if (std::strstr(where, "ProcessState::initialize"))
    return 0;
  if (std::strstr(where, "submitCollective"))
    return 1;
  if (std::strstr(where, "GraphExecutor::runGraph"))
    return 2;
  if (std::strstr(where, "GraphExecutor::start"))
    return 3;
  if (std::strstr(where, "HostStagingRegion::acquire"))
    return 4;
  if (std::strstr(where, "HostStagingRegion::release"))
    return 5;
  return 0;
}

bool stagingLeaseDisabled() {
  static int disabled = -1;
  if (disabled < 0) {
    const char* e = std::getenv("PCCL_DISABLE_STAGING_LEASE");
    disabled = (e && std::atoi(e) != 0) ? 1 : 0;
  }
  return disabled != 0;
}

int pcclTimeoutMsFromEnv(const char* name, int default_ms) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0')
    return default_ms;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0)
    return default_ms;
  return static_cast<int>(parsed);
}

void lockUserReadyCommSlot(std::atomic<int>& lock) {
  int expected = 0;
  while (!lock.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                     std::memory_order_relaxed)) {
    expected = 0;
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    _mm_pause();
#endif
    std::this_thread::yield();
  }
}

void unlockUserReadyCommSlot(std::atomic<int>& lock) {
  lock.store(0, std::memory_order_release);
}

}  // namespace

// ---- Single-process constructor (heap allocation) ----

SharedStateManager::SharedStateManager(int world_size) : world_size_(world_size) {
  // In-process allocation (no SHM)
  state_ = new SharedState();
  std::memset(state_, 0, sizeof(SharedState));
  new (state_) SharedState();
  valid_ = true;
  LOG_INFO("SharedStateManager: Allocated in-process state (%zu bytes)", sizeof(SharedState));
}

// ---- Multi-process constructor (POSIX shared memory) ----

SharedStateManager::SharedStateManager(int world_size, const std::string& unique_id, bool is_creator)
    : world_size_(world_size), multi_process_(true), is_creator_(is_creator), unique_id_(unique_id) {
  std::string name = "/pccl_" + unique_id + "_state";
  size_t size = sizeof(SharedState);

  int fd;
  if (is_creator) {
    // Remove stale shm
    shm_unlink(name.c_str());
    fd = shm_open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  } else {
    // Joiner: spin until shm exists
    fd = -1;
    const int timeout_ms = pcclTimeoutMsFromEnv("PCCL_BOOTSTRAP_TIMEOUT_MS", 120000);
    for (int attempt = 0; attempt < timeout_ms; attempt++) {
      fd = shm_open(name.c_str(), O_RDWR, 0666);
      if (fd >= 0) break;
      usleep(1000);
    }
  }

  if (fd < 0) {
    LOG_ERROR("SharedStateManager: shm_open(%s) failed: %s", name.c_str(), strerror(errno));
    state_ = nullptr;
    return;
  }

  if (is_creator) {
    if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
      LOG_ERROR("SharedStateManager: ftruncate failed: %s", strerror(errno));
      close(fd);
      shm_unlink(name.c_str());
      state_ = nullptr;
      return;
    }
  }

  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);

  if (p == MAP_FAILED) {
    LOG_ERROR("SharedStateManager: mmap failed: %s", strerror(errno));
    state_ = nullptr;
    return;
  }

  state_ = static_cast<SharedState*>(p);

  if (is_creator) {
    std::memset(state_, 0, sizeof(SharedState));
    new (state_) SharedState();
  }
  // Joiners: state_ already initialized by creator, just attach

  valid_ = true;
  LOG_INFO("SharedStateManager: %s shared state '%s' (%zu bytes)",
           is_creator ? "Created" : "Joined", name.c_str(), size);
}

SharedStateManager::~SharedStateManager() {
  if (!state_)
    return;
  if (multi_process_) {
    munmap(state_, sizeof(SharedState));
    if (is_creator_) {
      std::string name = "/pccl_" + unique_id_ + "_state";
      shm_unlink(name.c_str());
    }
  } else {
    delete state_;
  }
  state_ = nullptr;
  LOG_INFO("SharedStateManager: Released state");
}

bool SharedStateManager::barrier(int comm_id, int parties, const char* where, int rank, int epoch) {
  int comm = (comm_id >= 0 && comm_id < PCCL_MAX_CONCURRENT_COMMS) ? comm_id : 0;
  int expected = parties > 0 ? parties : world_size_;
  int slot = barrierSlotForWhere(where);
  uint64_t my_gen = state_->barrier_generation[comm][slot].load(std::memory_order_acquire);
  int arrived = state_->barrier_count[comm][slot].fetch_add(1, std::memory_order_acq_rel) + 1;
  if (arrived == expected) {
    state_->barrier_count[comm][slot].store(0, std::memory_order_relaxed);
    state_->barrier_generation[comm][slot].store(my_gen + 1, std::memory_order_release);
    return true;
  }

  const int timeout_ms = pcclTimeoutMsFromEnv("PCCL_BARRIER_TIMEOUT_MS", 120000);
  auto start = std::chrono::steady_clock::now();
  uint64_t spins = 0;
  while (state_->barrier_generation[comm][slot].load(std::memory_order_acquire) <= my_gen) {
    if ((++spins & 0x3fffff) == 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (elapsed_ms > timeout_ms) {
        LOG_ERROR("SharedStateManager: barrier timeout at %s comm=%d slot=%d rank=%d epoch=%d generation=%llu "
                  "arrived=%d/%d timeout_ms=%d",
                  where ? where : "unknown",
                  comm,
                  slot,
                  rank,
                  epoch,
                  static_cast<unsigned long long>(my_gen),
                  state_->barrier_count[comm][slot].load(std::memory_order_acquire),
                  expected,
                  timeout_ms);
        return false;
      }
      std::this_thread::yield();
    } else {
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
      _mm_pause();
#endif
    }
  }
  return true;
}

bool SharedStateManager::barrier(const char* where, int rank, int epoch) {
  return barrier(0, world_size_, where, rank, epoch);
}

void SharedStateManager::publishUserReadySeq(int comm_id, int rank, int slot_idx, uint64_t work_seq) {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX) {
    LOG_ERROR("SharedStateManager: invalid user-ready seq publish comm=%d rank=%d slot=%d work_seq=%llu",
              comm_id,
              rank,
              slot_idx,
              static_cast<unsigned long long>(work_seq));
    return;
  }

  auto& slot = state_->user_ready_seq[comm_id][rank][slot_idx];
  uint64_t old = slot.load(std::memory_order_relaxed);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG shared publish_user_ready_seq comm=%d rank=%d slot=%d work_seq=%llu old=%llu",
             comm_id,
             rank,
             slot_idx,
             static_cast<unsigned long long>(work_seq),
             static_cast<unsigned long long>(old));
  }
  slot.store(work_seq, std::memory_order_release);
}

uint64_t SharedStateManager::getUserReadySeq(int comm_id, int rank, int slot_idx) const {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX)
    return 0;
  uint64_t seq = state_->user_ready_seq[comm_id][rank][slot_idx].load(std::memory_order_acquire);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG shared read_user_ready_seq comm=%d rank=%d slot=%d work_seq=%llu",
             comm_id,
             rank,
             slot_idx,
             static_cast<unsigned long long>(seq));
  }
  return seq;
}

int SharedStateManager::userReadyCommSlot(uint64_t comm_hash) {
  if (comm_hash == 0)
    comm_hash = 1;
  for (int attempt = 0; attempt < PCCL_MAX_CONCURRENT_COMMS * 2; ++attempt) {
    int idx = static_cast<int>((comm_hash + static_cast<uint64_t>(attempt)) %
                               static_cast<uint64_t>(PCCL_MAX_CONCURRENT_COMMS));
    auto& lock = state_->user_ready_comm_lock[idx];
    lockUserReadyCommSlot(lock);
    uint64_t current = state_->user_ready_comm_hash[idx].load(std::memory_order_acquire);
    if (current == 0 || current == comm_hash) {
      if (current == 0) {
        state_->user_ready_comm_hash[idx].store(comm_hash, std::memory_order_relaxed);
        state_->user_ready_comm_refcount[idx].store(0, std::memory_order_relaxed);
      }
      state_->user_ready_comm_refcount[idx].fetch_add(1, std::memory_order_acq_rel);
      unlockUserReadyCommSlot(lock);
      return idx;
    }
    unlockUserReadyCommSlot(lock);
  }
  LOG_ERROR("SharedStateManager: exhausted user-ready communicator slots hash=%llu",
            static_cast<unsigned long long>(comm_hash));
  return -1;
}

bool SharedStateManager::releaseUserReadyCommSlot(int comm_id, uint64_t comm_hash) {
  if (comm_hash == 0)
    comm_hash = 1;
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS)
    return false;

  auto& lock = state_->user_ready_comm_lock[comm_id];
  lockUserReadyCommSlot(lock);
  const uint64_t current = state_->user_ready_comm_hash[comm_id].load(std::memory_order_acquire);
  if (current != comm_hash) {
    unlockUserReadyCommSlot(lock);
    LOG_WARN("SharedStateManager: refusing user-ready slot release comm=%d hash=%llu current=%llu",
             comm_id,
             static_cast<unsigned long long>(comm_hash),
             static_cast<unsigned long long>(current));
    return false;
  }

  const int refs = state_->user_ready_comm_refcount[comm_id].load(std::memory_order_relaxed);
  if (refs <= 0) {
    unlockUserReadyCommSlot(lock);
    LOG_WARN("SharedStateManager: user-ready slot comm=%d has no references for hash=%llu",
             comm_id, static_cast<unsigned long long>(comm_hash));
    return false;
  }

  const int remaining = state_->user_ready_comm_refcount[comm_id].fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (remaining == 0) {
    state_->user_ready_comm_refcount[comm_id].store(0, std::memory_order_relaxed);
    state_->user_ready_comm_hash[comm_id].store(0, std::memory_order_release);
  }
  unlockUserReadyCommSlot(lock);
  return true;
}

bool SharedStateManager::acquireHostStagingRegion(
    int comm_id, int parties, int rank, const size_t required[MAX_NUMA_NODES],
    const size_t capacity[MAX_NUMA_NODES], size_t offsets[MAX_NUMA_NODES]) {
  if (!state_ || comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || parties <= 0)
    return false;

  if (stagingLeaseDisabled()) {
    for (int numa = 0; numa < MAX_NUMA_NODES; ++numa) {
      offsets[numa] = 0;
    }
    return true;
  }

  const int expected = std::max(1, std::min(parties, MAX_RANKS));
  const int timeout_ms = pcclTimeoutMsFromEnv(
      "PCCL_HOST_STAGING_LEASE_TIMEOUT_MS", pcclTimeoutMsFromEnv("PCCL_BARRIER_TIMEOUT_MS", 120000));
  constexpr size_t kAlignment = 2ULL * 1024ULL * 1024ULL;
  auto align_up = [](size_t value) {
    return (value + kAlignment - 1) / kAlignment * kAlignment;
  };

  if (rank == 0) {
    const auto start = std::chrono::steady_clock::now();
    uint64_t spins = 0;
    while (true) {
      lockUserReadyCommSlot(state_->host_staging_state_lock);
      bool fits = true;
      size_t candidate[MAX_NUMA_NODES] = {};
      for (int numa = 0; numa < MAX_NUMA_NODES && fits; ++numa) {
        const size_t bytes = align_up(required[numa]);
        if (bytes == 0)
          continue;
        if (bytes > capacity[numa]) {
          fits = false;
          break;
        }

        size_t cursor = 0;
        while (true) {
          bool advanced = false;
          for (int other = 0; other < PCCL_MAX_CONCURRENT_COMMS; ++other) {
            if (other == comm_id)
              continue;
            const size_t other_bytes =
                state_->host_staging_bytes[other][numa].load(std::memory_order_acquire);
            if (other_bytes == 0)
              continue;
            const size_t other_offset =
                state_->host_staging_offset[other][numa].load(std::memory_order_acquire);
            if (cursor < other_offset + other_bytes && other_offset < cursor + bytes) {
              cursor = align_up(other_offset + other_bytes);
              advanced = true;
              break;
            }
          }
          if (!advanced)
            break;
        }
        if (cursor > capacity[numa] || bytes > capacity[numa] - cursor) {
          fits = false;
          break;
        }
        candidate[numa] = cursor;
      }

      if (fits) {
        for (int numa = 0; numa < MAX_NUMA_NODES; ++numa) {
          state_->host_staging_offset[comm_id][numa].store(candidate[numa],
                                                           std::memory_order_relaxed);
          state_->host_staging_bytes[comm_id][numa].store(align_up(required[numa]),
                                                          std::memory_order_release);
        }
        unlockUserReadyCommSlot(state_->host_staging_state_lock);
        break;
      }
      unlockUserReadyCommSlot(state_->host_staging_state_lock);

      if ((++spins & 0x3fffff) == 0) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
          LOG_ERROR("SharedStateManager: host staging region timeout comm=%d rank=%d",
                    comm_id, rank);
          return false;
        }
        std::this_thread::yield();
      }
    }
  }

  if (!barrier(comm_id, expected, "HostStagingRegion::acquire", rank)) {
    if (rank == 0) {
      lockUserReadyCommSlot(state_->host_staging_state_lock);
      for (int numa = 0; numa < MAX_NUMA_NODES; ++numa)
        state_->host_staging_bytes[comm_id][numa].store(0, std::memory_order_release);
      unlockUserReadyCommSlot(state_->host_staging_state_lock);
    }
    return false;
  }
  for (int numa = 0; numa < MAX_NUMA_NODES; ++numa)
    offsets[numa] =
        state_->host_staging_offset[comm_id][numa].load(std::memory_order_acquire);
  LOG_DEBUG("SharedStateManager: host staging region acquired comm=%d rank=%d parties=%d",
            comm_id, rank, expected);
  return true;
}

bool SharedStateManager::releaseHostStagingRegion(int comm_id, int parties, int rank) {
  if (!state_ || comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || parties <= 0)
    return false;

  if (stagingLeaseDisabled()) {
    return true;
  }

  const int expected = std::max(1, std::min(parties, MAX_RANKS));
  if (!barrier(comm_id, expected, "HostStagingRegion::release", rank))
    return false;

  if (rank == 0) {
    lockUserReadyCommSlot(state_->host_staging_state_lock);
    for (int numa = 0; numa < MAX_NUMA_NODES; ++numa)
      state_->host_staging_bytes[comm_id][numa].store(0, std::memory_order_release);
    unlockUserReadyCommSlot(state_->host_staging_state_lock);
  }
  LOG_DEBUG("SharedStateManager: host staging region released comm=%d rank=%d", comm_id, rank);
  return true;
}

void SharedStateManager::resetEpoch(int epoch) {
  resetEpoch(0, epoch);
}

void SharedStateManager::resetEpoch(int comm_id, int epoch) {
  int comm = commIndex(comm_id);
  int ep = epoch & 1;
  for (int n = 0; n < MAX_NUMA_NODES; n++) {
    for (int c = 0; c < MAX_CHUNKS; c++) {
      auto& cs = state_->numa_nodes[comm][n].chunks[c];
      for (int v = 0; v < MAX_PROGRESS_VERSIONS; ++v) {
        cs.write_progress[v][ep].store(0, std::memory_order_relaxed);
        // Double-buffer consume-side reset
        cs.consume_progress[v][ep].store(0, std::memory_order_relaxed);
        for (int r = 0; r < MAX_PROGRESS_READERS; r++)
          cs.reader_done[r][v][ep].store(0, std::memory_order_relaxed);
      }
      // Note: num_readers is NOT reset — it is stable per plan lifetime
    }
  }
  for (int s = 0; s < DeviceProgressSpaceCount; s++) {
    for (int r = 0; r < MAX_RANKS; r++) {
      for (int c = 0; c < MAX_CHUNKS; c++) {
        auto& cs = state_->device_spaces[comm][s][r].chunks[c];
        for (int v = 0; v < MAX_PROGRESS_VERSIONS; ++v) {
          cs.write_progress[v][ep].store(0, std::memory_order_relaxed);
          cs.consume_progress[v][ep].store(0, std::memory_order_relaxed);
          for (int reader = 0; reader < MAX_PROGRESS_READERS; reader++)
            cs.reader_done[reader][v][ep].store(0, std::memory_order_relaxed);
        }
      }
    }
  }
  std::atomic_thread_fence(std::memory_order_release);
}

void SharedStateManager::resetEpochKeys(int epoch, const std::vector<ProgressKey>& keys) {
  resetEpochKeys(0, epoch, keys);
}

void SharedStateManager::resetEpochKeys(int comm_id, int epoch, const std::vector<ProgressKey>& keys) {
  int ep = epoch & 1;
  for (const auto& key : keys) {
    if (key.owner < 0 || key.chunk < 0 || key.chunk >= MAX_CHUNKS)
      continue;
    if (key.space == ProgressSpace::HOST) {
      if (key.owner >= MAX_NUMA_NODES)
        continue;
    } else if (key.owner >= MAX_RANKS) {
      continue;
    }
    auto& cs = chunkState(comm_id, key);
    int version = progressVersionIndex(key.version);
    uint64_t old_write = cs.write_progress[version][ep].load(std::memory_order_relaxed);
    uint64_t old_consume = cs.consume_progress[version][ep].load(std::memory_order_relaxed);
    uint64_t old_reader_done[MAX_PROGRESS_READERS] = {};
    bool log_key = debugProgressKeySelected(key);
    if (log_key) {
      for (int reader = 0; reader < MAX_PROGRESS_READERS; ++reader)
        old_reader_done[reader] = cs.reader_done[reader][version][ep].load(std::memory_order_relaxed);
    }
    cs.write_progress[version][ep].store(0, std::memory_order_relaxed);
    cs.consume_progress[version][ep].store(0, std::memory_order_relaxed);
    for (int reader = 0; reader < MAX_PROGRESS_READERS; ++reader)
      cs.reader_done[reader][version][ep].store(0, std::memory_order_relaxed);
    if (log_key) {
      LOG_INFO("PROGDBG shared reset_key epoch=%d ep=%d key=(space=%s/%d owner=%d chunk=%d version=%d idx=%d) "
               "old_write=%llu old_consume=%llu readers=%d rd0=%llu rd1=%llu rd2=%llu rd3=%llu",
               epoch,
               ep,
               progressSpaceDebugName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version,
               version,
               static_cast<unsigned long long>(old_write),
               static_cast<unsigned long long>(old_consume),
               cs.num_readers[version],
               static_cast<unsigned long long>(old_reader_done[0]),
               static_cast<unsigned long long>(old_reader_done[1]),
               static_cast<unsigned long long>(old_reader_done[2]),
               static_cast<unsigned long long>(old_reader_done[3]));
    }
  }
  std::atomic_thread_fence(std::memory_order_release);
}

}  // namespace pccl
