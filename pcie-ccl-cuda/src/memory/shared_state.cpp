#include "shared_state.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
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
  if (std::strstr(where, "SmallMessageExecutor::execute"))
    return 3;
  if (std::strstr(where, "GraphExecutor::start"))
    return 3;
  return 0;
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

}  // namespace

// ---- Single-process constructor (heap allocation) ----

SharedStateManager::SharedStateManager(int world_size) : world_size_(world_size) {
  // In-process allocation (no SHM)
  state_ = new SharedState();
  std::memset(state_, 0, sizeof(SharedState));
  new (state_) SharedState();
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
    for (int attempt = 0; attempt < 10000; attempt++) {
      fd = shm_open(name.c_str(), O_RDWR, 0666);
      if (fd >= 0) break;
      usleep(1000);
    }
  }

  if (fd < 0) {
    LOG_ERROR("SharedStateManager: shm_open(%s) failed: %s", name.c_str(), strerror(errno));
    state_ = new SharedState();
    std::memset(state_, 0, sizeof(SharedState));
    new (state_) SharedState();
    multi_process_ = false;  // fall back to heap
    return;
  }

  if (is_creator) {
    if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
      LOG_ERROR("SharedStateManager: ftruncate failed: %s", strerror(errno));
      close(fd);
      state_ = new SharedState();
      std::memset(state_, 0, sizeof(SharedState));
      new (state_) SharedState();
      multi_process_ = false;
      return;
    }
  }

  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);

  if (p == MAP_FAILED) {
    LOG_ERROR("SharedStateManager: mmap failed: %s", strerror(errno));
    state_ = new SharedState();
    std::memset(state_, 0, sizeof(SharedState));
    new (state_) SharedState();
    multi_process_ = false;
    return;
  }

  state_ = static_cast<SharedState*>(p);

  if (is_creator) {
    std::memset(state_, 0, sizeof(SharedState));
    new (state_) SharedState();
  }
  // Joiners: state_ already initialized by creator, just attach

  LOG_INFO("SharedStateManager: %s shared state '%s' (%zu bytes)",
           is_creator ? "Created" : "Joined", name.c_str(), size);
}

SharedStateManager::~SharedStateManager() {
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
    }
  }
  return true;
}

bool SharedStateManager::barrier(const char* where, int rank, int epoch) {
  return barrier(0, world_size_, where, rank, epoch);
}

void SharedStateManager::publishDeviceIpcBuffer(int comm_id, int kind, int rank, uint64_t base_addr,
                                                uint64_t offset, uint64_t size, const void* handle,
                                                size_t handle_size) {
  int comm = commIndex(comm_id);
  if (kind < 0 || kind >= DEVICE_IPC_BUFFER_KINDS || rank < 0 || rank >= MAX_RANKS || !handle ||
      handle_size > DEVICE_IPC_HANDLE_BYTES) {
    LOG_ERROR("SharedStateManager: invalid device IPC publish comm=%d kind=%d rank=%d handle_size=%zu",
              comm_id, kind, rank, handle_size);
    return;
  }

  auto& slot = state_->device_ipc[comm][kind][rank];
  uint64_t seq = slot.seq.load(std::memory_order_relaxed);
  if ((seq & 1) != 0)
    seq++;
  slot.seq.store(seq + 1, std::memory_order_release);
  slot.base_addr.store(base_addr, std::memory_order_relaxed);
  slot.offset.store(offset, std::memory_order_relaxed);
  slot.size.store(size, std::memory_order_relaxed);
  std::memset(slot.handle, 0, DEVICE_IPC_HANDLE_BYTES);
  std::memcpy(slot.handle, handle, handle_size);
  std::atomic_thread_fence(std::memory_order_release);
  slot.seq.store(seq + 2, std::memory_order_release);
}

bool SharedStateManager::readDeviceIpcBuffer(int comm_id, int kind, int rank, DeviceIpcBufferRecord* out) const {
  int comm = commIndex(comm_id);
  if (!out || kind < 0 || kind >= DEVICE_IPC_BUFFER_KINDS || rank < 0 || rank >= MAX_RANKS) {
    return false;
  }

  const auto& slot = state_->device_ipc[comm][kind][rank];
  for (int attempt = 0; attempt < 1000; ++attempt) {
    uint64_t seq1 = slot.seq.load(std::memory_order_acquire);
    if (seq1 == 0 || (seq1 & 1) != 0) {
      std::this_thread::yield();
      continue;
    }
    DeviceIpcBufferRecord tmp;
    tmp.seq = seq1;
    tmp.base_addr = slot.base_addr.load(std::memory_order_relaxed);
    tmp.offset = slot.offset.load(std::memory_order_relaxed);
    tmp.size = slot.size.load(std::memory_order_relaxed);
    std::memcpy(tmp.handle, slot.handle, DEVICE_IPC_HANDLE_BYTES);
    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t seq2 = slot.seq.load(std::memory_order_acquire);
    if (seq1 == seq2 && (seq2 & 1) == 0) {
      *out = tmp;
      return tmp.base_addr != 0 && tmp.size != 0;
    }
  }
  return false;
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

void SharedStateManager::publishFastpathReadySeq(int comm_id, int rank, int slot_idx, uint64_t work_seq) {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX) {
    LOG_ERROR("SharedStateManager: invalid fastpath-ready seq publish comm=%d rank=%d slot=%d work_seq=%llu",
              comm_id,
              rank,
              slot_idx,
              static_cast<unsigned long long>(work_seq));
    return;
  }

  auto& slot = state_->fastpath_ready_seq[comm_id][rank][slot_idx];
  uint64_t old = slot.load(std::memory_order_relaxed);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG shared publish_fastpath_ready_seq comm=%d rank=%d slot=%d work_seq=%llu old=%llu",
             comm_id,
             rank,
             slot_idx,
             static_cast<unsigned long long>(work_seq),
             static_cast<unsigned long long>(old));
  }
  slot.store(work_seq, std::memory_order_release);
}

uint64_t SharedStateManager::getFastpathReadySeq(int comm_id, int rank, int slot_idx) const {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX)
    return 0;
  uint64_t seq = state_->fastpath_ready_seq[comm_id][rank][slot_idx].load(std::memory_order_acquire);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG shared read_fastpath_ready_seq comm=%d rank=%d slot=%d work_seq=%llu",
             comm_id,
             rank,
             slot_idx,
             static_cast<unsigned long long>(seq));
  }
  return seq;
}

void SharedStateManager::publishFastpathStageSeq(int comm_id, int rank, int slot_idx, uint64_t work_seq) {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX) {
    LOG_ERROR("SharedStateManager: invalid fastpath-stage seq publish comm=%d rank=%d slot=%d work_seq=%llu",
              comm_id,
              rank,
              slot_idx,
              static_cast<unsigned long long>(work_seq));
    return;
  }

  auto& slot = state_->fastpath_stage_seq[comm_id][rank][slot_idx];
  uint64_t old = slot.load(std::memory_order_relaxed);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG shared publish_fastpath_stage_seq comm=%d rank=%d slot=%d work_seq=%llu old=%llu",
             comm_id,
             rank,
             slot_idx,
             static_cast<unsigned long long>(work_seq),
             static_cast<unsigned long long>(old));
  }
  slot.store(work_seq, std::memory_order_release);
}

uint64_t SharedStateManager::getFastpathStageSeq(int comm_id, int rank, int slot_idx) const {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX)
    return 0;
  uint64_t seq = state_->fastpath_stage_seq[comm_id][rank][slot_idx].load(std::memory_order_acquire);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG shared read_fastpath_stage_seq comm=%d rank=%d slot=%d work_seq=%llu",
             comm_id,
             rank,
             slot_idx,
             static_cast<unsigned long long>(seq));
  }
  return seq;
}

void SharedStateManager::publishFastpathDirectStatus(int comm_id, int rank, int slot_idx, uint64_t work_seq,
                                                     bool all_direct) {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX) {
    LOG_ERROR("SharedStateManager: invalid fastpath direct status publish comm=%d rank=%d slot=%d work_seq=%llu",
              comm_id,
              rank,
              slot_idx,
              static_cast<unsigned long long>(work_seq));
    return;
  }
  auto& ok_slot = state_->fastpath_direct_ok_seq[comm_id][rank][slot_idx];
  auto& missing_slot = state_->fastpath_direct_missing_seq[comm_id][rank][slot_idx];
  if (all_direct) {
    missing_slot.store(0, std::memory_order_release);
    ok_slot.store(work_seq, std::memory_order_release);
  } else {
    ok_slot.store(0, std::memory_order_release);
    missing_slot.store(work_seq, std::memory_order_release);
  }
}

int SharedStateManager::getFastpathDirectStatus(int comm_id, int rank, int slot_idx, uint64_t work_seq) const {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS ||
      slot_idx < 0 || slot_idx >= PCCL_EXECUTOR_QUEUE_DEPTH_MAX)
    return -1;
  if (state_->fastpath_direct_ok_seq[comm_id][rank][slot_idx].load(std::memory_order_acquire) == work_seq)
    return 1;
  if (state_->fastpath_direct_missing_seq[comm_id][rank][slot_idx].load(std::memory_order_acquire) == work_seq)
    return 0;
  return -1;
}

void SharedStateManager::publishFastpathStatus(int comm_id, int rank, bool available, uint64_t threshold_bytes) {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS) {
    LOG_ERROR("SharedStateManager: invalid fastpath status publish comm=%d rank=%d", comm_id, rank);
    return;
  }
  state_->fastpath_threshold_bytes[comm_id][rank].store(threshold_bytes, std::memory_order_release);
  state_->fastpath_available[comm_id][rank].store(available ? 1 : 0, std::memory_order_release);
}

bool SharedStateManager::isFastpathRankAvailable(int comm_id, int rank) const {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS)
    return false;
  return state_->fastpath_available[comm_id][rank].load(std::memory_order_acquire) != 0;
}

uint64_t SharedStateManager::getFastpathThresholdBytes(int comm_id, int rank) const {
  if (comm_id < 0 || comm_id >= PCCL_MAX_CONCURRENT_COMMS || rank < 0 || rank >= MAX_RANKS)
    return 0;
  return state_->fastpath_threshold_bytes[comm_id][rank].load(std::memory_order_acquire);
}

int SharedStateManager::userReadyCommSlot(uint64_t comm_hash) {
  if (comm_hash == 0)
    comm_hash = 1;
  for (int attempt = 0; attempt < PCCL_MAX_CONCURRENT_COMMS * 2; ++attempt) {
    int idx = static_cast<int>((comm_hash + static_cast<uint64_t>(attempt)) %
                               static_cast<uint64_t>(PCCL_MAX_CONCURRENT_COMMS));
    uint64_t empty = 0;
    if (state_->user_ready_comm_hash[idx].compare_exchange_strong(empty, comm_hash, std::memory_order_acq_rel,
                                                                  std::memory_order_acquire) ||
        empty == comm_hash) {
      return idx;
    }
  }
  LOG_ERROR("SharedStateManager: exhausted user-ready communicator slots hash=%llu",
            static_cast<unsigned long long>(comm_hash));
  return -1;
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
