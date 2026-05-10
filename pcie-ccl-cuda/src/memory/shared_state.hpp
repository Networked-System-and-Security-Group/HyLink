#ifndef PCCL_SHARED_STATE_HPP
#define PCCL_SHARED_STATE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../core/limits.hpp"
#include "../hal/device_rt.hpp"
#include "../ir/ir.hpp"
#include "../util/debug_flags.hpp"
#include "../util/log.hpp"

namespace pccl {

constexpr int DEVICE_IPC_BUFFER_KINDS = 3;   // input, output, internal
constexpr int DEVICE_IPC_HANDLE_BYTES = 64;  // cudaIpcMemHandle_t size
constexpr int SHARED_BARRIER_SLOTS = 4;

struct alignas(64) ChunkState {
  // Write side: how many subchunks have been produced.
  std::atomic<uint64_t> write_progress[MAX_PROGRESS_VERSIONS][2]{};  // indexed by version, epoch & 1

  // Double-buffer flow control: consume_progress tracks the minimum subchunk
  // index that ALL readers have finished.  Writer of sc_k waits until
  // consume_progress >= k - 1 (i.e. the slot about to be overwritten is free).
  // Monotonically increasing — no reset races.
  std::atomic<uint64_t> consume_progress[MAX_PROGRESS_VERSIONS][2]{};
  // Per-reader cursor: reader i has consumed up to this many subchunks.
  // consume_progress = min(all reader_done values).
  std::atomic<uint64_t> reader_done[MAX_PROGRESS_READERS][MAX_PROGRESS_VERSIONS][2]{};  // [reader_id][version][epoch]
  // Number of readers for this chunk version. 0 = no readers.
  int num_readers[MAX_PROGRESS_VERSIONS] = {};
};

struct alignas(64) NumaNodeState {
  ChunkState chunks[MAX_CHUNKS];
};

struct DeviceIpcBufferRecord {
  uint64_t seq = 0;
  uint64_t base_addr = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
  char handle[DEVICE_IPC_HANDLE_BYTES] = {};
};

struct alignas(64) DeviceIpcBufferSlot {
  // Seqlock: odd while writer updates handle fields, even when stable.
  std::atomic<uint64_t> seq{0};
  std::atomic<uint64_t> base_addr{0};
  std::atomic<uint64_t> offset{0};
  std::atomic<uint64_t> size{0};
  char handle[DEVICE_IPC_HANDLE_BYTES]{};
};

struct alignas(64) SharedState {
  std::atomic<int> initialized_count{0};
  std::atomic<int> transfer_complete_count{0};
  // Per-communicator barrier state. Keyed by comm_id so that concurrent
  // communicators (e.g. Megatron's TP / DP / embedding groups) do not collide
  // on the same barrier slots, which would corrupt arrival counts and hang.
  std::atomic<int> barrier_count[PCCL_MAX_CONCURRENT_COMMS][SHARED_BARRIER_SLOTS]{};
  std::atomic<uint64_t> barrier_generation[PCCL_MAX_CONCURRENT_COMMS][SHARED_BARRIER_SLOTS]{};
  std::atomic<uint64_t> user_ready_comm_hash[PCCL_MAX_CONCURRENT_COMMS]{};
  DeviceIpcBufferSlot device_ipc[PCCL_MAX_CONCURRENT_COMMS][DEVICE_IPC_BUFFER_KINDS][MAX_RANKS];
  std::atomic<uint64_t> user_ready_seq[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS][PCCL_EXECUTOR_QUEUE_DEPTH_MAX]{};
  std::atomic<uint64_t> fastpath_ready_seq[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS][PCCL_EXECUTOR_QUEUE_DEPTH_MAX]{};
  std::atomic<uint64_t> fastpath_stage_seq[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS][PCCL_EXECUTOR_QUEUE_DEPTH_MAX]{};
  std::atomic<uint64_t> fastpath_direct_ok_seq[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS][PCCL_EXECUTOR_QUEUE_DEPTH_MAX]{};
  std::atomic<uint64_t> fastpath_direct_missing_seq[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS][PCCL_EXECUTOR_QUEUE_DEPTH_MAX]{};
  std::atomic<int> fastpath_available[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS]{};
  std::atomic<uint64_t> fastpath_threshold_bytes[PCCL_MAX_CONCURRENT_COMMS][MAX_RANKS]{};
  // Per-communicator progress state. Keyed by comm_id so concurrent
  // communicators do not collide / reset each other's cross-rank progress keys.
  NumaNodeState numa_nodes[PCCL_MAX_CONCURRENT_COMMS][MAX_NUMA_NODES];
  NumaNodeState device_spaces[PCCL_MAX_CONCURRENT_COMMS][DeviceProgressSpaceCount][MAX_RANKS];
};

inline const char* progressSpaceDebugName(ProgressSpace space) {
  switch (space) {
    case ProgressSpace::HOST:
      return "HOST";
    case ProgressSpace::INPUT:
      return "INPUT";
    case ProgressSpace::OUTPUT:
      return "OUTPUT";
    case ProgressSpace::SCRATCH:
      return "SCRATCH";
    case ProgressSpace::INTERNAL:
      return "INTERNAL";
  }
  return "Unknown";
}

inline bool debugProgressKeySelected(const ProgressKey& key) {
  if (!debugProgressKeysEnabled())
    return false;
  if (debugProgressHostOnly() && key.space != ProgressSpace::HOST)
    return false;
  return debugProgressChunkSelected(key.chunk);
}

class SharedStateManager {
 public:
  // Single-process mode: heap-allocated SharedState
  SharedStateManager(int world_size);

  // Multi-process mode: SharedState in POSIX shared memory
  SharedStateManager(int world_size, const std::string& unique_id, bool is_creator);

  ~SharedStateManager();

  SharedStateManager(const SharedStateManager&) = delete;
  SharedStateManager& operator=(const SharedStateManager&) = delete;

  // ── Write-progress accessors (existing, renamed from progress → write_progress) ──

  uint64_t getChunkProgress(int numa_node, int chunk_idx, int epoch) const {
    return getWriteProgress(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, epoch);
  }

  uint64_t getConsumeProgress(int numa_node, int chunk_idx, int epoch) const {
    return getConsumeProgress(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, epoch);
  }

  int getChunkNumReaders(int numa_node, int chunk_idx) const {
    return getNumReaders(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0});
  }

  void setChunkProgress(int numa_node, int chunk_idx, uint64_t value, int epoch) {
    setWriteProgress(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, value, epoch);
  }

  void advanceChunkProgress(int numa_node, int chunk_idx, uint64_t new_progress, int epoch) {
    advanceWriteProgress(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, new_progress, epoch);
  }

  uint64_t getDeviceProgress(DeviceBufferKind buf, int owner, int chunk_idx, int epoch) const {
    return getWriteProgress(ProgressKey{progressSpaceForDeviceBuffer(buf), owner, chunk_idx, 0}, epoch);
  }

  void setDeviceProgress(DeviceBufferKind buf, int owner, int chunk_idx, uint64_t value, int epoch) {
    setWriteProgress(ProgressKey{progressSpaceForDeviceBuffer(buf), owner, chunk_idx, 0}, value, epoch);
  }

  void advanceDeviceProgress(DeviceBufferKind buf, int owner, int chunk_idx, uint64_t new_progress, int epoch) {
    advanceWriteProgress(ProgressKey{progressSpaceForDeviceBuffer(buf), owner, chunk_idx, 0}, new_progress, epoch);
  }

  uint64_t getWriteProgress(const ProgressKey& key, int epoch) const { return getWriteProgress(0, key, epoch); }
  uint64_t getWriteProgress(int comm_id, const ProgressKey& key, int epoch) const {
    const auto& cs = chunkState(comm_id, key);
    return cs.write_progress[progressVersionIndex(key.version)][epoch & 1].load(std::memory_order_acquire);
  }

  uint64_t getConsumeProgress(const ProgressKey& key, int epoch) const { return getConsumeProgress(0, key, epoch); }
  uint64_t getConsumeProgress(int comm_id, const ProgressKey& key, int epoch) const {
    const auto& cs = chunkState(comm_id, key);
    return cs.consume_progress[progressVersionIndex(key.version)][epoch & 1].load(std::memory_order_acquire);
  }

  int getNumReaders(const ProgressKey& key) const { return getNumReaders(0, key); }
  int getNumReaders(int comm_id, const ProgressKey& key) const {
    const auto& cs = chunkState(comm_id, key);
    return cs.num_readers[progressVersionIndex(key.version)];
  }

  void setWriteProgress(const ProgressKey& key, uint64_t value, int epoch) {
    setWriteProgress(0, key, value, epoch);
  }
  void setWriteProgress(int comm_id, const ProgressKey& key, uint64_t value, int epoch) {
    auto& cs = chunkState(comm_id, key);
    int version = progressVersionIndex(key.version);
    int ep = epoch & 1;
    auto& p = cs.write_progress[version][ep];
    uint64_t old = p.load(std::memory_order_relaxed);
    p.store(value, std::memory_order_release);
    if (debugProgressKeySelected(key)) {
      LOG_INFO("PROGDBG shared set_write epoch=%d ep=%d key=(space=%s/%d owner=%d chunk=%d version=%d idx=%d) "
               "old=%llu new=%llu readers=%d consume=%llu",
               epoch,
               ep,
               progressSpaceDebugName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version,
               version,
               static_cast<unsigned long long>(old),
               static_cast<unsigned long long>(value),
               cs.num_readers[version],
               static_cast<unsigned long long>(cs.consume_progress[version][ep].load(std::memory_order_acquire)));
    }
  }

  void advanceWriteProgress(const ProgressKey& key, uint64_t new_progress, int epoch) {
    advanceWriteProgress(0, key, new_progress, epoch);
  }
  void advanceWriteProgress(int comm_id, const ProgressKey& key, uint64_t new_progress, int epoch) {
    auto& cs = chunkState(comm_id, key);
    int version = progressVersionIndex(key.version);
    int ep = epoch & 1;
    auto& p = cs.write_progress[version][ep];
    uint64_t old = p.load(std::memory_order_relaxed);
    uint64_t observed_old = old;
    bool advanced = false;
    while (new_progress > old) {
      if (p.compare_exchange_weak(old, new_progress, std::memory_order_release, std::memory_order_relaxed)) {
        advanced = true;
        break;
      }
    }
    if (debugProgressKeySelected(key)) {
      LOG_INFO("PROGDBG shared advance_write epoch=%d ep=%d key=(space=%s/%d owner=%d chunk=%d version=%d idx=%d) "
               "old=%llu observed_after=%llu requested=%llu advanced=%d readers=%d consume=%llu",
               epoch,
               ep,
               progressSpaceDebugName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version,
               version,
               static_cast<unsigned long long>(observed_old),
               static_cast<unsigned long long>(p.load(std::memory_order_acquire)),
               static_cast<unsigned long long>(new_progress),
               advanced ? 1 : 0,
               cs.num_readers[version],
               static_cast<unsigned long long>(cs.consume_progress[version][ep].load(std::memory_order_acquire)));
    }
  }

  // ── Double-buffer flow control (monotonic counters — no reset races) ──

  void setChunkNumReaders(int numa_node, int chunk_idx, int count) {
    setNumReaders(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, count);
  }

  void setNumReaders(const ProgressKey& key, int count) { setNumReaders(0, key, count); }
  void setNumReaders(int comm_id, const ProgressKey& key, int count) {
    auto& cs = chunkState(comm_id, key);
    int version = progressVersionIndex(key.version);
    int old = cs.num_readers[version];
    cs.num_readers[version] = count;
    if (debugProgressKeySelected(key)) {
      LOG_INFO("PROGDBG shared set_readers key=(space=%s/%d owner=%d chunk=%d version=%d idx=%d) old=%d new=%d",
               progressSpaceDebugName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version,
               version,
               old,
               count);
    }
  }

  // Called by each reader task's barrier last-thread after consuming sc_idx.
  // reader_id is unique per-chunk, assigned at compile time.
  void advanceReaderDone(int numa_node, int chunk_idx, int reader_id, uint64_t new_val, int epoch) {
    advanceReaderDone(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, reader_id, new_val, epoch);
  }

  void advanceReaderDone(const ProgressKey& key, int reader_id, uint64_t new_val, int epoch) {
    advanceReaderDone(0, key, reader_id, new_val, epoch);
  }
  void advanceReaderDone(int comm_id, const ProgressKey& key, int reader_id, uint64_t new_val, int epoch) {
    if (reader_id < 0 || reader_id >= MAX_PROGRESS_READERS)
      return;
    auto& cs = chunkState(comm_id, key);
    int version = progressVersionIndex(key.version);
    int ep = epoch & 1;
    auto& rd = cs.reader_done[reader_id][version][ep];
    uint64_t old_rd = rd.load(std::memory_order_relaxed);
    uint64_t observed_old_rd = old_rd;
    bool reader_advanced = false;
    while (new_val > old_rd) {
      if (rd.compare_exchange_weak(old_rd, new_val, std::memory_order_release, std::memory_order_relaxed)) {
        reader_advanced = true;
        break;
      }
    }
    // Compute min across all readers, CAS-advance consume_progress
    uint64_t min_rd = new_val;
    for (int i = 0; i < cs.num_readers[version]; i++) {
      uint64_t v = cs.reader_done[i][version][ep].load(std::memory_order_acquire);
      if (v < min_rd) min_rd = v;
    }
    auto& cp = cs.consume_progress[version][ep];
    uint64_t old_cp = cp.load(std::memory_order_relaxed);
    uint64_t observed_old_cp = old_cp;
    bool consume_advanced = false;
    while (min_rd > old_cp) {
      if (cp.compare_exchange_weak(old_cp, min_rd, std::memory_order_release, std::memory_order_relaxed)) {
        consume_advanced = true;
        break;
      }
    }
    if (debugProgressKeySelected(key)) {
      LOG_INFO("PROGDBG shared reader_done epoch=%d ep=%d key=(space=%s/%d owner=%d chunk=%d version=%d idx=%d) "
               "reader=%d readers=%d rd_old=%llu rd_after=%llu requested=%llu rd_advanced=%d min_rd=%llu "
               "consume_old=%llu consume_after=%llu consume_advanced=%d",
               epoch,
               ep,
               progressSpaceDebugName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version,
               version,
               reader_id,
               cs.num_readers[version],
               static_cast<unsigned long long>(observed_old_rd),
               static_cast<unsigned long long>(rd.load(std::memory_order_acquire)),
               static_cast<unsigned long long>(new_val),
               reader_advanced ? 1 : 0,
               static_cast<unsigned long long>(min_rd),
               static_cast<unsigned long long>(observed_old_cp),
               static_cast<unsigned long long>(cp.load(std::memory_order_acquire)),
               consume_advanced ? 1 : 0);
    }
  }

  // Writer waits until consume_progress confirms the slot is safe to overwrite.
  // For sc_idx with DOUBLE_BUFFER_SLOTS=2, slot sc_idx&1 was last written at
  // sc_idx-2, so readers must have consumed at least sc_idx-2+1 = sc_idx-1.
  void waitSlotFree(int numa_node, int chunk_idx, int sc_idx, int epoch) const {
    waitSlotFree(ProgressKey{ProgressSpace::HOST, numa_node, chunk_idx, 0}, sc_idx, epoch);
  }

  void waitSlotFree(const ProgressKey& key, int sc_idx, int epoch) const { waitSlotFree(0, key, sc_idx, epoch); }
  void waitSlotFree(int comm_id, const ProgressKey& key, int sc_idx, int epoch) const {
    const auto& cs = chunkState(comm_id, key);
    int version = progressVersionIndex(key.version);
    int ep = epoch & 1;
    uint64_t needed = static_cast<uint64_t>(sc_idx) - DOUBLE_BUFFER_SLOTS + 1;
    while (cs.consume_progress[version][ep].load(std::memory_order_acquire) < needed) {}
  }

  // Transfer completion
  int incrementTransferCompleteCount() {
    return state_->transfer_complete_count.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  void resetTransferCompleteCount() { state_->transfer_complete_count.store(0, std::memory_order_release); }

  // Sense-reversing barrier. Returns false on diagnostic timeout.
  // Per-communicator variant: comm_id selects an independent barrier instance,
  // parties is the number of ranks in that communicator (defaults to world_size).
  bool barrier(int comm_id, int parties, const char* where = nullptr, int rank = -1, int epoch = -1);
  // Process-global barrier (comm 0, all ranks) — for init/teardown.
  bool barrier(const char* where = nullptr, int rank = -1, int epoch = -1);

  void publishDeviceIpcBuffer(int comm_id, int kind, int rank, uint64_t base_addr, uint64_t offset, uint64_t size,
                              const void* handle, size_t handle_size);
  bool readDeviceIpcBuffer(int comm_id, int kind, int rank, DeviceIpcBufferRecord* out) const;
  void publishUserReadySeq(int comm_id, int rank, int slot, uint64_t work_seq);
  uint64_t getUserReadySeq(int comm_id, int rank, int slot) const;
  void publishFastpathReadySeq(int comm_id, int rank, int slot, uint64_t work_seq);
  uint64_t getFastpathReadySeq(int comm_id, int rank, int slot) const;
  void publishFastpathStageSeq(int comm_id, int rank, int slot, uint64_t work_seq);
  uint64_t getFastpathStageSeq(int comm_id, int rank, int slot) const;
  void publishFastpathDirectStatus(int comm_id, int rank, int slot, uint64_t work_seq, bool all_direct);
  int getFastpathDirectStatus(int comm_id, int rank, int slot, uint64_t work_seq) const;
  void publishFastpathStatus(int comm_id, int rank, bool available, uint64_t threshold_bytes);
  bool isFastpathRankAvailable(int comm_id, int rank) const;
  uint64_t getFastpathThresholdBytes(int comm_id, int rank) const;
  bool isMultiProcess() const { return multi_process_; }

  // Reset a specific epoch's chunk state
  void resetEpoch(int epoch);
  void resetEpoch(int comm_id, int epoch);
  void resetEpochKeys(int epoch, const std::vector<ProgressKey>& keys);
  void resetEpochKeys(int comm_id, int epoch, const std::vector<ProgressKey>& keys);

  int userReadyCommSlot(uint64_t comm_hash);

  SharedState* getRawState() { return state_; }

 private:
  static int progressVersionIndex(int version) {
    return (version < 0) ? 0 : (version % MAX_PROGRESS_VERSIONS);
  }

  static int commIndex(int comm_id) {
    return (comm_id >= 0 && comm_id < PCCL_MAX_CONCURRENT_COMMS) ? comm_id : 0;
  }

  ChunkState& chunkState(const ProgressKey& key) { return chunkState(0, key); }
  const ChunkState& chunkState(const ProgressKey& key) const { return chunkState(0, key); }

  ChunkState& chunkState(int comm_id, const ProgressKey& key) {
    int comm = commIndex(comm_id);
    if (key.space == ProgressSpace::HOST)
      return state_->numa_nodes[comm][key.owner].chunks[key.chunk];
    return state_->device_spaces[comm][deviceProgressSpaceIndex(key.space)][key.owner].chunks[key.chunk];
  }

  const ChunkState& chunkState(int comm_id, const ProgressKey& key) const {
    int comm = commIndex(comm_id);
    if (key.space == ProgressSpace::HOST)
      return state_->numa_nodes[comm][key.owner].chunks[key.chunk];
    return state_->device_spaces[comm][deviceProgressSpaceIndex(key.space)][key.owner].chunks[key.chunk];
  }

  SharedState* state_;
  int world_size_;
  bool multi_process_ = false;
  bool is_creator_ = false;
  std::string unique_id_;
};

}  // namespace pccl

#endif /* PCCL_SHARED_STATE_HPP */
