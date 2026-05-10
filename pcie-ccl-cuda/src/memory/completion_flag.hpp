#ifndef PCCL_COMPLETION_FLAG_HPP
#define PCCL_COMPLETION_FLAG_HPP

#include <cstdint>
#include <cstddef>
#include <string>

#include "../core/limits.hpp"
#include "../hal/device_rt.hpp"

namespace pccl {

enum class CollectiveDebugOp {
  AllReduce,
  AllGather,
  ReduceScatter,
  Broadcast,
  Unknown,
};

class CompletionFlag {
 public:
  CompletionFlag();
  ~CompletionFlag();

  CompletionFlag(const CompletionFlag&) = delete;
  CompletionFlag& operator=(const CompletionFlag&) = delete;

  int initialize(bool multi_process = false);

  int queueDepth() const { return queue_depth_; }

  uint64_t readySequence(int slot) const;
  bool isReady(int slot, uint64_t work_seq) const;

  int prepareUserWait(int slot, devStream user_stream, CollectiveDebugOp op, int rank, uint64_t seq,
                      size_t count, size_t dtype, const void* comm, const void* stream);

  // Signal completion: uses dedicated signal_stream_ to avoid deadlock.
  int signalComplete(int slot, uint64_t seq);
  int signalComplete(int slot);

  // Compatibility wrapper for non-queued callers.
  int enqueueUserWait(devStream user_stream);

  // Compatibility wrappers for non-queued callers.
  int signalComplete();
  int reset();
  int ensureResetVisible(devStream user_stream);
  int recordWaitComplete(devStream user_stream);

 private:
  struct Slot {
    uint32_t* dev_flag = nullptr;
    uint64_t* ready_seq_host = nullptr;
    uint64_t* ready_seq_dev = nullptr;
    // complete_seq is host-mapped: the executor (CPU) writes the completion
    // sequence directly to complete_seq_host, and the user stream's
    // cuStreamWaitValue64 polls complete_seq_dev (the device-side mapping).
    // This avoids head-of-line deadlock: the completion signal needs no GPU
    // hardware channel, so it can never be queued behind the very WaitValue
    // it must release (which happens when signal_stream and the parked user
    // stream collide on the same connection under CUDA_DEVICE_MAX_CONNECTIONS).
    uint64_t* complete_seq_host = nullptr;
    uint64_t* complete_seq_dev = nullptr;
    uint64_t prepared_seq = 0;
    devEvent reset_event{};
    devEvent wait_done_event{};
    bool wait_done_recorded = false;
  };

  int slotCountFromEnv() const;
  bool validSlot(int slot) const;
  std::string kernelName(CollectiveDebugOp op, int rank, int slot) const;

  bool initialized_;
  bool multi_process_;
  bool use_stream_mem_ops_;
  bool flush_remote_writes_;
  uint64_t compat_seq_;
  uint32_t* host_flag_zero_;  // pinned host memory, value = 0
  uint32_t* host_flag_one_;   // pinned host memory, value = 1
  int queue_depth_;

  devStream reset_stream_;   // dedicated stream for flag resets
  devStream signal_stream_;  // dedicated stream for completion signals
  Slot slots_[PCCL_EXECUTOR_QUEUE_DEPTH_MAX];
};

}  // namespace pccl

#endif /* PCCL_COMPLETION_FLAG_HPP */
