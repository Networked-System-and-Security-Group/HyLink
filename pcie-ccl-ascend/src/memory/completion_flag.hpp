#ifndef PCCL_COMPLETION_FLAG_HPP
#define PCCL_COMPLETION_FLAG_HPP

#include <cstdint>
#include <cstddef>
#include <string>

#include "../core/limits.hpp"
#include "../runtime/ascend_runtime.hpp"

namespace pccl {

enum class CollectiveDebugOp {
  AllReduce,
  AllGather,
  ReduceScatter,
  AllToAll,
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
    uint64_t* ready_seq_dev = nullptr;
    uint64_t* complete_seq_dev = nullptr;
    uint64_t prepared_seq = 0;
    devEvent ready_event{};
    devEvent wait_done_event{};
    devEvent completion_event{};
    bool ready_recorded = false;
    bool wait_done_recorded = false;
  };

  int slotCountFromEnv() const;
  bool validSlot(int slot) const;

  bool initialized_;
  bool multi_process_;
  bool event_wait_ = false;  // experimental; default keeps the sequence doorbell
  uint64_t compat_seq_;
  int queue_depth_;

  devStream reset_stream_;   // dedicated stream for flag resets
  devStream signal_stream_;  // dedicated stream for completion signals
  Slot slots_[PCCL_EXECUTOR_QUEUE_DEPTH_MAX];
};

}  // namespace pccl

#endif /* PCCL_COMPLETION_FLAG_HPP */
