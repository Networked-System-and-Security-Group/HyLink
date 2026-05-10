#include "completion_flag.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#include "../util/debug_flags.hpp"
#include "../util/log.hpp"

namespace pccl {

namespace {

const char* waitOpName(CollectiveDebugOp op) {
  switch (op) {
    case CollectiveDebugOp::AllReduce:
      return "AllReduce";
    case CollectiveDebugOp::AllGather:
      return "AllGather";
    case CollectiveDebugOp::ReduceScatter:
      return "ReduceScatter";
    case CollectiveDebugOp::AllToAll:
      return "AllToAll";
    case CollectiveDebugOp::Broadcast:
      return "Broadcast";
    case CollectiveDebugOp::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

}  // namespace

CompletionFlag::CompletionFlag()
    : initialized_(false),
      multi_process_(false),
      compat_seq_(0),
      queue_depth_(PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT) {
  reset_stream_.handle = nullptr;
  signal_stream_.handle = nullptr;
}

CompletionFlag::~CompletionFlag() {
  if (!initialized_)
    return;

  LOG_INFO("CompletionFlag: Releasing resources");
  for (auto& slot : slots_) {
    if (slot.ready_seq_dev) {
      devFree(slot.ready_seq_dev);
      slot.ready_seq_dev = nullptr;
    }
    if (slot.complete_seq_dev) {
      devFree(slot.complete_seq_dev);
      slot.complete_seq_dev = nullptr;
    }
    if (slot.ready_event.handle) {
      devDestroyEvent(slot.ready_event);
      slot.ready_event.handle = nullptr;
    }
    if (slot.wait_done_event.handle) {
      devDestroyEvent(slot.wait_done_event);
      slot.wait_done_event.handle = nullptr;
    }
    if (slot.completion_event.handle) {
      devDestroyEvent(slot.completion_event);
      slot.completion_event.handle = nullptr;
    }
  }
  if (reset_stream_.handle) {
    devDestroyStream(reset_stream_);
    reset_stream_.handle = nullptr;
  }
  if (signal_stream_.handle) {
    devDestroyStream(signal_stream_);
    signal_stream_.handle = nullptr;
  }
  initialized_ = false;
}

int CompletionFlag::slotCountFromEnv() const {
  const char* value = std::getenv("PCCL_EXECUTOR_QUEUE_DEPTH");
  if (!value || value[0] == '\0')
    return PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0)
    return PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  return static_cast<int>(std::max<long>(1, std::min<long>(PCCL_EXECUTOR_QUEUE_DEPTH_MAX, parsed)));
}

bool CompletionFlag::validSlot(int slot) const {
  return initialized_ && slot >= 0 && slot < queue_depth_;
}

int CompletionFlag::initialize(bool multi_process) {
  if (initialized_)
    return 0;

  multi_process_ = multi_process;
  const char* mode = std::getenv("PCCL_COMPLETION_WAIT_MODE");
  if (mode && mode[0] && std::strcmp(mode, "value") != 0 && std::strcmp(mode, "event") != 0) {
    LOG_ERROR("CompletionFlag: invalid PCCL_COMPLETION_WAIT_MODE=%s", mode);
    return -1;
  }
  event_wait_ = mode && std::strcmp(mode, "event") == 0;
  queue_depth_ = slotCountFromEnv();

  devStatus status = devCreateStream(&reset_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to create reset stream");
    return -1;
  }

  status = devCreateStream(&signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to create signal stream");
    devDestroyStream(reset_stream_);
    reset_stream_.handle = nullptr;
    return -1;
  }

  for (int i = 0; i < queue_depth_; ++i) {
    auto& slot = slots_[i];
    status = devMalloc(reinterpret_cast<void**>(&slot.ready_seq_dev), sizeof(uint64_t));
    if (status != devSuccess || !slot.ready_seq_dev) {
      LOG_ERROR("CompletionFlag: Failed to allocate ready sequence slot=%d status=%d", i,
                static_cast<int>(status));
      return -1;
    }
    status = devMalloc(reinterpret_cast<void**>(&slot.complete_seq_dev), sizeof(uint64_t));
    if (status != devSuccess || !slot.complete_seq_dev) {
      LOG_ERROR("CompletionFlag: Failed to allocate completion sequence slot=%d status=%d", i,
                static_cast<int>(status));
      return -1;
    }
    status = devMemsetAsync(slot.ready_seq_dev, 0, sizeof(uint64_t), reset_stream_);
    if (status == devSuccess)
      status = devMemsetAsync(slot.complete_seq_dev, 0, sizeof(uint64_t), reset_stream_);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to initialize doorbell memory slot=%d status=%d", i,
                static_cast<int>(status));
      return -1;
    }
    status = devCreateSyncEvent(&slot.ready_event);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to create ready event slot=%d", i);
      return -1;
    }
    status = devCreateSyncEvent(&slot.wait_done_event);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to create wait-done event slot=%d", i);
      return -1;
    }
    status = devRecordEvent(slot.wait_done_event, reset_stream_);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to initialize wait-done event slot=%d", i);
      return -1;
    }
    slot.wait_done_recorded = true;
    if (event_wait_) {
      status = devCreateSyncEvent(&slot.completion_event);
      if (status != devSuccess) {
        LOG_ERROR("CompletionFlag: Failed to create completion event slot=%d", i);
        return -1;
      }
    }
  }
  status = devSynchronizeStream(reset_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to synchronize initialization stream status=%d", static_cast<int>(status));
    return -1;
  }

  initialized_ = true;
  LOG_INFO("CompletionFlag: completion_wait_mode=%s", event_wait_ ? "event" : "value");
  LOG_INFO("CompletionFlag: Initialized queue_depth=%d multi_process=%d", queue_depth_,
           multi_process_ ? 1 : 0);
  return 0;
}

uint64_t CompletionFlag::readySequence(int slot) const {
  if (!validSlot(slot))
    return 0;
  const Slot& s = slots_[slot];
  if (!s.ready_recorded || s.prepared_seq == 0)
    return 0;
  devEventStatus status = devEventStatusNotReady;
  if (devQueryEvent(s.ready_event, &status) != devSuccess)
    return 0;
  return status == devEventStatusComplete ? s.prepared_seq : 0;
}

bool CompletionFlag::isReady(int slot, uint64_t work_seq) const {
  return readySequence(slot) == work_seq;
}

int CompletionFlag::prepareUserWait(int slot_idx, devStream user_stream, CollectiveDebugOp op, int rank,
                                    uint64_t seq, size_t count, size_t dtype, const void* comm,
                                    const void* stream) {
  if (!validSlot(slot_idx))
    return -1;
  Slot& slot = slots_[slot_idx];

  // An event is not a monotonically increasing sequence doorbell. Before a
  // slot's next generation is submitted, its old wait/reset must have retired.
  // Only the experimental path pays this reuse check; the default is unchanged.
  if (event_wait_ && slot.prepared_seq != 0 && slot.wait_done_recorded) {
    auto start = std::chrono::steady_clock::now();
    long timeout_ms = 120000;
    if (const char* value = std::getenv("PCCL_EXECUTOR_TIMEOUT_MS")) {
      long parsed = std::strtol(value, nullptr, 10);
      if (parsed > 0) timeout_ms = parsed;
    }
    while (true) {
      devEventStatus done;
      if (devQueryEvent(slot.wait_done_event, &done) != devSuccess) return -1;
      if (done == devEventStatusComplete) break;
      if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(timeout_ms)) {
        LOG_ERROR("CompletionFlag: completion event reuse timeout slot=%d", slot_idx);
        return -1;
      }
      std::this_thread::yield();
    }
  }

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion prepare begin slot=%d op=%s rank=%d seq=%llu count=%zu dtype=%zu comm=%p "
             "stream=%p ready_dev=%p complete_dev=%p reset_stream=%p signal_stream=%p",
             slot_idx,
             waitOpName(op),
             rank,
             static_cast<unsigned long long>(seq),
             count,
             dtype,
             comm,
             stream,
             static_cast<void*>(slot.ready_seq_dev),
             static_cast<void*>(slot.complete_seq_dev),
             reset_stream_.handle,
             signal_stream_.handle);
  }

  devStatus status = devSuccess;
  if (slot.wait_done_recorded) {
    status = devStreamWaitEvent(reset_stream_, slot.wait_done_event);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to wait for slot reuse slot=%d status=%d", slot_idx,
                static_cast<int>(status));
      return -1;
    }
  }

  slot.prepared_seq = seq;
  slot.ready_recorded = false;

  status = devStreamWriteValue64(user_stream, slot.ready_seq_dev, seq);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to enqueue ready sequence write slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  status = devRecordEvent(slot.ready_event, user_stream);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to record ready event slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  slot.ready_recorded = true;

  if (event_wait_) {
    // ACL_EVENT_SYNC supports Wait-before-Record. Reset executes after the
    // wait on the same stream; signalComplete records from the signal stream.
    status = devStreamWaitEvent(user_stream, slot.completion_event);
    if (status == devSuccess)
      status = aclrtResetEvent(slot.completion_event.handle, user_stream.handle);
  } else {
    status = devStreamWaitValue64(user_stream, slot.complete_seq_dev, seq);
  }
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to enqueue completion sequence wait slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  status = devRecordEvent(slot.wait_done_event, user_stream);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to record wait-done event slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  slot.wait_done_recorded = true;

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion prepare end slot=%d ready_dev=%p complete_dev=%p wait_done=%p",
             slot_idx,
             static_cast<void*>(slot.ready_seq_dev),
             static_cast<void*>(slot.complete_seq_dev),
             slot.wait_done_event.handle);
  }
  return 0;
}

int CompletionFlag::signalComplete(int slot_idx, uint64_t seq) {
  if (!validSlot(slot_idx))
    return -1;

  Slot& slot = slots_[slot_idx];
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion signalComplete begin slot=%d seq=%llu signal_stream=%p complete_dev=%p",
             slot_idx,
             static_cast<unsigned long long>(seq),
             signal_stream_.handle,
             static_cast<void*>(slot.complete_seq_dev));
  }

  devStatus status = event_wait_ ? devRecordEvent(slot.completion_event, signal_stream_)
                               : devStreamWriteValue64(signal_stream_, slot.complete_seq_dev, seq);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to signal completion slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion signalComplete end slot=%d seq=%llu signal_stream=%p complete_dev=%p",
             slot_idx,
             static_cast<unsigned long long>(seq),
             signal_stream_.handle,
             static_cast<void*>(slot.complete_seq_dev));
  }
  return 0;
}

int CompletionFlag::signalComplete(int slot_idx) {
  if (!validSlot(slot_idx))
    return -1;
  uint64_t seq = slots_[slot_idx].prepared_seq;
  return signalComplete(slot_idx, seq == 0 ? 1 : seq);
}

int CompletionFlag::enqueueUserWait(devStream user_stream) {
  uint64_t seq = ++compat_seq_;
  return prepareUserWait(0, user_stream, CollectiveDebugOp::Unknown, -1, seq, 0, 0, nullptr,
                         user_stream.handle);
}

int CompletionFlag::signalComplete() {
  return signalComplete(0, compat_seq_);
}

int CompletionFlag::reset() {
  if (!validSlot(0))
    return -1;
  if (slots_[0].wait_done_recorded) {
    devStatus wait_status = devStreamWaitEvent(reset_stream_, slots_[0].wait_done_event);
    if (wait_status != devSuccess)
      return -1;
  }
  devStatus status = devMemsetAsync(slots_[0].ready_seq_dev, 0, sizeof(uint64_t), reset_stream_);
  if (status == devSuccess)
    status = devMemsetAsync(slots_[0].complete_seq_dev, 0, sizeof(uint64_t), reset_stream_);
  return status == devSuccess ? 0 : -1;
}

int CompletionFlag::ensureResetVisible(devStream user_stream) {
  if (!validSlot(0))
    return -1;
  devEvent reset_event{};
  if (devCreateSyncEvent(&reset_event) != devSuccess)
    return -1;
  devStatus status = devRecordEvent(reset_event, reset_stream_);
  if (status == devSuccess)
    status = devStreamWaitEvent(user_stream, reset_event);
  devDestroyEvent(reset_event);
  return status == devSuccess ? 0 : -1;
}

int CompletionFlag::recordWaitComplete(devStream user_stream) {
  if (!validSlot(0))
    return -1;
  return devRecordEvent(slots_[0].wait_done_event, user_stream) == devSuccess ? 0 : -1;
}

}  // namespace pccl
