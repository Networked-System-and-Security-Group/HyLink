#include "completion_flag.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
      use_stream_mem_ops_(false),
      flush_remote_writes_(false),
      compat_seq_(0),
      host_flag_zero_(nullptr),
      host_flag_one_(nullptr),
      queue_depth_(PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT) {
  reset_stream_.handle = nullptr;
  signal_stream_.handle = nullptr;
}

CompletionFlag::~CompletionFlag() {
  if (!initialized_)
    return;

  LOG_INFO("CompletionFlag: Releasing resources");
  for (auto& slot : slots_) {
    if (slot.dev_flag) {
      devFree(slot.dev_flag);
      slot.dev_flag = nullptr;
    }
    if (slot.complete_seq_host) {
      devFreeMappedHost(slot.complete_seq_host);
      slot.complete_seq_host = nullptr;
      slot.complete_seq_dev = nullptr;
    }
    if (slot.ready_seq_host) {
      devFreeMappedHost(slot.ready_seq_host);
      slot.ready_seq_host = nullptr;
      slot.ready_seq_dev = nullptr;
    }
    if (slot.reset_event.handle) {
      devDestroyEvent(slot.reset_event);
      slot.reset_event.handle = nullptr;
    }
    if (slot.wait_done_event.handle) {
      devDestroyEvent(slot.wait_done_event);
      slot.wait_done_event.handle = nullptr;
    }
  }
  if (host_flag_zero_) {
    devFreeHost(host_flag_zero_);
    host_flag_zero_ = nullptr;
  }
  if (host_flag_one_) {
    devFreeHost(host_flag_one_);
    host_flag_one_ = nullptr;
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

std::string CompletionFlag::kernelName(CollectiveDebugOp op, int rank, int slot) const {
  char name[80];
  if (rank >= 0 && rank < 8) {
    std::snprintf(name, sizeof(name), "PcclCollective_%s_R%d_S%02d", waitOpName(op), rank, slot);
  } else {
    std::snprintf(name, sizeof(name), "PcclCollective_%s_RX_S%02d", waitOpName(op), slot);
  }
  return std::string(name);
}

int CompletionFlag::initialize(bool multi_process) {
  if (initialized_)
    return 0;

  multi_process_ = multi_process;
  queue_depth_ = slotCountFromEnv();
  use_stream_mem_ops_ = devCanUseStreamMemOps64();
  // Diagnostic / workaround override: PCCL_DISABLE_STREAM_MEM_OPS=1 forces the
  // kernel-wait completion path (a spinning kernel on the user stream that polls
  // a device flag) instead of cuStreamWaitValue. A spinning kernel occupies an
  // SM (compute engine), not a copy-engine channel, so it cannot head-of-line
  // block the executor's D2H/H2D lanes the way a parked cuStreamWaitValue on the
  // user stream's hardware connection can.
  {
    const char* disable = std::getenv("PCCL_DISABLE_STREAM_MEM_OPS");
    if (disable && disable[0] == '1') {
      use_stream_mem_ops_ = false;
      LOG_INFO("CompletionFlag: stream memory ops disabled via PCCL_DISABLE_STREAM_MEM_OPS");
    }
  }
  flush_remote_writes_ = use_stream_mem_ops_ && devCanFlushRemoteWrites();
  devStatus status = devMallocHost(reinterpret_cast<void**>(&host_flag_zero_), sizeof(uint32_t));
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to allocate host flag zero");
    return -1;
  }
  *host_flag_zero_ = 0;

  status = devMallocHost(reinterpret_cast<void**>(&host_flag_one_), sizeof(uint32_t));
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to allocate host flag one");
    devFreeHost(host_flag_zero_);
    host_flag_zero_ = nullptr;
    return -1;
  }
  *host_flag_one_ = 1;

  status = devCreateStream(&reset_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to create reset stream");
    devFreeHost(host_flag_one_);
    host_flag_one_ = nullptr;
    devFreeHost(host_flag_zero_);
    host_flag_zero_ = nullptr;
    return -1;
  }

  status = devCreateStream(&signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to create signal stream");
    devDestroyStream(reset_stream_);
    reset_stream_.handle = nullptr;
    devFreeHost(host_flag_one_);
    host_flag_one_ = nullptr;
    devFreeHost(host_flag_zero_);
    host_flag_zero_ = nullptr;
    return -1;
  }

  for (int i = 0; i < queue_depth_; ++i) {
    auto& slot = slots_[i];
    status = devMalloc(reinterpret_cast<void**>(&slot.dev_flag), sizeof(uint32_t));
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to allocate device flag slot=%d", i);
      return -1;
    }
    // Host-mapped completion sequence: the executor (CPU) writes complete_seq_host
    // directly in signalComplete; the user stream's cuStreamWaitValue64 polls
    // complete_seq_dev (device-side mapping). Releasing the wait needs no GPU
    // hardware channel, so the completion signal can never be head-of-line blocked
    // behind the very WaitValue it must release (signal_stream/user-stream
    // connection collision under CUDA_DEVICE_MAX_CONNECTIONS).
    status = devMallocMappedHost(reinterpret_cast<void**>(&slot.complete_seq_host),
                                 reinterpret_cast<void**>(&slot.complete_seq_dev),
                                 sizeof(uint64_t));
    if (status != devSuccess || !slot.complete_seq_host || !slot.complete_seq_dev) {
      LOG_ERROR("CompletionFlag: Failed to allocate mapped completion sequence slot=%d status=%d", i,
                static_cast<int>(status));
      return -1;
    }
    __atomic_store_n(slot.complete_seq_host, 0, __ATOMIC_RELEASE);
    status = devMallocMappedHost(reinterpret_cast<void**>(&slot.ready_seq_host),
                                 reinterpret_cast<void**>(&slot.ready_seq_dev),
                                 sizeof(uint64_t));
    if (status != devSuccess || !slot.ready_seq_host || !slot.ready_seq_dev) {
      LOG_ERROR("CompletionFlag: Failed to allocate mapped ready doorbell slot=%d status=%d", i,
                static_cast<int>(status));
      return -1;
    }
    *slot.ready_seq_host = 0;
    status = devCreateSyncEvent(&slot.reset_event);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to create reset event slot=%d", i);
      return -1;
    }
    status = devCreateSyncEvent(&slot.wait_done_event);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to create wait-done event slot=%d", i);
      return -1;
    }
    status = devMemcpyAsync(slot.dev_flag, host_flag_zero_, sizeof(uint32_t), MemcpyDirection::H2D,
                            reset_stream_);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to initialize flag slot=%d", i);
      return -1;
    }
    status = devRecordEvent(slot.wait_done_event, reset_stream_);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to initialize wait-done event slot=%d", i);
      return -1;
    }
    slot.wait_done_recorded = true;
    // Touch each slot's host-mapped sequence words with a stream memory op once
    // here. The first cuStreamWrite/WaitValue64 on a new address costs ~0.7 ms;
    // left lazy, the first queue_depth_ collectives each pay it, which shows up
    // as a dip at the first graph-path message size (e.g. 1M on 4x V100).
    if (use_stream_mem_ops_) {
      if (devStreamWriteValue64(reset_stream_, slot.ready_seq_dev, 0) != devSuccess ||
          devStreamWaitValue64(reset_stream_, slot.complete_seq_dev, 0, flush_remote_writes_) != devSuccess) {
        LOG_WARN("CompletionFlag: stream mem op warmup failed slot=%d", i);
      }
    }
  }
  devSynchronizeStream(reset_stream_);

  initialized_ = true;
  LOG_INFO("CompletionFlag: Initialized queue_depth=%d multi_process=%d stream_mem_ops=%d flush_remote_writes=%d",
           queue_depth_,
           multi_process_ ? 1 : 0,
           use_stream_mem_ops_ ? 1 : 0,
           flush_remote_writes_ ? 1 : 0);
  return 0;
}

uint64_t CompletionFlag::readySequence(int slot) const {
  if (!validSlot(slot) || !slots_[slot].ready_seq_host)
    return 0;
  return __atomic_load_n(slots_[slot].ready_seq_host, __ATOMIC_ACQUIRE);
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

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion prepare begin slot=%d op=%s rank=%d seq=%llu count=%zu dtype=%zu comm=%p "
             "stream=%p flag=%p ready_host=%p ready_dev=%p reset_stream=%p signal_stream=%p",
             slot_idx,
             waitOpName(op),
             rank,
             static_cast<unsigned long long>(seq),
             count,
             dtype,
             comm,
             stream,
             static_cast<void*>(slot.dev_flag),
             static_cast<void*>(slot.ready_seq_host),
             static_cast<void*>(slot.ready_seq_dev),
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

  __atomic_store_n(slot.ready_seq_host, 0, __ATOMIC_RELEASE);
  slot.prepared_seq = seq;

  if (use_stream_mem_ops_) {
    status = devRecordEvent(slot.reset_event, reset_stream_);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to record stream-mem-op reuse event slot=%d status=%d", slot_idx,
                static_cast<int>(status));
      return -1;
    }
    status = devStreamWaitEvent(user_stream, slot.reset_event);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to order user stream after stream-mem-op reuse slot=%d status=%d",
                slot_idx,
                static_cast<int>(status));
      return -1;
    }
    status = devStreamWriteValue64(user_stream, slot.ready_seq_dev, seq);
    if (status != devSuccess) {
      LOG_ERROR("CompletionFlag: Failed to enqueue ready sequence write slot=%d status=%d", slot_idx,
                static_cast<int>(status));
      return -1;
    }
    status = devStreamWaitValue64(user_stream, slot.complete_seq_dev, seq, flush_remote_writes_);
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
      LOG_INFO("ASYNCDBG completion prepare end slot=%d stream_mem_ops=1 ready_host=%p ready_dev=%p complete_dev=%p "
               "wait_done=%p",
               slot_idx,
               static_cast<void*>(slot.ready_seq_host),
               static_cast<void*>(slot.ready_seq_dev),
               static_cast<void*>(slot.complete_seq_dev),
               slot.wait_done_event.handle);
    }
    return 0;
  }

  status = devMemcpyAsync(slot.dev_flag, host_flag_zero_, sizeof(uint32_t), MemcpyDirection::H2D,
                          reset_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to reset flag slot=%d status=%d", slot_idx, static_cast<int>(status));
    return -1;
  }
  status = devRecordEvent(slot.reset_event, reset_stream_);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to record reset event slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  status = devStreamWaitEvent(user_stream, slot.reset_event);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to order user stream after reset slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  struct CollectiveArgs {
    void* complete_flag_ptr;
    void* ready_seq_ptr;
    uint64_t work_seq;
  } args{slot.dev_flag, slot.ready_seq_dev, seq};
  std::string name = kernelName(op, rank, slot_idx);
  status = devLaunchKernel(name.c_str(), 1, &args, sizeof(args), user_stream);
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to launch %s slot=%d status=%d", name.c_str(), slot_idx,
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
    LOG_INFO("ASYNCDBG completion prepare end slot=%d kernel=%s ready_host=%p wait_done=%p",
             slot_idx,
             name.c_str(),
             static_cast<void*>(slot.ready_seq_host),
             slot.wait_done_event.handle);
  }
  return 0;
}

int CompletionFlag::signalComplete(int slot_idx, uint64_t seq) {
  if (!validSlot(slot_idx))
    return -1;

  Slot& slot = slots_[slot_idx];
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion signalComplete begin slot=%d seq=%llu signal_stream=%p flag=%p complete_dev=%p",
             slot_idx,
             static_cast<unsigned long long>(seq),
             signal_stream_.handle,
             static_cast<void*>(slot.dev_flag),
             static_cast<void*>(slot.complete_seq_dev));
  }
  devStatus status = devSuccess;
  if (use_stream_mem_ops_) {
    // Write the completion sequence directly from the CPU into host-mapped memory.
    // The user stream's WaitValue64 polls the device-side mapping over the GPU's
    // memory-op engine, so no GPU stream/channel is needed to publish completion.
    // This is what breaks the head-of-line deadlock: previously this was a
    // devStreamWriteValue64 on signal_stream_, which could be queued behind the
    // user stream's parked WaitValue on the same hardware connection.
    // The collective's DMAs are already GPU-visible at this point (the executor
    // confirmed all CUDA events complete before calling signalComplete), so the
    // data is coherent before the flag is published.
    __atomic_store_n(slot.complete_seq_host, seq, __ATOMIC_RELEASE);
  } else {
    status = devMemcpyAsync(slot.dev_flag, host_flag_one_, sizeof(uint32_t), MemcpyDirection::H2D,
                            signal_stream_);
  }
  if (status != devSuccess) {
    LOG_ERROR("CompletionFlag: Failed to signal completion slot=%d status=%d", slot_idx,
              static_cast<int>(status));
    return -1;
  }
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG completion signalComplete end slot=%d seq=%llu signal_stream=%p flag=%p complete_dev=%p",
             slot_idx,
             static_cast<unsigned long long>(seq),
             signal_stream_.handle,
             static_cast<void*>(slot.dev_flag),
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
  devStatus status = devMemcpyAsync(slots_[0].dev_flag, host_flag_zero_, sizeof(uint32_t), MemcpyDirection::H2D,
                                    reset_stream_);
  if (status != devSuccess)
    return -1;
  status = devRecordEvent(slots_[0].reset_event, reset_stream_);
  return status == devSuccess ? 0 : -1;
}

int CompletionFlag::ensureResetVisible(devStream user_stream) {
  if (!validSlot(0))
    return -1;
  return devStreamWaitEvent(user_stream, slots_[0].reset_event) == devSuccess ? 0 : -1;
}

int CompletionFlag::recordWaitComplete(devStream user_stream) {
  if (!validSlot(0))
    return -1;
  return devRecordEvent(slots_[0].wait_done_event, user_stream) == devSuccess ? 0 : -1;
}

}  // namespace pccl
