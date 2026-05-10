#include "include/scheduler.hpp"

#include <algorithm>
#include <map>

#include "include/host_memory_pool.hpp"
#include "include/log.hpp"
#include "include/mem_op_manager.hpp"
#include "include/nvtx.hpp"
#include "include/state_manager.hpp"
#include "include/sync_flags.hpp"
#include "kernels/wait_kernel.cpp"

namespace pccl {

// Constructor
Scheduler::Scheduler(int my_rank, int world_size, int my_numa_node, StateManager* state_manager,
                     MemoryOperationManager* mem_op_manager, HostMemoryPool* host_mem_pool,
                     SyncFlagManager* sync_flag_manager)
    : my_rank_(my_rank),
      world_size_(world_size),
      my_numa_node_(my_numa_node),
      device_id_(0),
      state_manager_(state_manager),
      mem_op_manager_(mem_op_manager),
      host_mem_pool_(host_mem_pool),
      sync_flag_manager_(sync_flag_manager),
      error_state_(pcclSuccess),
      d2h_streams_{},
      h2d_streams_{},
      d2d_stream_{nullptr},
      d2h_stream_rr_(0),
      h2d_stream_rr_(0),
      stop_d2h_thread_(false),
      stop_h2d_thread_(false),
      d2h_batch_pending_(false),
      fast_sync_ev_{} {
  LOG_INFO("Rank %d: Initializing Scheduler (event-based D2H/H2D)", my_rank_);

  // Get this rank's NUMA node
  if (host_mem_pool_) {
    my_numa_node_ = host_mem_pool_->getMyNUMANode();
  }

  // Get current device ID for thread context
  devGetDevice(&device_id_);

  // Create internal streams for D2H/H2D/D2D parallelization
  devStatus ret;
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    ret = devCreateStream(&d2h_streams_[i]);
    if (ret != devSuccess) {
      LOG_ERROR("Rank %d: Failed to create D2H stream %d, status=%d", my_rank_, i, ret);
      for (int j = 0; j < i; j++) devDestroyStream(d2h_streams_[j]);
      error_state_.store(static_cast<int>(pcclDeviceError));
      return;
    }
  }

  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    ret = devCreateStream(&h2d_streams_[i]);
    if (ret != devSuccess) {
      LOG_ERROR("Rank %d: Failed to create H2D stream %d, status=%d", my_rank_, i, ret);
      for (int j = 0; j < i; j++) devDestroyStream(h2d_streams_[j]);
      for (int j = 0; j < PCCL_TRANSFER_STREAM_COUNT; j++) devDestroyStream(d2h_streams_[j]);
      error_state_.store(static_cast<int>(pcclDeviceError));
      return;
    }
  }

  ret = devCreateStream(&d2d_stream_);
  if (ret != devSuccess) {
    LOG_ERROR("Rank %d: Failed to create D2D stream, status=%d", my_rank_, ret);
    for (int j = 0; j < PCCL_TRANSFER_STREAM_COUNT; j++) devDestroyStream(d2h_streams_[j]);
    for (int j = 0; j < PCCL_TRANSFER_STREAM_COUNT; j++) devDestroyStream(h2d_streams_[j]);
    error_state_.store(static_cast<int>(pcclDeviceError));
    return;
  }

  LOG_DEBUG("Rank %d: Created internal streams (%d D2H, %d H2D, 1 D2D)", my_rank_, PCCL_TRANSFER_STREAM_COUNT,
            PCCL_TRANSFER_STREAM_COUNT);

  // Pre-allocate lightweight events for small-data fast path spin-sync
  for (int i = 0; i < 2; i++) {
    devCreateEvent(&fast_sync_ev_[i]);
  }

  // Start D2H executor thread
  d2h_thread_ = std::thread(&Scheduler::d2hExecutorThreadFunc, this);

  // Start H2D executor thread
  h2d_thread_ = std::thread(&Scheduler::h2dExecutorThreadFunc, this);

  LOG_INFO("Rank %d: Scheduler initialized (numa_node=%d)", my_rank_, my_numa_node_);
}

// Destructor
Scheduler::~Scheduler() {
  // Stop D2H executor thread
  {
    std::lock_guard<std::mutex> lock(d2h_mutex_);
    stop_d2h_thread_.store(true);
  }
  d2h_cv_.notify_one();
  if (d2h_thread_.joinable()) {
    d2h_thread_.join();
  }

  // Stop H2D executor thread
  {
    std::lock_guard<std::mutex> lock(h2d_mutex_);
    stop_h2d_thread_.store(true);
  }
  h2d_cv_.notify_one();
  if (h2d_thread_.joinable()) {
    h2d_thread_.join();
  }

  // Destroy fast-path sync events
  for (int i = 0; i < 2; i++) {
    if (fast_sync_ev_[i].handle) devDestroyEvent(fast_sync_ev_[i]);
  }

  // Destroy internal streams
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    if (h2d_streams_[i].handle) {
      devDestroyStream(h2d_streams_[i]);
    }
  }
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    if (d2h_streams_[i].handle) {
      devDestroyStream(d2h_streams_[i]);
    }
  }
  if (d2d_stream_.handle) {
    devDestroyStream(d2d_stream_);
  }

  LOG_DEBUG("Rank %d: Destroyed internal streams", my_rank_);

#if LOG_LEVEL <= 0
  // Destroy any remaining profiling events
  for (auto& record : profiling_records_) {
    if (record.start_event.handle) {
      devDestroyEvent(record.start_event);
    }
    if (record.end_event.handle) {
      devDestroyEvent(record.end_event);
    }
  }
  profiling_records_.clear();
#endif

  LOG_INFO("Rank %d: Scheduler destroyed", my_rank_);
}

// D2H executor thread function
// Processes D2HTask queue, uses dual-thread producer-consumer pattern:
// - Producer (this thread): submits memcpy + records events
// - Consumer (progress thread): polls events and updates progress
void Scheduler::d2hExecutorThreadFunc() {
  // Set device context for this thread
  devSetDevice(device_id_);

  EventRingBuffer event_ring(16);
  if (event_ring.initialize() != 0) {
    LOG_ERROR("Rank %d: D2H executor thread failed to initialize EventRingBuffer", my_rank_);
    error_state_.store(static_cast<int>(pcclInternalError));
    return;
  }

  LOG_DEBUG("Rank %d: D2H executor thread started", my_rank_);

  while (true) {
    std::vector<D2HTask> tasks;

    // Wait for tasks or stop signal
    {
      std::unique_lock<std::mutex> lock(d2h_mutex_);
      d2h_cv_.wait(lock, [this] { return !d2h_tasks_.empty() || stop_d2h_thread_.load(); });

      if (stop_d2h_thread_.load() && d2h_tasks_.empty()) {
        break;
      }

      // Take all pending tasks
      tasks.swap(d2h_tasks_);
    }

    // Process each D2H task
    for (const auto& task : tasks) {
      int chunk_idx = task.effect_chunk_idx;
      size_t total_slices = task.sub_chunks.size();

      char nvtx_task_buf[128];
      snprintf(nvtx_task_buf, sizeof(nvtx_task_buf), "D2H R%d c%d %zus", my_rank_, chunk_idx, total_slices);
      NvtxRangeGuard nvtx_d2h_task(nvtx_task_buf, PCCL_NVTX_COLOR_D2H_EXEC);

      // 1. Set version via state_manager (version = 1, indicates data is being written)
      state_manager_->setChunkVersion(my_numa_node_, chunk_idx, 1);

      // 2. Reset progress to 0
      state_manager_->setChunkProgress(my_numa_node_, chunk_idx, 0);

#if LOG_LEVEL <= 0
      ProfilingRecord record;
      record.op_type = ProfilingRecord::D2H;
      record.src_chunk_idx = task.src_chunk_idx;
      record.dst_chunk_idx = task.dst_chunk_idx;
      record.size_bytes = task.chunk_size;
      devCreateTimingEvent(&record.start_event);
      devCreateTimingEvent(&record.end_event);
      devRecordEvent(record.start_event, d2h_streams_[0]);
#endif

      // NOTE: single-subchunk tasks (total_slices == 1) are now handled by
      // submitSmallData() before reaching the executor, so we only have the
      // multi-subchunk producer-consumer path here.
      {
        // Start progress consumer thread
        // This thread polls events and updates progress, running concurrently with memcpy submissions
        PCCL_NVTX_RANGE_PUSH("D2H_ThreadCreate", PCCL_NVTX_COLOR_NOTIFY);
        std::thread progress_thread([&, chunk_idx, total_slices]() {
          char nvtx_prog_buf[128];
          snprintf(nvtx_prog_buf, sizeof(nvtx_prog_buf), "D2H_Progress R%d c%d", my_rank_, chunk_idx);
          NvtxRangeGuard nvtx_prog(nvtx_prog_buf, PCCL_NVTX_COLOR_SPIN_WAIT);

          size_t consumed = 0;
          while (consumed < total_slices) {
            if (!event_ring.hasPublished()) {
              continue;  // No published events yet, keep spinning
            }

            devEvent ev = event_ring.peekOldest();
            devEventStatus status;
            devStatus ret = devQueryEvent(ev, &status);
            if (ret != devSuccess) {
              LOG_ERROR("Rank %d: D2H progress thread devQueryEvent failed, status=%d", my_rank_, ret);
              break;
            }

            if (status == devEventStatusComplete) {
              consumed++;
              state_manager_->setChunkProgress(my_numa_node_, chunk_idx, consumed);
              event_ring.release();  // Release slot, may wake up producer
            }
          }
        });
        PCCL_NVTX_RANGE_POP();  // D2H_ThreadCreate

        // Producer: submit all memcpy + record + publish (no event query here!)
        for (size_t i = 0; i < total_slices; i++) {
          const D2HSubChunk& sc = task.sub_chunks[i];

          devStream& stream = d2h_streams_[d2h_stream_rr_];
          d2h_stream_rr_ = (d2h_stream_rr_ + 1) % PCCL_TRANSFER_STREAM_COUNT;

          devStatus ret = devMemcpyAsync(sc.host_dst, sc.dev_src, sc.size, MemcpyDirection::D2H, stream);
          if (ret != devSuccess) {
            LOG_ERROR("Rank %d: D2H slice %zu memcpyAsync failed, status=%d", my_rank_, i, ret);
            error_state_.store(static_cast<int>(pcclDeviceError));
            break;
          }

          // Acquire event slot (blocks if ring full, waits for consumer to release)
          devEvent ev = event_ring.acquire();
          ret = devRecordEvent(ev, stream);
          if (ret != devSuccess) {
            LOG_ERROR("Rank %d: D2H slice %zu devRecordEvent failed, status=%d", my_rank_, i, ret);
            error_state_.store(static_cast<int>(pcclDeviceError));
            break;
          }

          // Mark as published so consumer can poll it
          event_ring.publish();
        }

        // Wait for progress thread to finish consuming all events
        PCCL_NVTX_RANGE_PUSH("D2H_ThreadJoin", PCCL_NVTX_COLOR_NOTIFY);
        progress_thread.join();
        PCCL_NVTX_RANGE_POP();  // D2H_ThreadJoin
      }
#if LOG_LEVEL <= 0
      // Sync all D2H streams to streams_[0] before recording end event
      for (int s = 1; s < PCCL_TRANSFER_STREAM_COUNT; s++) {
        devEvent sync_ev;
        devCreateSyncEvent(&sync_ev);
        devRecordEvent(sync_ev, d2h_streams_[s]);
        devStreamWaitEvent(d2h_streams_[0], sync_ev);
        devDestroyEvent(sync_ev);
      }
      devRecordEvent(record.end_event, d2h_streams_[0]);
      // Need mutex to safely push profiling records
      {
        std::lock_guard<std::mutex> lock(profiling_mutex_);
        profiling_records_.push_back(record);
      }
#endif

      LOG_DEBUG("Rank %d: D2H task [%d->%d] completed (%zu slices)", my_rank_, task.src_chunk_idx, task.dst_chunk_idx,
                total_slices);
    }
    // Mark D2H batch as complete so H2D executor can proceed with global barrier
    if (!tasks.empty()) {
      d2h_batch_pending_.store(false, std::memory_order_release);
      LOG_DEBUG("Rank %d: D2H batch complete, cleared d2h_batch_pending", my_rank_);
    }
  }

  event_ring.cleanup();
  LOG_DEBUG("Rank %d: D2H executor thread stopped", my_rank_);
}

// H2D executor thread function
// Round-robin monitors chunk progress across all H2D tasks and issues memcpy
// as data becomes available, enabling interleaved subchunk pipelining.
void Scheduler::h2dExecutorThreadFunc() {
  // Set device context for this thread
  devSetDevice(device_id_);

  // Array of events for tracking H2D completion across all streams
  devEvent completion_events[PCCL_TRANSFER_STREAM_COUNT];
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    if (devCreateEvent(&completion_events[i]) != devSuccess) {
      LOG_ERROR("Rank %d: H2D executor thread failed to create completion event %d", my_rank_, i);
      for (int j = 0; j < i; j++) devDestroyEvent(completion_events[j]);
      error_state_.store(static_cast<int>(pcclInternalError));
      return;
    }
  }

  LOG_DEBUG("Rank %d: H2D executor thread started", my_rank_);

  while (true) {
    std::vector<H2DTask> tasks;

    // Wait for tasks or stop signal
    {
      std::unique_lock<std::mutex> lock(h2d_mutex_);
      h2d_cv_.wait(lock, [this] { return !h2d_tasks_.empty() || stop_h2d_thread_.load(); });

      if (stop_h2d_thread_.load() && h2d_tasks_.empty()) {
        break;
      }

      // Take all pending tasks (submit() pushes all H2D tasks at once)
      tasks.swap(h2d_tasks_);
    }

    if (tasks.empty()) continue;

    // Check if this is a barrier-only batch (no real H2D work)
    bool has_h2d_work = false;
    for (const auto& t : tasks) {
      if (!t.is_barrier_only) {
        has_h2d_work = true;
        break;
      }
    }

    if (has_h2d_work) {
      char nvtx_h2d_buf[128];
      snprintf(nvtx_h2d_buf, sizeof(nvtx_h2d_buf), "H2D_Batch R%d", my_rank_);
      NvtxRangeGuard nvtx_h2d_batch(nvtx_h2d_buf, PCCL_NVTX_COLOR_H2D_EXEC);
#if LOG_LEVEL <= 0
      ProfilingRecord record;
      record.op_type = ProfilingRecord::H2D;
      record.src_chunk_idx = -1;
      record.dst_chunk_idx = -1;
      record.size_bytes = 0;
      devCreateTimingEvent(&record.start_event);
      devCreateTimingEvent(&record.end_event);
      devRecordEvent(record.start_event, h2d_streams_[0]);
      for (const auto& t : tasks) {
        if (!t.is_barrier_only) record.size_bytes += t.chunk_size;
      }
#endif

      // Round-robin progress monitoring across all chunks
      size_t num_tasks = tasks.size();
      std::vector<size_t> next_slice(num_tasks, 0);  // Next subchunk to copy per task
      size_t completed_tasks = 0;

      LOG_DEBUG("Rank %d: H2D round-robin started (%zu tasks)", my_rank_, num_tasks);

      {
        char nvtx_rr_buf[128];
        snprintf(nvtx_rr_buf, sizeof(nvtx_rr_buf), "H2D_RoundRobin R%d %zut", my_rank_, num_tasks);
        NvtxRangeGuard nvtx_rr(nvtx_rr_buf, PCCL_NVTX_COLOR_H2D_EXEC);

      while (completed_tasks < num_tasks) {
        for (size_t t = 0; t < num_tasks; t++) {
          const H2DTask& task = tasks[t];

          // Skip barrier-only or already-completed tasks
          if (task.is_barrier_only) {
            if (next_slice[t] == 0) {
              next_slice[t] = SIZE_MAX;
              completed_tasks++;
            }
            continue;
          }

          size_t total = task.sub_chunks.size();
          if (next_slice[t] >= total) continue;

          // Check if progress allows the next subchunk
          size_t i = next_slice[t];
          bool dep_satisfied = true;
          for (const auto& dep : task.deps) {
            if (state_manager_->getChunkProgress(dep.numa_node, dep.chunk_idx) < i + 1) {
              dep_satisfied = false;
              break;
            }
          }

          if (dep_satisfied) {
            const H2DSubChunk& sc = task.sub_chunks[i];

            devStream& stream = h2d_streams_[h2d_stream_rr_];
            h2d_stream_rr_ = (h2d_stream_rr_ + 1) % PCCL_TRANSFER_STREAM_COUNT;

            devStatus ret = devMemcpyAsync(sc.dev_dst, sc.host_src, sc.size, MemcpyDirection::H2D, stream);
            if (ret != devSuccess) {
              LOG_ERROR("Rank %d: H2D task %zu slice %zu memcpyAsync failed, status=%d", my_rank_, t, i, ret);
              error_state_.store(static_cast<int>(pcclDeviceError));
              next_slice[t] = total;  // Mark as done to avoid infinite loop
              completed_tasks++;
              continue;
            }
            next_slice[t] = i + 1;

            if (next_slice[t] >= total) {
              completed_tasks++;
              LOG_DEBUG("Rank %d: H2D task [%d->%d] all slices submitted (%zu)", my_rank_, task.src_chunk_idx,
                        task.dst_chunk_idx, total);
            }
          }
        }
      }
      }  // NvtxRangeGuard H2D_RoundRobin

      // Wait for all H2D work to complete across all streams
      PCCL_NVTX_RANGE_PUSH("H2D_EvSync", PCCL_NVTX_COLOR_SPIN_WAIT);
      for (int s = 0; s < PCCL_TRANSFER_STREAM_COUNT; s++) {
        devRecordEvent(completion_events[s], h2d_streams_[s]);
      }
      for (int s = 0; s < PCCL_TRANSFER_STREAM_COUNT; s++) {
        devEventStatus status;
        while (true) {
          devStatus ret = devQueryEvent(completion_events[s], &status);
          if (ret != devSuccess) {
            LOG_ERROR("Rank %d: H2D completion event query failed for stream %d, status=%d", my_rank_, s, ret);
            break;
          }
          if (status == devEventStatusComplete) {
            break;
          }
        }
      }
      PCCL_NVTX_RANGE_POP();  // H2D_EvSync

      LOG_DEBUG("Rank %d: H2D round-robin complete, all transfers done", my_rank_);

#if LOG_LEVEL <= 0
      // Sync all H2D streams to streams_[0] before recording end event
      for (int s = 1; s < PCCL_TRANSFER_STREAM_COUNT; s++) {
        devEvent sync_ev;
        devCreateSyncEvent(&sync_ev);
        devRecordEvent(sync_ev, h2d_streams_[s]);
        devStreamWaitEvent(h2d_streams_[0], sync_ev);
        devDestroyEvent(sync_ev);
      }
      devRecordEvent(record.end_event, h2d_streams_[0]);
      {
        std::lock_guard<std::mutex> lock(profiling_mutex_);
        profiling_records_.push_back(record);
      }
#endif
    }

    // Global barrier: wait for all ranks to complete their collective communication
    {
      // Wait for this rank's D2H batch to complete before entering global barrier
      PCCL_NVTX_RANGE_PUSH("WaitD2H", PCCL_NVTX_COLOR_SPIN_WAIT);
      while (d2h_batch_pending_.load(std::memory_order_acquire)) {
      }
      PCCL_NVTX_RANGE_POP();  // WaitD2H

      // Completion barrier: all ranks synchronize after finishing their transfers.
      // Uses generation-based barrier (safe for repeated use, no reset needed).
      PCCL_NVTX_RANGE_PUSH("Barrier1", PCCL_NVTX_COLOR_BARRIER);
      state_manager_->barrier();
      PCCL_NVTX_RANGE_POP();  // Barrier1
      LOG_DEBUG("Rank %d: All ranks completed transfers", my_rank_);

      // Post-barrier cleanup: reset shared state for the next iteration.
      // All ranks are synchronized here, so resetting is race-free.
      // This must happen BEFORE signalH2DComplete() because the user may call
      // submit() again as soon as devSynchronizeStream() returns.
      PCCL_NVTX_RANGE_PUSH("Cleanup", PCCL_NVTX_COLOR_COMPUTE);
      mem_op_manager_->drainAndResetPipelineState();
      state_manager_->resetNUMANodeChunks(my_numa_node_);
      PCCL_NVTX_RANGE_POP();  // Cleanup
      LOG_DEBUG("Rank %d: Post-barrier cleanup done (NUMA %d chunks reset, pipeline state drained)", my_rank_,
                my_numa_node_);

      // Second barrier: ensure all ranks finished cleanup before signaling completion.
      // Without this, a fast rank could signal, the user could call submit() and
      // enqueue new D2H work, and a slow rank's resetNUMANodeChunks() would clobber
      // the fresh progress being written by the new D2H.
      PCCL_NVTX_RANGE_PUSH("Barrier2", PCCL_NVTX_COLOR_BARRIER);
      state_manager_->barrier();
      PCCL_NVTX_RANGE_POP();  // Barrier2
      LOG_DEBUG("Rank %d: Post-cleanup barrier passed", my_rank_);

      // Signal global H2D completion flag after barrier passes
      PCCL_NVTX_RANGE_PUSH("SignalDone", PCCL_NVTX_COLOR_COMPUTE);
      sync_flag_manager_->signalH2DComplete(0);
      PCCL_NVTX_RANGE_POP();  // SignalDone
      LOG_DEBUG("Rank %d: Signaled global H2D completion after barrier", my_rank_);
    }
  }

  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    devDestroyEvent(completion_events[i]);
  }
  LOG_DEBUG("Rank %d: H2D executor thread stopped", my_rank_);
}

// Enqueue D2D operation
pcclResult_t Scheduler::enqueueD2D(const Instruction& inst, void* dev_sendbuff, void* dev_recvbuff, size_t chunk_size) {
  void* dev_src = static_cast<char*>(dev_sendbuff) + (inst.src_chunk_idx * chunk_size);
  void* dev_dst = static_cast<char*>(dev_recvbuff) + (inst.dst_chunk_idx * chunk_size);

  devStatus ret = devMemcpyAsync(dev_dst, dev_src, chunk_size, MemcpyDirection::D2D, d2d_stream_);
  if (ret != devSuccess) {
    LOG_ERROR("Rank %d: D2D memcpyAsync failed, status=%d", my_rank_, ret);
    return pcclDeviceError;
  }

  LOG_DEBUG("Rank %d: Enqueued D2D[%d->%d] on d2d_stream (dev_src=%p, dev_dst=%p, size=%zu)", my_rank_,
            inst.src_chunk_idx, inst.dst_chunk_idx, dev_src, dev_dst, chunk_size);
  return pcclSuccess;
}

// Enqueue WaitCompletionKernel on user_stream
pcclResult_t Scheduler::enqueueWaitCompletionKernel(void* flag_ptr, devStream user_stream) {
  WaitCompletionKernelArgs args;
  args.flag_ptr = flag_ptr;

  devStatus ret =
      devLaunchKernel(WAIT_COMPLETION_KERNEL_NAME, WAIT_COMPLETION_KERNEL_BLOCK_DIM, &args, sizeof(args), user_stream);
  if (ret != devSuccess) {
    LOG_ERROR("Rank %d: Failed to launch WaitCompletionKernel, status=%d", my_rank_, ret);
    return pcclDeviceError;
  }
  return pcclSuccess;
}

// Small-data fast path: bare D2H/H2D, no version tracking, no sync flags,
// no WaitCompletionKernel, no executor threads. Uses devMemcpyAsync with
// batched event sync — submit all transfers to the stream, then sync once.
// This avoids per-call blocking of devMemcpySync and lets the DMA engine
// pipeline setup across multiple queued transfers.
//
// Host pool chunks are spaced at pool_chunk_capacity (e.g. 1MB), not
// chunk_size, so coalescing across chunks is impossible.
pcclResult_t Scheduler::submitSmallData(const IRProgram& program, void* dev_sendbuff, void* dev_recvbuff,
                                        size_t /* count */, size_t chunk_size, devStream /* user_stream */) {
  devStream& ws = d2h_streams_[0];

  // --- Phase 1: D2H (batch async, single sync) ---
  PCCL_NVTX_RANGE_PUSH("FastD2H", PCCL_NVTX_COLOR_D2H_EXEC);
  bool has_d2h = false;
  for (const auto& inst : program.instructions) {
    if (inst.op != OpCode::D2H) continue;
    devMemcpyAsync(host_mem_pool_->getLocalChunk(inst.dst_chunk_idx),
                   static_cast<char*>(dev_sendbuff) + inst.src_chunk_idx * chunk_size, chunk_size,
                   MemcpyDirection::D2H, ws);
    has_d2h = true;
  }
  if (has_d2h) {
    devRecordEvent(fast_sync_ev_[0], ws);
    devEventStatus st;
    do {
      devQueryEvent(fast_sync_ev_[0], &st);
    } while (st != devEventStatusComplete);
  }
  PCCL_NVTX_RANGE_POP();  // FastD2H

  // --- Barrier 1: all ranks' D2H data visible in shared host memory ---
  PCCL_NVTX_RANGE_PUSH("FastBarrier1", PCCL_NVTX_COLOR_BARRIER);
  state_manager_->barrier();
  PCCL_NVTX_RANGE_POP();  // FastBarrier1

  // --- Phase 2: H2H / H2H_REDUCE ---
  PCCL_NVTX_RANGE_PUSH("FastH2H", PCCL_NVTX_COLOR_COMPUTE);
  bool has_host_op = false;
  bool reduce_written[MAX_CHUNKS] = {};
  for (const auto& inst : program.instructions) {
    if (inst.op == OpCode::H2H) {
      memcpy(host_mem_pool_->getLocalChunk(inst.dst_chunk_idx),
             host_mem_pool_->getChunkBuffer(inst.src_numa, inst.src_chunk_idx), chunk_size);
      has_host_op = true;
    } else if (inst.op == OpCode::H2H_REDUCE) {
      float* dst = static_cast<float*>(host_mem_pool_->getLocalChunk(inst.dst_chunk_idx));
      const float* src =
          static_cast<const float*>(host_mem_pool_->getChunkBuffer(inst.src_numa, inst.src_chunk_idx));
      if (!reduce_written[inst.dst_chunk_idx]) {
        memcpy(dst, src, chunk_size);
        reduce_written[inst.dst_chunk_idx] = true;
      } else {
        hostMemcpyAdd(dst, src, chunk_size / sizeof(float));
      }
      has_host_op = true;
    }
  }
  PCCL_NVTX_RANGE_POP();  // FastH2H

  // --- Barrier 2 (AllReduce only): ensure cross-rank reduce outputs visible ---
  if (has_host_op) {
    PCCL_NVTX_RANGE_PUSH("FastBarrier2", PCCL_NVTX_COLOR_BARRIER);
    state_manager_->barrier();
    PCCL_NVTX_RANGE_POP();  // FastBarrier2
  }

  // --- Phase 3: D2D + H2D (batch async, single sync) ---
  PCCL_NVTX_RANGE_PUSH("FastH2D", PCCL_NVTX_COLOR_H2D_EXEC);
  bool has_dev_work = false;
  for (const auto& inst : program.instructions) {
    if (inst.op == OpCode::D2D) {
      devMemcpyAsync(static_cast<char*>(dev_recvbuff) + inst.dst_chunk_idx * chunk_size,
                     static_cast<char*>(dev_sendbuff) + inst.src_chunk_idx * chunk_size, chunk_size,
                     MemcpyDirection::D2D, ws);
      has_dev_work = true;
    } else if (inst.op == OpCode::H2D) {
      devMemcpyAsync(static_cast<char*>(dev_recvbuff) + inst.dst_chunk_idx * chunk_size,
                     host_mem_pool_->getLocalChunk(inst.src_chunk_idx), chunk_size, MemcpyDirection::H2D, ws);
      has_dev_work = true;
    }
  }
  if (has_dev_work) {
    devRecordEvent(fast_sync_ev_[1], ws);
    devEventStatus st;
    do {
      devQueryEvent(fast_sync_ev_[1], &st);
    } while (st != devEventStatusComplete);
  }
  PCCL_NVTX_RANGE_POP();  // FastH2D

  // --- Barrier 3: safe to reuse shared buffers ---
  PCCL_NVTX_RANGE_PUSH("FastBarrier3", PCCL_NVTX_COLOR_BARRIER);
  state_manager_->barrier();
  PCCL_NVTX_RANGE_POP();  // FastBarrier3

  return pcclSuccess;
}

// Submit a collective communication task
pcclResult_t Scheduler::submit(IRProgram program, void* dev_sendbuff, void* dev_recvbuff, size_t count,
                               devStream user_stream) {
  char nvtx_submit_buf[128];
  snprintf(nvtx_submit_buf, sizeof(nvtx_submit_buf), "Submit R%d %zuel", my_rank_, count);
  NvtxRangeGuard nvtx_submit(nvtx_submit_buf, PCCL_NVTX_COLOR_SUBMIT);
  // Check error state
  pcclResult_t error = static_cast<pcclResult_t>(error_state_.load());
  if (error != pcclSuccess) {
    LOG_ERROR("Rank %d: Cannot submit task, scheduler in error state: %s", my_rank_, pcclGetErrorString(error));
    return error;
  }

  LOG_INFO("Rank %d: Submitting task with %zu instructions, count=%zu", my_rank_, program.instructions.size(), count);

  // Calculate chunk size
  int chunk_count = program.input_chunk_count > 0 ? program.input_chunk_count : program.output_chunk_count;
  if (chunk_count <= 0) {
    LOG_ERROR("Rank %d: Invalid program - both input and output chunk counts are 0", my_rank_);
    return pcclInvalidArgument;
  }
  size_t chunk_size = (count * sizeof(float)) / chunk_count;

  // Validate count > 0 (zero-size chunks produce empty sub_chunk lists,
  // causing the H2D round-robin loop to never complete)
  if (count == 0 || chunk_size == 0) {
    LOG_ERROR("Rank %d: Invalid count=%zu (chunk_size=%zu) - must be > 0", my_rank_, count, chunk_size);
    return pcclInvalidArgument;
  }

  // Validate chunk_size fits within the host memory pool's per-chunk slot.
  // The pool allocates HOST_BUFFER_SIZE_PER_RANK bytes per chunk index;
  // exceeding this would overflow into adjacent chunks' memory.
  size_t pool_chunk_capacity = host_mem_pool_->getChunkSize();
  if (chunk_size > pool_chunk_capacity) {
    LOG_ERROR("Rank %d: chunk_size=%zu exceeds host memory pool capacity=%zu (count=%zu, chunk_count=%d)", my_rank_,
              chunk_size, pool_chunk_capacity, count, chunk_count);
    return pcclInvalidArgument;
  }

  // Small-data fast path: bypass executor threads for chunks ≤ 4MB
  // (normal path's subchunk pipelining only wins above this due to thread overhead)
  // static constexpr size_t FAST_PATH_MAX_CHUNK = 4 * SUBCHUNK_SLICE_SIZE;
  static constexpr size_t FAST_PATH_MAX_CHUNK = 0;
  if (chunk_size <= FAST_PATH_MAX_CHUNK) {
    return submitSmallData(program, dev_sendbuff, dev_recvbuff, count, chunk_size, user_stream);
  }

  // 1. Reset completion flags (per-device, not shared memory)
  if (sync_flag_manager_->resetCompletionFlags() != 0) {
    LOG_ERROR("Rank %d: Failed to reset completion flags", my_rank_);
    return pcclDeviceError;
  }
  // NOTE: transfer_complete_count and chunk versions/progress are now reset
  // in the H2D executor's post-barrier cleanup (end of previous iteration),
  // so they are already clean when a new submit() begins.

  // 2. Build D2HTask queue and submit to D2H executor thread
  {
    PCCL_NVTX_RANGE_PUSH("BuildD2H", PCCL_NVTX_COLOR_COMPUTE);
    std::vector<D2HTask> d2h_batch;

    for (const auto& inst : program.instructions) {
      if (inst.op != OpCode::D2H)
        continue;

      D2HTask task;
      task.src_chunk_idx = inst.src_chunk_idx;
      task.dst_chunk_idx = inst.dst_chunk_idx;
      task.effect_chunk_idx = inst.effects.empty() ? inst.dst_chunk_idx : inst.effects[0].chunk_idx;
      task.numa_node = my_numa_node_;
      task.dev_base = static_cast<char*>(dev_sendbuff) + (inst.src_chunk_idx * chunk_size);
      task.host_base = host_mem_pool_->getLocalChunk(inst.dst_chunk_idx);
      task.chunk_size = chunk_size;

      // Build sub-chunks
      size_t num_slices = (chunk_size + SUBCHUNK_SLICE_SIZE - 1) / SUBCHUNK_SLICE_SIZE;
      task.sub_chunks.reserve(num_slices);
      for (size_t i = 0; i < num_slices; i++) {
        D2HSubChunk sc;
        size_t offset = i * SUBCHUNK_SLICE_SIZE;
        sc.dev_src = static_cast<char*>(task.dev_base) + offset;
        sc.host_dst = static_cast<char*>(task.host_base) + offset;
        sc.size = std::min(SUBCHUNK_SLICE_SIZE, chunk_size - offset);
        sc.slice_idx = static_cast<int>(i);
        task.sub_chunks.push_back(sc);
      }

      d2h_batch.push_back(std::move(task));
    }

    // Signal D2H executor thread
    if (!d2h_batch.empty()) {
      PCCL_NVTX_RANGE_POP();  // BuildD2H
      PCCL_NVTX_RANGE_PUSH("NotifyD2H", PCCL_NVTX_COLOR_NOTIFY);
      d2h_batch_pending_.store(true, std::memory_order_release);
      std::lock_guard<std::mutex> lock(d2h_mutex_);
      d2h_tasks_.insert(d2h_tasks_.end(), std::make_move_iterator(d2h_batch.begin()),
                        std::make_move_iterator(d2h_batch.end()));
      d2h_cv_.notify_one();
      PCCL_NVTX_RANGE_POP();  // NotifyD2H
    } else {
      PCCL_NVTX_RANGE_POP();  // BuildD2H
    }
  }

  // 3. Enqueue D2D operations on dedicated d2d_stream_
  PCCL_NVTX_RANGE_PUSH("EnqD2D", PCCL_NVTX_COLOR_COPY_OUTER);
  bool has_d2d = false;
  for (const auto& inst : program.instructions) {
    if (inst.op == OpCode::D2D) {
      pcclResult_t result = enqueueD2D(inst, dev_sendbuff, dev_recvbuff, chunk_size);
      if (result != pcclSuccess) {
        error_state_.store(static_cast<int>(result));
        return result;
      }
      has_d2d = true;
    }
  }
  PCCL_NVTX_RANGE_POP();  // EnqD2D

  // 4. Compute final_version per chunk from IR and set on pipeline manager
  std::map<int, uint64_t> final_versions;
  for (const auto& inst : program.instructions) {
    if (inst.op == OpCode::H2H || inst.op == OpCode::H2H_REDUCE) {
      for (const auto& eff : inst.effects) {
        uint64_t delta = (eff.version_delta > 0) ? eff.version_delta : 1;
        final_versions[eff.chunk_idx] += delta;
      }
    }
  }
  for (const auto& kv : final_versions) {
    mem_op_manager_->setFinalVersion(kv.first, kv.second);
    LOG_DEBUG("Rank %d: Chunk %d final_version=%lu", my_rank_, kv.first, kv.second);
  }

  // 5. Register H2H tasks with MemOpManager (CPU execution)
  for (const auto& inst : program.instructions) {
    if (inst.op == OpCode::H2H) {
      const float* host_src =
          static_cast<const float*>(host_mem_pool_->getChunkBuffer(inst.src_numa, inst.src_chunk_idx));
      float* host_dst = static_cast<float*>(host_mem_pool_->getLocalChunk(inst.dst_chunk_idx));
      size_t elements_per_chunk = chunk_size / sizeof(float);

      mem_op_manager_->submitH2H(inst, host_src, host_dst, elements_per_chunk, my_numa_node_, inst.dst_chunk_idx);
    }
  }

  // 6. Register H2H_REDUCE tasks with MemOpManager
  for (const auto& inst : program.instructions) {
    if (inst.op == OpCode::H2H_REDUCE) {
      const float* host_src =
          static_cast<const float*>(host_mem_pool_->getChunkBuffer(inst.src_numa, inst.src_chunk_idx));
      float* host_dst = static_cast<float*>(host_mem_pool_->getLocalChunk(inst.dst_chunk_idx));
      size_t elements_per_chunk = chunk_size / sizeof(float);

      mem_op_manager_->submitH2HReduce(inst, host_src, host_dst, elements_per_chunk, my_numa_node_, inst.dst_chunk_idx);
    }
  }

  // 7. Build H2DTask list and submit directly to H2D executor thread
  {
    PCCL_NVTX_RANGE_PUSH("BuildH2D", PCCL_NVTX_COLOR_COMPUTE);
    std::vector<H2DTask> h2d_batch;

    for (const auto& inst : program.instructions) {
      if (inst.op != OpCode::H2D)
        continue;

      H2DTask task;
      task.src_chunk_idx = inst.src_chunk_idx;
      task.dst_chunk_idx = inst.dst_chunk_idx;
      task.completion_flag_chunk_idx = inst.dst_chunk_idx;
      task.host_base = host_mem_pool_->getLocalChunk(inst.src_chunk_idx);
      task.dev_base = static_cast<char*>(dev_recvbuff) + (inst.dst_chunk_idx * chunk_size);
      task.chunk_size = chunk_size;
      task.deps = inst.deps;

      // Build sub-chunks
      size_t num_slices = (chunk_size + SUBCHUNK_SLICE_SIZE - 1) / SUBCHUNK_SLICE_SIZE;
      task.sub_chunks.reserve(num_slices);
      for (size_t i = 0; i < num_slices; i++) {
        H2DSubChunk sc;
        size_t offset = i * SUBCHUNK_SLICE_SIZE;
        sc.host_src = static_cast<char*>(task.host_base) + offset;
        sc.dev_dst = static_cast<char*>(task.dev_base) + offset;
        sc.size = std::min(SUBCHUNK_SLICE_SIZE, chunk_size - offset);
        sc.slice_idx = static_cast<int>(i);
        task.sub_chunks.push_back(sc);
      }

      h2d_batch.push_back(std::move(task));
    }

    if (!h2d_batch.empty()) {
      PCCL_NVTX_RANGE_POP();  // BuildH2D
      PCCL_NVTX_RANGE_PUSH("NotifyH2D", PCCL_NVTX_COLOR_NOTIFY);
      // Push all H2D tasks at once to the executor
      std::lock_guard<std::mutex> lock(h2d_mutex_);
      h2d_tasks_.insert(h2d_tasks_.end(), std::make_move_iterator(h2d_batch.begin()),
                        std::make_move_iterator(h2d_batch.end()));
      h2d_cv_.notify_one();
      PCCL_NVTX_RANGE_POP();  // NotifyH2D
    } else {
      PCCL_NVTX_RANGE_POP();  // BuildH2D
      // No H2D operations — submit a barrier-only task so the executor
      // still participates in the global completion barrier.
      H2DTask barrier_task;
      barrier_task.is_barrier_only = true;
      barrier_task.src_chunk_idx = -1;
      barrier_task.dst_chunk_idx = -1;
      barrier_task.completion_flag_chunk_idx = -1;
      barrier_task.host_base = nullptr;
      barrier_task.dev_base = nullptr;
      barrier_task.chunk_size = 0;
      {
        std::lock_guard<std::mutex> lock(h2d_mutex_);
        h2d_tasks_.push_back(std::move(barrier_task));
      }
      h2d_cv_.notify_one();
      LOG_DEBUG("Rank %d: Submitted barrier-only H2D task (no H2D in program)", my_rank_);
    }
  }

  // 8. Enqueue global WaitCompletionKernel
  // All ranks use H2D completion flag - H2D executor handles the global barrier
  // for both H2D-having and barrier-only (no H2D) ranks.
  {
    PCCL_NVTX_RANGE_PUSH("LaunchWaitKernel", PCCL_NVTX_COLOR_NOTIFY);
    void* flag_ptr = sync_flag_manager_->getDeviceH2DCompletionFlagPtr(0);  // Global flag at index 0
    if (!flag_ptr) {
      LOG_ERROR("Rank %d: Failed to get global H2D completion flag ptr", my_rank_);
      return pcclInternalError;
    }
    pcclResult_t result = enqueueWaitCompletionKernel(flag_ptr, user_stream);
    PCCL_NVTX_RANGE_POP();  // LaunchWaitKernel
    if (result != pcclSuccess) {
      error_state_.store(static_cast<int>(result));
      return result;
    }
    LOG_DEBUG("Rank %d: Enqueued global WaitCompletionKernel for H2D (barrier-based)", my_rank_);
  }

  // 9. Make user_stream wait for d2d_stream_ completion (if any D2D operations)
  if (has_d2d) {
    devEvent d2d_completion_event;
    devStatus ret = devCreateSyncEvent(&d2d_completion_event);
    if (ret != devSuccess) {
      LOG_ERROR("Rank %d: Failed to create D2D completion event, status=%d", my_rank_, ret);
      return pcclDeviceError;
    }
    ret = devRecordEvent(d2d_completion_event, d2d_stream_);
    if (ret != devSuccess) {
      LOG_ERROR("Rank %d: Failed to record D2D completion event, status=%d", my_rank_, ret);
      devDestroyEvent(d2d_completion_event);
      return pcclDeviceError;
    }
    ret = devStreamWaitEvent(user_stream, d2d_completion_event);
    if (ret != devSuccess) {
      LOG_ERROR("Rank %d: Failed to make user_stream wait for D2D, status=%d", my_rank_, ret);
      devDestroyEvent(d2d_completion_event);
      return pcclDeviceError;
    }
    devDestroyEvent(d2d_completion_event);
    LOG_DEBUG("Rank %d: user_stream waiting for d2d_stream completion", my_rank_);
  }

  LOG_INFO("Rank %d: Task submitted successfully (async, %zu instructions enqueued)", my_rank_,
           program.instructions.size());

  return pcclSuccess;
}

// Synchronize internal streams
void Scheduler::synchronizeInternalStreams() {
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    if (d2h_streams_[i].handle) {
      devSynchronizeStream(d2h_streams_[i]);
    }
  }
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    if (h2d_streams_[i].handle) {
      devSynchronizeStream(h2d_streams_[i]);
    }
  }
  if (d2d_stream_.handle) {
    devSynchronizeStream(d2d_stream_);
  }
  LOG_DEBUG("Rank %d: Internal streams synchronized", my_rank_);
}

#if LOG_LEVEL <= 0
void Scheduler::printProfilingStats() {
  if (profiling_records_.empty()) {
    LOG_DEBUG("Rank %d: No profiling records", my_rank_);
    return;
  }

  // Synchronize the internal streams before querying event elapsed times
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    devSynchronizeStream(d2h_streams_[i]);
  }
  for (int i = 0; i < PCCL_TRANSFER_STREAM_COUNT; i++) {
    devSynchronizeStream(h2d_streams_[i]);
  }

  LOG_DEBUG("Rank %d: === D2H/H2D Profiling Statistics ===", my_rank_);

  for (const auto& record : profiling_records_) {
    float elapsed_ms = 0.0f;
    devEventElapsedTime(&elapsed_ms, record.start_event, record.end_event);

    const char* op_name;
    if (record.op_type == ProfilingRecord::D2H) {
      op_name = "D2H";
      double size_gb = static_cast<double>(record.size_bytes) / (1024.0 * 1024.0 * 1024.0);
      double bandwidth_gbps = (elapsed_ms > 0) ? size_gb / (elapsed_ms / 1000.0) : 0.0;
      LOG_DEBUG("Rank %d: %s[%d->%d] size=%zu bytes, time=%.3f ms, bandwidth=%.2f GB/s", my_rank_, op_name,
                record.src_chunk_idx, record.dst_chunk_idx, record.size_bytes, elapsed_ms, bandwidth_gbps);
    } else if (record.op_type == ProfilingRecord::H2D) {
      op_name = "H2D";
      double size_gb = static_cast<double>(record.size_bytes) / (1024.0 * 1024.0 * 1024.0);
      double bandwidth_gbps = (elapsed_ms > 0) ? size_gb / (elapsed_ms / 1000.0) : 0.0;
      LOG_DEBUG("Rank %d: %s[%d->%d] size=%zu bytes, time=%.3f ms, bandwidth=%.2f GB/s", my_rank_, op_name,
                record.src_chunk_idx, record.dst_chunk_idx, record.size_bytes, elapsed_ms, bandwidth_gbps);
    }
  }

  LOG_DEBUG("Rank %d: === End Profiling Statistics ===", my_rank_);
}

void Scheduler::clearProfilingRecords() {
  for (auto& record : profiling_records_) {
    if (record.start_event.handle) {
      devDestroyEvent(record.start_event);
    }
    if (record.end_event.handle) {
      devDestroyEvent(record.end_event);
    }
  }
  profiling_records_.clear();
}
#endif

}  // namespace pccl
