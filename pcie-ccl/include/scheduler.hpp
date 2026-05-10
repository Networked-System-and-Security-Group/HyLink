#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "error.hpp"
#include "hal/device_rt.hpp"
#include "ir.hpp"
#include "log.hpp"
#include "transfer_tasks.hpp"

#ifndef PCCL_TRANSFER_STREAM_COUNT
#define PCCL_TRANSFER_STREAM_COUNT 1
#endif

namespace pccl {

#if LOG_LEVEL <= 0
// Record for one D2H/H2D operation's profiling data (debug mode only)
struct ProfilingRecord {
  enum OpType { D2H, H2D };
  OpType op_type;
  int src_chunk_idx;
  int dst_chunk_idx;
  size_t size_bytes;
  devEvent start_event;
  devEvent end_event;
};
#endif

// Forward declarations
class StateManager;
class MemoryOperationManager;
class HostMemoryPool;
class SyncFlagManager;

// Scheduler: Manages collective communication task submission
//
// Architecture:
// - D2H: Dedicated executor thread processes D2HTask queue, uses events to track
//   per-slice completion, updates progress from CPU side
// - H2D: Dedicated executor thread round-robin monitors chunk progress and issues
//   memcpy as data becomes available (subchunk-level pipelining across all chunks)
// - D2D: Enqueued on dedicated d2d_stream_ for parallelism with D2H/H2D
// - Sync: WaitCompletionKernel on user_stream polls CPU-written completion flags
//
// PCIe full-duplex parallelism:
// - D2H executor thread uses d2h_stream_ for D2H transfers
// - H2D executor thread uses h2d_stream_ for H2D transfers
// - D2D uses d2d_stream_ for device-to-device copies
// - All run concurrently with user_stream
class Scheduler {
 private:
  // Basic information
  int my_rank_;
  int world_size_;
  int my_numa_node_;
  int device_id_;  // For setting device context in executor threads

  // Non-owning pointers to dependencies
  StateManager* state_manager_;
  MemoryOperationManager* mem_op_manager_;
  HostMemoryPool* host_mem_pool_;
  SyncFlagManager* sync_flag_manager_;

  // Error state
  std::atomic<int> error_state_;

  // Internal streams for D2H/H2D/D2D parallelization
  devStream d2h_streams_[PCCL_TRANSFER_STREAM_COUNT];
  devStream h2d_streams_[PCCL_TRANSFER_STREAM_COUNT];
  devStream d2d_stream_;

  size_t d2h_stream_rr_;  // round-robin counter, only accessed by D2H thread
  size_t h2d_stream_rr_;  // round-robin counter, only accessed by H2D thread

  // D2H executor thread infrastructure
  std::vector<D2HTask> d2h_tasks_;
  std::mutex d2h_mutex_;
  std::condition_variable d2h_cv_;
  std::thread d2h_thread_;
  std::atomic<bool> stop_d2h_thread_;

  // H2D executor thread infrastructure
  std::vector<H2DTask> h2d_tasks_;
  std::mutex h2d_mutex_;
  std::condition_variable h2d_cv_;
  std::thread h2d_thread_;
  std::atomic<bool> stop_h2d_thread_;

  // D2H batch completion flag: indicates if D2H tasks are pending for current batch
  // Set to true by submit() when D2H tasks are submitted, set to false by D2H executor when done.
  // H2D executor waits for this to be false before participating in global barrier.
  std::atomic<bool> d2h_batch_pending_;

  // Pre-allocated events for small-data fast path spin-sync (avoids create/destroy per call)
  devEvent fast_sync_ev_[2];

  // D2H executor thread function
  void d2hExecutorThreadFunc();

  // H2D executor thread function
  void h2dExecutorThreadFunc();

#if LOG_LEVEL <= 0
  // Profiling records for D2H/H2D operations (debug mode only)
  std::vector<ProfilingRecord> profiling_records_;
  std::mutex profiling_mutex_;
#endif

  // Private methods for enqueuing operations
  pcclResult_t enqueueD2D(const Instruction& inst, void* dev_sendbuff, void* dev_recvbuff, size_t chunk_size);

  // Enqueue WaitCompletionKernel on user_stream for a completion flag
  pcclResult_t enqueueWaitCompletionKernel(void* flag_ptr, devStream user_stream);

  // Small-data fast path: bypass executor threads entirely for single-subchunk transfers
  pcclResult_t submitSmallData(const IRProgram& program, void* dev_sendbuff, void* dev_recvbuff, size_t count,
                               size_t chunk_size, devStream user_stream);

 public:
  // Constructor with dependency injection
  Scheduler(int my_rank, int world_size, int my_numa_node, StateManager* state_manager,
            MemoryOperationManager* mem_op_manager, HostMemoryPool* host_mem_pool, SyncFlagManager* sync_flag_manager);

  // Destructor
  ~Scheduler();

  // Delete copy and move constructors/operators (non-copyable)
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  // Submit a collective communication task
  // All operations are enqueued to user_stream and returns immediately
  // User calls devSynchronizeStream(user_stream) to wait for completion
  pcclResult_t submit(IRProgram program, void* dev_sendbuff, void* dev_recvbuff, size_t count, devStream user_stream);

  // Synchronize internal D2H/H2D streams
  // This must be called before freeing device memory to ensure internal streams complete
  void synchronizeInternalStreams();

  // Get current error state
  pcclResult_t getError() const { return static_cast<pcclResult_t>(error_state_.load()); }

#if LOG_LEVEL <= 0
  // Print profiling statistics (call after stream sync)
  void printProfilingStats();
  // Clear profiling records for next submission
  void clearProfilingRecords();
#endif
};

}  // namespace pccl

#endif /* SCHEDULER_HPP */
