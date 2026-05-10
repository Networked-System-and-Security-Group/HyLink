#ifndef PCIECCL_MEM_OP_MANAGER_HPP
#define PCIECCL_MEM_OP_MANAGER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>

#include "include/hal/device_rt.hpp"
#include "include/hal/host_rt.hpp"
#include "include/ir.hpp"
#include "include/pipeline_reduce_manager.hpp"
#include "include/transfer_tasks.hpp"

namespace pccl {

// Forward declarations
class StateManager;
class SyncFlagManager;
class PipelineReduceManager;

//==============================================================================
// H2H Task Operation Type
//==============================================================================

enum class H2HOpType { COPY, REDUCE };

//==============================================================================
// Unified H2H Task Structure (for both copy and reduce)
//==============================================================================

struct H2HTask {
  float* dst;                        // Destination host buffer
  const float* src;                  // Source host buffer
  size_t count;                      // Number of float elements
  int dst_numa;                      // Destination NUMA node
  int dst_chunk_idx;                 // Destination chunk index
  std::vector<DataDependency> deps;  // Dependencies to wait for
  H2HOpType op_type;                 // Operation type: COPY or REDUCE
  int src_numa;                      // Source NUMA node
  int src_chunk_idx;                 // Source chunk index
  uint64_t version_delta;            // Version increment from OutputEffect
};

//==============================================================================
// Memory Operation Manager
//==============================================================================

// MemoryOperationManager: Handles H2H execution
//
// The Monitor thread is responsible for:
// 1. Executing H2H tasks when their dependencies are satisfied
class MemoryOperationManager {
 public:
  explicit MemoryOperationManager(StateManager* state_manager, SyncFlagManager* sync_flag_manager = nullptr,
                                  int device_id = 0);
  ~MemoryOperationManager();

  // Delete copy/move constructors and assignment operators
  MemoryOperationManager(const MemoryOperationManager&) = delete;
  MemoryOperationManager& operator=(const MemoryOperationManager&) = delete;
  MemoryOperationManager(MemoryOperationManager&&) = delete;
  MemoryOperationManager& operator=(MemoryOperationManager&&) = delete;

  // Initialize the manager (start monitor thread)
  // Returns 0 on success, non-zero on failure
  int initialize();

  // Shutdown the manager (stop monitor thread)
  void shutdown();

  // Set the SyncFlagManager (can be set after construction)
  void setSyncFlagManager(SyncFlagManager* sync_flag_manager);

  // Submit H2H task (CPU execution)
  // The task will be executed when all dependencies are satisfied
  void submitH2H(const Instruction& inst, const float* host_src, float* host_dst, size_t count, int dst_numa,
                 int dst_chunk_idx);

  // Submit H2H Reduce task (CPU execution with accumulation)
  // The task will be executed when all dependencies are satisfied
  void submitH2HReduce(const Instruction& inst, const float* host_src, float* host_dst, size_t count, int dst_numa,
                       int dst_chunk_idx);

  // Get number of pending operations (for monitoring/debugging)
  size_t getPendingCount() const;

  // Block until all pending H2H operations complete, then reset pipeline state.
  // Called from H2D executor's post-barrier cleanup to guarantee clean state
  // before the next iteration's submit().
  void drainAndResetPipelineState();

  // Set final version for a chunk on the pipeline manager
  void setFinalVersion(int chunk_idx, uint64_t final_ver);

 private:
  // State manager (non-owning pointer)
  StateManager* state_manager_;

  // Sync flag manager (non-owning pointer)
  SyncFlagManager* sync_flag_manager_;

  // Pipeline reduce manager (owning pointer)
  std::unique_ptr<PipelineReduceManager> pipeline_mgr_;

  // Device ID for setting context in monitor thread
  int device_id_;

  // H2H task queue (protected by mutex)
  mutable std::mutex h2h_mutex_;
  std::deque<H2HTask> h2h_tasks_;

  // Monitor thread
  std::thread monitor_thread_;
  std::atomic<bool> stop_monitor_;
  bool initialized_;

  // Counters for pending operations
  std::atomic<size_t> pending_h2h_count_;

  // Monitor thread function
  void monitorThreadFunc();

  // Check if all dependencies are satisfied
  bool checkDependencies(const std::vector<DataDependency>& deps);

  // Check if a single dependency is satisfied
  bool checkDependency(const DataDependency& dep);
};

}  // namespace pccl

#endif  // PCIECCL_MEM_OP_MANAGER_HPP
