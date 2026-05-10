#include "include/mem_op_manager.hpp"

#include "include/hal/device.hpp"
#include "include/log.hpp"
#include "include/numa_topology.hpp"
#include "include/nvtx.hpp"
#include "include/pipeline_reduce_manager.hpp"
#include "include/state_manager.hpp"
#include "include/sync_flags.hpp"

namespace pccl {

//==============================================================================
// MemoryOperationManager Implementation
//==============================================================================

MemoryOperationManager::MemoryOperationManager(StateManager* state_manager, SyncFlagManager* sync_flag_manager,
                                               int device_id)
    : state_manager_(state_manager),
      sync_flag_manager_(sync_flag_manager),
      device_id_(device_id),
      stop_monitor_(false),
      initialized_(false),
      pending_h2h_count_(0) {
  if (!state_manager_) {
    LOG_ERROR("MemoryOperationManager: state_manager cannot be null");
    return;
  }

  LOG_INFO("MemoryOperationManager: Created");
}

MemoryOperationManager::~MemoryOperationManager() {
  shutdown();
}

int MemoryOperationManager::initialize() {
  if (initialized_) {
    LOG_WARN("MemoryOperationManager: Already initialized");
    return 0;
  }

  LOG_INFO("MemoryOperationManager: Initializing");

  // Initialize pipeline reduce manager
  pipeline_mgr_.reset(new PipelineReduceManager(state_manager_));

  // Start monitor thread
  stop_monitor_.store(false, std::memory_order_relaxed);
  monitor_thread_ = std::thread(&MemoryOperationManager::monitorThreadFunc, this);

  initialized_ = true;
  LOG_INFO("MemoryOperationManager: Initialization complete");
  return 0;
}

void MemoryOperationManager::shutdown() {
  if (!initialized_) {
    return;
  }

  LOG_INFO("MemoryOperationManager: Shutting down");

  // Stop monitor thread
  stop_monitor_.store(true, std::memory_order_release);
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }

  // Log pending operations
  size_t pending = getPendingCount();
  if (pending > 0) {
    LOG_WARN("MemoryOperationManager: %zu pending operations at shutdown", pending);
  }

  initialized_ = false;
  LOG_INFO("MemoryOperationManager: Shutdown complete");
}

void MemoryOperationManager::setSyncFlagManager(SyncFlagManager* sync_flag_manager) {
  sync_flag_manager_ = sync_flag_manager;
}

void MemoryOperationManager::submitH2H(const Instruction& inst, const float* host_src, float* host_dst, size_t count,
                                       int dst_numa, int dst_chunk_idx) {
  if (!initialized_) {
    LOG_ERROR("MemoryOperationManager: Not initialized, cannot submit H2H");
    return;
  }

  H2HTask task;
  task.dst = host_dst;
  task.src = host_src;
  task.count = count;
  task.dst_numa = dst_numa;
  task.dst_chunk_idx = dst_chunk_idx;
  task.deps = inst.deps;
  task.op_type = H2HOpType::COPY;
  task.src_numa = inst.src_numa;
  task.src_chunk_idx = inst.src_chunk_idx;
  // Extract version_delta from output effect
  task.version_delta = 1;
  if (!inst.effects.empty() && inst.effects[0].version_delta > 0) {
    task.version_delta = inst.effects[0].version_delta;
  }

  {
    std::lock_guard<std::mutex> lock(h2h_mutex_);
    h2h_tasks_.push_back(task);
  }
  pending_h2h_count_.fetch_add(1, std::memory_order_relaxed);

  LOG_DEBUG("MemoryOperationManager: Submitted H2H Copy task (dst_numa=%d, dst_chunk=%d, count=%zu, deps=%zu)",
            dst_numa, dst_chunk_idx, count, inst.deps.size());
}

void MemoryOperationManager::submitH2HReduce(const Instruction& inst, const float* host_src, float* host_dst,
                                             size_t count, int dst_numa, int dst_chunk_idx) {
  if (!initialized_) {
    LOG_ERROR("MemoryOperationManager: Not initialized, cannot submit H2H Reduce");
    return;
  }

  if (!pipeline_mgr_) {
    LOG_ERROR("MemoryOperationManager: Pipeline manager not initialized, cannot submit H2H Reduce");
    return;
  }

  H2HTask task;
  task.dst = host_dst;
  task.src = host_src;
  task.count = count;
  task.dst_numa = dst_numa;
  task.dst_chunk_idx = dst_chunk_idx;
  task.deps = inst.deps;
  task.op_type = H2HOpType::REDUCE;
  task.src_numa = inst.src_numa;
  task.src_chunk_idx = inst.src_chunk_idx;
  // Extract version_delta from output effect
  task.version_delta = 1;
  if (!inst.effects.empty() && inst.effects[0].version_delta > 0) {
    task.version_delta = inst.effects[0].version_delta;
  }

  {
    std::lock_guard<std::mutex> lock(h2h_mutex_);
    h2h_tasks_.push_back(task);
  }
  pending_h2h_count_.fetch_add(1, std::memory_order_relaxed);

  LOG_DEBUG("MemoryOperationManager: Submitted H2H Reduce task (dst_numa=%d, dst_chunk=%d, count=%zu, deps=%zu)",
            dst_numa, dst_chunk_idx, count, inst.deps.size());
}

size_t MemoryOperationManager::getPendingCount() const {
  return pending_h2h_count_.load(std::memory_order_acquire);
}

void MemoryOperationManager::drainAndResetPipelineState() {
  // Wait for all detached H2H threads to complete.
  // By this point all compute work is done (H2D depends on reduce progress),
  // so only tail cleanup (counter decrements) remains — the wait is very short.
  while (pending_h2h_count_.load(std::memory_order_acquire) > 0) {
  }
  // Explicitly reset all PipelineReduceManager chunk states so the next
  // iteration's submit() sees clean pipeline_depth / accumulated_version.
  if (pipeline_mgr_) {
    pipeline_mgr_->resetAllChunkStates();
  }
}

void MemoryOperationManager::setFinalVersion(int chunk_idx, uint64_t final_ver) {
  if (pipeline_mgr_) {
    pipeline_mgr_->setFinalVersion(chunk_idx, final_ver);
  }
}

bool MemoryOperationManager::checkDependency(const DataDependency& dep) {
  uint64_t current_ver = state_manager_->getChunkVersion(dep.numa_node, dep.chunk_idx);
  return current_ver >= dep.required_ver;
}

bool MemoryOperationManager::checkDependencies(const std::vector<DataDependency>& deps) {
  for (const auto& dep : deps) {
    if (!checkDependency(dep)) {
      return false;
    }
  }
  return true;
}

void MemoryOperationManager::monitorThreadFunc() {
  LOG_INFO("MemoryOperationManager: Monitor thread started");

  while (!stop_monitor_.load(std::memory_order_acquire)) {
    bool made_progress = false;

    // --- Process H2H tasks (batch async submission) ---
    {
      std::lock_guard<std::mutex> lock(h2h_mutex_);

      // Collect all tasks with satisfied dependencies
      std::vector<H2HTask> ready_tasks;
      auto it = h2h_tasks_.begin();

      while (it != h2h_tasks_.end()) {
        if (checkDependencies(it->deps)) {
          ready_tasks.push_back(*it);
          it = h2h_tasks_.erase(it);  // Remove from queue and get next iterator
        } else {
          ++it;
        }
      }

      // Launch async execution for all ready tasks
      for (const auto& task : ready_tasks) {
        const char* op_name = (task.op_type == H2HOpType::COPY) ? "Copy" : "Reduce";
        LOG_DEBUG("MemoryOperationManager: Async launching H2H %s (dst_numa=%d, chunk=%d, count=%zu)", op_name,
                  task.dst_numa, task.dst_chunk_idx, task.count);

        // Launch async thread for this task
        std::thread([task, this]() {
          const char* op_name = (task.op_type == H2HOpType::COPY) ? "Copy" : "Reduce";

          char nvtx_buf[128];
          snprintf(nvtx_buf, sizeof(nvtx_buf), "H2H_%s n%d_c%d %zuKB", op_name, task.dst_numa, task.dst_chunk_idx,
                   task.count * sizeof(float) / 1024);
          uint32_t nvtx_color =
              (task.op_type == H2HOpType::COPY) ? PCCL_NVTX_COLOR_COPY_OUTER : PCCL_NVTX_COLOR_REDUCE_OUTER;
          NvtxRangeGuard nvtx_guard(nvtx_buf, nvtx_color);

          LOG_DEBUG("MemoryOperationManager: Executing H2H %s (dst_numa=%d, chunk=%d, count=%zu)", op_name,
                    task.dst_numa, task.dst_chunk_idx, task.count);

          constexpr int NUM_H2H_THREADS = 2;
          if (task.op_type == H2HOpType::COPY) {
            pipeline_mgr_->submitChunkCopy(task.dst, task.src, task.count, task.dst_chunk_idx, task.dst_numa,
                                           NUM_H2H_THREADS, task.src_numa, task.src_chunk_idx, task.version_delta);
          } else {
            pipeline_mgr_->submitChunkReduce(task.dst, task.src, task.count, task.dst_chunk_idx, task.dst_numa,
                                             NUM_H2H_THREADS, task.src_numa, task.src_chunk_idx, task.version_delta);
          }

          LOG_DEBUG("MemoryOperationManager: H2H %s completed (dst_numa=%d, chunk=%d)", op_name, task.dst_numa,
                    task.dst_chunk_idx);

          pending_h2h_count_.fetch_sub(1, std::memory_order_release);
        }).detach();

        made_progress = true;
      }
    }

    // If no progress was made, yield to avoid busy spinning
    if (!made_progress) {
    }
  }

  LOG_INFO("MemoryOperationManager: Monitor thread stopped");
}

}  // namespace pccl
