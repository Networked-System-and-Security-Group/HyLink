#include "include/pipeline_reduce_manager.hpp"

#include <algorithm>
#include <thread>
#include <vector>

#include "include/hal/host_rt.hpp"
#include "include/log.hpp"
#include "include/nvtx.hpp"
#include "include/state_manager.hpp"

namespace pccl {

PipelineReduceManager::PipelineReduceManager(StateManager* state_manager) : state_manager_(state_manager) {
  if (!state_manager_) {
    LOG_ERROR("PipelineReduceManager: state_manager cannot be null");
    return;
  }
  LOG_INFO("PipelineReduceManager: Created");
}

PipelineReduceManager::~PipelineReduceManager() {
  LOG_INFO("PipelineReduceManager: Destroyed");
}

PipelineReduceManager::ChunkState* PipelineReduceManager::getOrCreateChunkState(int chunk_idx) {
  std::lock_guard<std::mutex> lock(states_mutex_);
  auto it = chunk_states_.find(chunk_idx);
  if (it != chunk_states_.end()) {
    return it->second.get();
  }
  // Use aligned allocation for ChunkState (required by alignas(64) on thread_rounds)
  void* mem = nullptr;
  if (posix_memalign(&mem, 64, sizeof(ChunkState)) != 0 || !mem) {
    LOG_ERROR("PipelineReduceManager: Failed to allocate aligned ChunkState for chunk %d", chunk_idx);
    return nullptr;
  }
  std::unique_ptr<ChunkState, ChunkStateDeleter> chunk_state(new (mem) ChunkState());
  ChunkState* ptr = chunk_state.get();
  chunk_states_[chunk_idx] = std::move(chunk_state);
  LOG_DEBUG("PipelineReduceManager: Created ChunkState for chunk %d", chunk_idx);
  return ptr;
}

void PipelineReduceManager::resetChunkState(ChunkState* chunk_state) {
  chunk_state->pipeline_depth.store(0, std::memory_order_relaxed);
  chunk_state->initial_version.store(0, std::memory_order_relaxed);
  chunk_state->accumulated_version.store(0, std::memory_order_relaxed);
  for (int i = 0; i < MAX_PIPELINE_DEPTH; i++) {
    for (int j = 0; j < MAX_H2H_THREADS; j++) {
      chunk_state->thread_rounds[i][j].store(0, std::memory_order_relaxed);
    }
    chunk_state->slot_num_threads[i].store(0, std::memory_order_relaxed);
    chunk_state->threads_finished[i].store(0, std::memory_order_relaxed);
  }
}

void PipelineReduceManager::resetAllChunkStates() {
  std::lock_guard<std::mutex> lock(states_mutex_);
  for (auto& kv : chunk_states_) {
    resetChunkState(kv.second.get());
  }
  LOG_DEBUG("PipelineReduceManager: All chunk states reset");
}

void PipelineReduceManager::setFinalVersion(int chunk_idx, uint64_t final_ver) {
  ChunkState* cs = getOrCreateChunkState(chunk_idx);
  cs->final_version = final_ver;
  LOG_DEBUG("PipelineReduceManager: Set final_version=%lu for chunk %d", final_ver, chunk_idx);
}

void PipelineReduceManager::submitChunkCopy(float* dst, const float* src, size_t count, int dst_chunk_idx, int dst_numa,
                                            int num_threads, int src_numa, int src_chunk_idx, uint64_t version_delta) {
  char nvtx_buf[128];
  snprintf(nvtx_buf, sizeof(nvtx_buf), "ChunkCopy c%d n%d->n%d %zuKB", dst_chunk_idx, src_numa, dst_numa,
           count * sizeof(float) / 1024);
  NvtxRangeGuard nvtx_guard(nvtx_buf, PCCL_NVTX_COLOR_COPY_OUTER);

  LOG_DEBUG("PipelineReduceManager: submitChunkCopy (chunk=%d, numa=%d, count=%zu, threads=%d, delta=%lu)",
            dst_chunk_idx, dst_numa, count, num_threads, version_delta);

  ChunkState* chunk_state = getOrCreateChunkState(dst_chunk_idx);
  int pipeline_slot = chunk_state->pipeline_depth.fetch_add(1, std::memory_order_relaxed);

  // First slot resets destination progress and accumulated version
  if (pipeline_slot == 0) {
    state_manager_->setChunkProgress(dst_numa, dst_chunk_idx, 0);
    chunk_state->accumulated_version.store(0, std::memory_order_relaxed);
  }

  // Compute if this slot achieves final version
  uint64_t my_target_ver =
      chunk_state->accumulated_version.fetch_add(version_delta, std::memory_order_acq_rel) + version_delta;
  bool is_final_slot = (my_target_ver == chunk_state->final_version);

  // Version increment BEFORE launching workers
  state_manager_->fetchAndAddChunkVersion(dst_numa, dst_chunk_idx, version_delta);

  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int tid = 0; tid < num_threads; tid++) {
    workers.emplace_back([=]() {
      processSubchunksRoundRobin(chunk_state, tid, num_threads, dst, src, count, pipeline_slot, src_numa, src_chunk_idx,
                                 dst_numa, dst_chunk_idx, is_final_slot, "Copy");
    });
  }

  for (auto& w : workers)
    w.join();

  LOG_DEBUG("PipelineReduceManager: submitChunkCopy completed (chunk=%d, numa=%d)", dst_chunk_idx, dst_numa);
}

void PipelineReduceManager::submitChunkReduce(float* dst, const float* src, size_t count, int dst_chunk_idx,
                                              int dst_numa, int num_threads, int src_numa, int src_chunk_idx,
                                              uint64_t version_delta) {
  char nvtx_buf[128];
  snprintf(nvtx_buf, sizeof(nvtx_buf), "ChunkReduce c%d n%d->n%d %zuKB", dst_chunk_idx, src_numa, dst_numa,
           count * sizeof(float) / 1024);
  NvtxRangeGuard nvtx_guard(nvtx_buf, PCCL_NVTX_COLOR_REDUCE_OUTER);

  LOG_DEBUG("PipelineReduceManager: submitChunkReduce (chunk=%d, numa=%d, count=%zu, threads=%d, delta=%lu)",
            dst_chunk_idx, dst_numa, count, num_threads, version_delta);

  ChunkState* chunk_state = getOrCreateChunkState(dst_chunk_idx);
  int pipeline_slot = chunk_state->pipeline_depth.fetch_add(1, std::memory_order_relaxed);

  if (pipeline_slot == 0) {
    uint64_t current_ver = state_manager_->getChunkVersion(dst_numa, dst_chunk_idx);
    chunk_state->initial_version.store(current_ver, std::memory_order_release);
    state_manager_->setChunkProgress(dst_numa, dst_chunk_idx, 0);
    chunk_state->accumulated_version.store(current_ver, std::memory_order_relaxed);
    LOG_DEBUG("PipelineReduceManager: Slot 0 assigned, initial_version=%lu", current_ver);
  }

  // Compute if this slot achieves final version
  uint64_t my_target_ver =
      chunk_state->accumulated_version.fetch_add(version_delta, std::memory_order_acq_rel) + version_delta;
  bool is_final_slot = (my_target_ver == chunk_state->final_version);

  LOG_DEBUG("PipelineReduceManager: pipeline_slot=%d, my_target_ver=%lu, final_ver=%lu, is_final=%d", pipeline_slot,
            my_target_ver, chunk_state->final_version, is_final_slot);

  // Version increment BEFORE launching workers
  state_manager_->fetchAndAddChunkVersion(dst_numa, dst_chunk_idx, version_delta);

  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int tid = 0; tid < num_threads; tid++) {
    workers.emplace_back([=]() {
      processSubchunksRoundRobin(chunk_state, tid, num_threads, dst, src, count, pipeline_slot, src_numa, src_chunk_idx,
                                 dst_numa, dst_chunk_idx, is_final_slot, "Reduce");
    });
  }

  for (auto& w : workers)
    w.join();

  LOG_DEBUG("PipelineReduceManager: submitChunkReduce completed (chunk=%d, slot=%d)", dst_chunk_idx, pipeline_slot);
}

void PipelineReduceManager::processSubchunksRoundRobin(ChunkState* chunk_state, int thread_id, int num_threads,
                                                       float* dst, const float* src, size_t count, int pipeline_slot,
                                                       int src_numa, int src_chunk_idx, int dst_numa, int dst_chunk_idx,
                                                       bool is_final_slot, const char* op_name) {
  const size_t elements_per_subchunk = SUBCHUNK_SLICE_SIZE / sizeof(float);
  const size_t num_subchunks = (count + elements_per_subchunk - 1) / elements_per_subchunk;
  const size_t num_rounds = (num_subchunks + num_threads - 1) / num_threads;

  bool is_first_write = (pipeline_slot == 0) && (chunk_state->initial_version.load(std::memory_order_acquire) == 0);

  // Thread 0 stores slot_num_threads for inter-slot dependency resolution
  if (thread_id == 0) {
    chunk_state->slot_num_threads[pipeline_slot].store(num_threads, std::memory_order_release);
  }

  // Outer NVTX range: entire function scope for this thread
  char nvtx_outer[128];
  snprintf(nvtx_outer, sizeof(nvtx_outer), "%s t%d/s%d c%d %zu_slices%s", op_name, thread_id, pipeline_slot,
           dst_chunk_idx, num_subchunks, is_final_slot ? " FINAL" : "");
  uint32_t thread_color = (op_name[0] == 'C') ? PCCL_NVTX_COLOR_COPY_THREAD : PCCL_NVTX_COLOR_REDUCE_THREAD;
  NvtxRangeGuard nvtx_outer_guard(nvtx_outer, thread_color);

  for (size_t round = 0; round < num_rounds; round++) {
    size_t subchunk_idx = round * num_threads + thread_id;

    if (subchunk_idx < num_subchunks) {
      // Per-subchunk NVTX range
      char nvtx_sc[128];
      snprintf(nvtx_sc, sizeof(nvtx_sc), "%s s%d c%d sc%zu/%zu", op_name, pipeline_slot, dst_chunk_idx, subchunk_idx,
               num_subchunks);
      PCCL_NVTX_RANGE_PUSH(nvtx_sc, thread_color);

      // Source dependency check: wait for source chunk's progress
      {
        char nvtx_wait[64];
        snprintf(nvtx_wait, sizeof(nvtx_wait), "WaitSrc sc%zu", subchunk_idx);
        PCCL_NVTX_RANGE_PUSH(nvtx_wait, PCCL_NVTX_COLOR_SPIN_WAIT);
        while (state_manager_->getChunkProgress(src_numa, src_chunk_idx) < subchunk_idx + 1) {
        }
        PCCL_NVTX_RANGE_POP();
      }

      // Inter-slot dependency: wait for the specific responsible thread in previous slot
      if (pipeline_slot > 0) {
        char nvtx_slot[64];
        snprintf(nvtx_slot, sizeof(nvtx_slot), "WaitSlot sc%zu", subchunk_idx);
        PCCL_NVTX_RANGE_PUSH(nvtx_slot, PCCL_NVTX_COLOR_SLOT_DEP);
        // Spin-wait until previous slot's thread 0 has stored slot_num_threads
        int prev_nt;
        while ((prev_nt = chunk_state->slot_num_threads[pipeline_slot - 1].load(std::memory_order_acquire)) == 0) {
        }
        int responsible_thread = static_cast<int>(subchunk_idx % prev_nt);
        uint64_t required_rounds = subchunk_idx / prev_nt + 1;
        while (chunk_state->thread_rounds[pipeline_slot - 1][responsible_thread].load(std::memory_order_acquire) <
               required_rounds) {
        }
        PCCL_NVTX_RANGE_POP();
      }

      // Process this subchunk
      size_t offset = subchunk_idx * elements_per_subchunk;
      size_t this_count = std::min(elements_per_subchunk, count - offset);

      {
        char nvtx_comp[64];
        snprintf(nvtx_comp, sizeof(nvtx_comp), "Compute sc%zu", subchunk_idx);
        PCCL_NVTX_RANGE_PUSH(nvtx_comp, PCCL_NVTX_COLOR_COMPUTE);
        if (is_first_write && pipeline_slot == 0) {
          hostMemcpy(dst + offset, src + offset, this_count * sizeof(float));
        } else {
          hostMemcpyAdd(dst + offset, src + offset, this_count);
        }
        PCCL_NVTX_RANGE_POP();
      }

      // Pop per-subchunk range
      PCCL_NVTX_RANGE_POP();
    }

    // Each thread stores its own progress after each round
    chunk_state->thread_rounds[pipeline_slot][thread_id].store(round + 1, std::memory_order_release);

    // Shared memory progress (final slot only): compute min across all threads
    if (is_final_slot) {
      uint64_t min_rounds = round + 1;
      for (int t = 0; t < num_threads; t++) {
        if (t == thread_id) continue;
        uint64_t tr = chunk_state->thread_rounds[pipeline_slot][t].load(std::memory_order_acquire);
        if (tr < min_rounds) min_rounds = tr;
      }
      uint64_t progress =
          std::min(static_cast<uint64_t>(min_rounds * num_threads), static_cast<uint64_t>(num_subchunks));
      state_manager_->advanceChunkProgress(dst_numa, dst_chunk_idx, progress);
    }
  }

  // Final slot: guarantee exact final progress using threads_finished counter
  if (is_final_slot) {
    int done = chunk_state->threads_finished[pipeline_slot].fetch_add(1, std::memory_order_acq_rel) + 1;
    if (done == num_threads) {
      state_manager_->setChunkProgress(dst_numa, dst_chunk_idx, num_subchunks);
    }
  }

  LOG_DEBUG("PipelineReduceManager: processSubchunksRoundRobin done (tid=%d, slot=%d)", thread_id, pipeline_slot);
}

}  // namespace pccl
