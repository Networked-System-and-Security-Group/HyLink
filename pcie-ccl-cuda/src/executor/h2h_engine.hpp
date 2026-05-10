#ifndef PCCL_H2H_ENGINE_HPP
#define PCCL_H2H_ENGINE_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../ir/compiled_graph.hpp"
#include "../ir/ir.hpp"

namespace pccl {

class SharedStateManager;
struct LocalTopology;

constexpr int MAX_H2H_THREADS_PER_TASK = 16;
constexpr int H2H_SIBLING_WORKERS = 0;   // pinned to same-die sibling NUMA
constexpr int H2H_LOCAL_WORKERS   = 16;   // pinned to GPU-local NUMA
constexpr int MAX_H2H_WORKERS_PER_NUMA = H2H_SIBLING_WORKERS + H2H_LOCAL_WORKERS;

class H2HEngine {
 public:
  H2HEngine(SharedStateManager* state_mgr, const LocalTopology* topo);
  ~H2HEngine();

  H2HEngine(const H2HEngine&) = delete;
  H2HEngine& operator=(const H2HEngine&) = delete;

  int initialize();
  void shutdown();

  int submitGraphWork(const InstructionNode& node, const RuntimeBinding& binding, size_t subchunk_idx,
                      size_t chunk_size, size_t bytes, int epoch,
                      std::shared_ptr<std::atomic<int>> done_counter);

 private:
  // Work item: one worker's share of a single graph subchunk.
  struct H2HWorkItem {
    char* dst;
    const char* src1;
    const char* src2;  // nullptr for COPY
    int src1_numa;
    int src2_numa;       // -1 for COPY
    int dst_numa;
    int dst_chunk_idx;
    bool is_copy;
    ScalarDataType scalar_type;
    ReduceOpKind reduce_op;
    size_t element_size;
    int tid;
    int num_threads;
    size_t graph_bytes;
    std::shared_ptr<std::atomic<int>> graph_done;
  };

  // Per-NUMA worker pool (workers created lazily on first submitTasks)
  struct NumaPool {
    std::deque<H2HWorkItem> queue;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::thread> workers;
    std::once_flag spawn_once;  // guards one-time worker creation
    bool spawned = false;       // fast-path check (set after spawn_once completes)
  };

  void workerLoop(int numa_node, int worker_idx);

  void processGraphWork(const H2HWorkItem& item);

  SharedStateManager* state_mgr_;
  const LocalTopology* topo_;
  int num_numa_nodes_;

  std::array<NumaPool, MAX_NUMA_NODES> numa_pools_;

  std::atomic<bool> stop_;
  std::atomic<size_t> pending_count_;
  bool initialized_;
};

}  // namespace pccl

#endif /* PCCL_H2H_ENGINE_HPP */
