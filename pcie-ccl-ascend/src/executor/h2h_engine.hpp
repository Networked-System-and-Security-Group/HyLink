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

#include "../ir/ir.hpp"

namespace pccl {

class SharedStateManager;
struct LocalTopology;

class H2HEngine {
 public:
  static constexpr size_t kRangeCompletionSubchunk = static_cast<size_t>(-1);

  struct CopyRequest {
    const void* owner = nullptr;
    uint64_t work_seq = 0;
    size_t node_idx = 0;
    size_t subchunk_idx = 0;
    char* dst = nullptr;
    const char* src = nullptr;
    size_t bytes = 0;
    int src_numa = 0;
    int dst_numa = 0;
    bool temporal_write = false;
    int64_t submit_ts = 0;
  };

  struct CopyRangeRequest {
    const void* owner = nullptr;
    uint64_t work_seq = 0;
    size_t node_idx = 0;
    char* dst_base = nullptr;
    const char* src_base = nullptr;
    size_t bytes = 0;
    size_t subchunk_bytes = SUBCHUNK_SLICE_SIZE;
    size_t num_subchunks = 0;
    ProgressKey input_key;
    ProgressKey output_key;
    bool input_ring_buffered = false;
    bool output_ring_buffered = false;
    int input_reader_id = -1;
    bool has_overwrite_guard = false;
    ProgressKey overwrite_guard;
    int comm_id = 0;
    int epoch = 0;
    int src_numa = 0;
    int dst_numa = 0;
    int threads_per_task = 0;
    bool temporal_write = false;
    int64_t submit_ts = 0;
  };

  struct Completion {
    const void* owner = nullptr;
    uint64_t work_seq = 0;
    size_t node_idx = 0;
    size_t subchunk_idx = 0;
    size_t bytes = 0;
    int src_numa = 0;
    int dst_numa = 0;
    int64_t submit_ts = 0;
    int64_t complete_ts = 0;
  };

  H2HEngine(SharedStateManager* state_mgr, const LocalTopology* topo);
  ~H2HEngine();

  H2HEngine(const H2HEngine&) = delete;
  H2HEngine& operator=(const H2HEngine&) = delete;

  int initialize();
  void shutdown();

  int workersPerNuma() const;
  int submitCopy(const CopyRequest& request);
  int submitCopyRange(const CopyRangeRequest& request);
  size_t pollCompletions(const void* owner, uint64_t work_seq, std::vector<Completion>& out,
                         size_t max_completions = 64);

 private:
  static constexpr int kMaxWorkersPerNuma = 32;
  static constexpr int kMaxRangeThreadsPerTask = 16;

  struct RangeSync {
    std::atomic<int> barrier_count{0};
    std::atomic<uint64_t> barrier_generation{0};
    std::atomic<int> threads_done{0};
  };

  struct WorkItem {
    enum class Kind {
      CopyShard,
      CopyRange,
    };

    Kind kind = Kind::CopyShard;
    CopyRequest request;
    CopyRangeRequest range;
    size_t offset = 0;
    size_t bytes = 0;
    int thread_idx = 0;
    int num_threads = 1;
    std::shared_ptr<std::atomic<int>> remaining;
    std::shared_ptr<RangeSync> range_sync;
  };

  struct NumaPool {
    std::deque<WorkItem> queue;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::thread> workers;
    std::once_flag spawn_once;
    std::atomic<int> queue_size{0};
    bool spawned = false;
  };

  void spawnPool(int dst_numa);
  void workerLoop(int dst_numa, int worker_idx);
  void completeWorkItem(const WorkItem& item, int64_t complete_ts);
  void processRangeWorkItem(const WorkItem& item);
  void pushCompletion(const Completion& completion);

  SharedStateManager* state_mgr_;
  const LocalTopology* topo_;
  int num_numa_nodes_;
  int workers_per_numa_;
  size_t shard_bytes_;

  std::array<NumaPool, MAX_NUMA_NODES> pools_;
  mutable std::mutex completions_mutex_;
  std::atomic<size_t> num_completions_{0};
  std::deque<Completion> completions_;

  std::atomic<bool> stop_;
  bool initialized_;
};

}  // namespace pccl

#endif /* PCCL_H2H_ENGINE_HPP */
