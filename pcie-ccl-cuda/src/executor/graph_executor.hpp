#ifndef PCCL_GRAPH_EXECUTOR_HPP
#define PCCL_GRAPH_EXECUTOR_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../core/limits.hpp"
#include "../hal/device_rt.hpp"
#include "../ir/compiled_graph.hpp"
#include "../memory/completion_flag.hpp"
#include "event_ring.hpp"
#include "pccl/types.h"

namespace pccl {

class H2HEngine;
class SharedStateManager;

class GraphExecutor {
 public:
  GraphExecutor(int rank, int world_size, int device_id, SharedStateManager* state_mgr,
                H2HEngine* h2h_engine, CompletionFlag* completion_flag, int comm_id = 0);
  ~GraphExecutor();

  GraphExecutor(const GraphExecutor&) = delete;
  GraphExecutor& operator=(const GraphExecutor&) = delete;

  pcclResult_t submit(const CompiledGraph& graph, const RuntimeBinding& binding, devStream user_stream,
                      CollectiveDebugOp op = CollectiveDebugOp::Unknown, size_t count = 0, size_t dtype = 0,
                      const void* comm = nullptr);
  bool drain();
  void synchronizeInternalStreams();

  // True while a collective is queued or actively being processed by the executor
  // thread. Used by the host-blocking CUDA interposers to decide whether to drain.
  bool hasInflight();

 private:
  static constexpr int DMA_LANE_D2H = 0;
  static constexpr int DMA_LANE_H2D = 1;
  static constexpr int DMA_LANE_D2D_PEER = 2;
  static constexpr int DMA_LANE_D2D_LOCAL = 3;
  static constexpr int DMA_LANE_D2D_REDUCE = 4;
  static constexpr int DMA_LANE_COUNT = 5;

  struct DmaSlotInfo {
    size_t node_idx = 0;
    size_t subchunk_idx = 0;
    size_t size = 0;
    int lane_idx = 0;
    int64_t submit_ts = 0;
    bool source_reader_pending = false;
  };

  struct H2HInflight {
    size_t node_idx = 0;
    size_t subchunk_idx = 0;
    std::shared_ptr<std::atomic<int>> done_counter;
    int expected = 0;
  };

  struct NodeState {
    size_t next_subchunk = 0;
    size_t completed_subchunks = 0;
    size_t h2h_pending = 0;
  };

  void executorThreadFunc();
  void runGraph();
  void pollCudaEvents(bool block);
  void pollCudaLane(int lane_idx, bool block);
  void pollH2HCompletions();
  pcclResult_t waitForUserReadyAndPublishInputs();
  pcclResult_t waitForH2HUserReady();
  bool remoteUserReady(int source_rank, int slot, uint64_t work_seq) const;
  bool trySubmitNode(size_t node_idx);
  int selectDmaLane(const InstructionNode& node) const;
  bool graphUsesMultipleHostNumas(const CompiledGraph& graph) const;
  bool inputsReady(const InstructionNode& node, size_t subchunk_idx) const;
  bool overwriteReady(const InstructionNode& node, size_t subchunk_idx) const;
  void publishNodeProgress(const InstructionNode& node, size_t subchunk_idx);
  void publishInputReaders(const InstructionNode& node, size_t subchunk_idx);
  char* resolve(const ProgressKey& key, size_t subchunk_idx, bool ring_buffered) const;
  size_t subchunkSize(const InstructionNode& node, size_t subchunk_idx) const;
  bool terminalsComplete() const;
  void finishCurrentWork(pcclResult_t result);
  pcclResult_t waitForIdle(const char* where);
  void logGraphDebugDump(const char* where) const;
  void logGraphProgress(const char* where) const;
  void logNodeBlockers(size_t node_idx, size_t subchunk_idx) const;

  int rank_;
  int world_size_;
  int device_id_;
  int comm_id_;
  SharedStateManager* state_mgr_;
  H2HEngine* h2h_engine_;
  CompletionFlag* completion_flag_;

  devStream streams_[DMA_LANE_COUNT];
  bool remote_user_ready_seen_[MAX_RANKS] = {};
  EventRingBuffer event_rings_[DMA_LANE_COUNT];
  DmaSlotInfo slot_info_rings_[DMA_LANE_COUNT][EventRingBuffer::DEFAULT_CAPACITY];

  struct WorkItem {
    const CompiledGraph* graph = nullptr;
    RuntimeBinding binding;
    devStream user_stream{nullptr};
    int epoch = 0;
    uint64_t work_seq = 0;
    int slot = 0;
    CollectiveDebugOp op = CollectiveDebugOp::Unknown;
    size_t count = 0;
    size_t dtype = 0;
    const void* comm = nullptr;
  };

  std::thread thread_;
  std::mutex submit_mutex_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::deque<WorkItem> queue_;
  int queue_depth_ = PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  // Leading collectives to submit synchronously (cold-start drain). Defaults off;
  // the cold-context hang is now cured by the init-time warm-up kernel. See
  // coldStartDrainFromEnv() in graph_executor.cpp for the rationale.
  int cold_start_drain_ = 0;
  bool busy_ = false;
  pcclResult_t last_result_ = pcclSuccess;

  const CompiledGraph* graph_ = nullptr;
  RuntimeBinding binding_;
  devStream user_stream_{nullptr};
  int current_epoch_ = 0;
  uint64_t work_seq_counter_ = 0;
  uint64_t current_work_seq_ = 0;
  int current_slot_ = 0;
  CollectiveDebugOp current_op_ = CollectiveDebugOp::Unknown;
  size_t current_count_ = 0;
  size_t current_dtype_ = 0;
  bool d2h_lead_window_enabled_ = false;
  bool initial_inputs_published_ = false;
  std::vector<NodeState> node_states_;
  std::vector<uint64_t> d2h_lead_block_log_signature_;
  std::vector<H2HInflight> h2h_inflight_;
};

}  // namespace pccl

#endif /* PCCL_GRAPH_EXECUTOR_HPP */
