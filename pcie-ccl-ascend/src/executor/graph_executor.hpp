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
#include "../runtime/ascend_runtime.hpp"
#include "../ir/compiled_graph.hpp"
#include "../memory/completion_flag.hpp"
#include "event_ring.hpp"
#include "pccl/types.h"

namespace pccl {

class SharedStateManager;
class H2HEngine;
struct LocalTopology;

class GraphExecutor {
 public:
  GraphExecutor(int rank, int world_size, int device_id, SharedStateManager* state_mgr,
                CompletionFlag* completion_flag, H2HEngine* h2h_engine = nullptr, int comm_id = 0,
                const LocalTopology* topo = nullptr);
  ~GraphExecutor();

  GraphExecutor(const GraphExecutor&) = delete;
  GraphExecutor& operator=(const GraphExecutor&) = delete;

  pcclResult_t submit(const CompiledGraph& graph, const RuntimeBinding& binding, devStream user_stream,
                      CollectiveDebugOp op = CollectiveDebugOp::Unknown, size_t count = 0, size_t dtype = 0,
                      const void* comm = nullptr);
  bool drain();
  void synchronizeInternalStreams();

  // True while a collective is queued or actively being processed by the executor
  // thread. Used by host-side wait shims to decide whether to drain.
  bool hasInflight();

 private:
#ifndef PCCL_DMA_LANES_PER_DIRECTION
#define PCCL_DMA_LANES_PER_DIRECTION 1
#endif
  static constexpr int DMA_LANES_PER_DIRECTION =
      PCCL_DMA_LANES_PER_DIRECTION < 1 ? 1 : PCCL_DMA_LANES_PER_DIRECTION;
  static constexpr int DMA_LANE_D2H_BASE = 0;
  static constexpr int DMA_LANE_H2D_BASE = DMA_LANE_D2H_BASE + DMA_LANES_PER_DIRECTION;
  static constexpr int DMA_LANE_D2D_LOCAL = DMA_LANE_H2D_BASE + DMA_LANES_PER_DIRECTION;
  static constexpr int DMA_LANE_D2D_REDUCE = DMA_LANE_D2D_LOCAL + 1;
  static constexpr int DMA_LANE_COUNT = DMA_LANE_D2D_REDUCE + 1;

  struct DmaSlotInfo {
    size_t node_idx = 0;
    size_t subchunk_idx = 0;
    size_t size = 0;
    int lane_idx = 0;
    int64_t submit_ts = 0;
    bool source_reader_pending = false;
  };

  struct NodeState {
    size_t next_subchunk = 0;
    size_t completed_subchunks = 0;
    bool h2h_range_submitted = false;
  };

  void executorThreadFunc();
  void runGraph();
  void pollDeviceEvents(bool block);
  void pollDeviceLane(int lane_idx, bool block);
  size_t pollH2HCompletions();
  pcclResult_t waitForUserReadyAndPublishInputs();
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
  int chooseActiveHostDmaLanes() const;

  int rank_;
  int world_size_;
  int device_id_;
  int comm_id_;
  SharedStateManager* state_mgr_;
  CompletionFlag* completion_flag_;
  H2HEngine* h2h_engine_;
  const LocalTopology* topo_ = nullptr;

  devStream streams_[DMA_LANE_COUNT];
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
  // kept as an event-driven runtime diagnostic knob for cold-context issues.
  // See coldStartDrainFromEnv() in graph_executor.cpp for the rationale.
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
  int active_dma_lanes_per_direction_ = 1;
  bool d2h_lead_window_enabled_ = false;
  bool initial_inputs_published_ = false;
  std::vector<NodeState> node_states_;
  std::vector<std::vector<uint8_t>> h2h_completion_pending_;
  std::vector<uint64_t> d2h_lead_block_log_signature_;
};

}  // namespace pccl

#endif /* PCCL_GRAPH_EXECUTOR_HPP */
