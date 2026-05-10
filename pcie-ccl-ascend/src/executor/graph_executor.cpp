#include "graph_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../memory/completion_flag.hpp"
#include "../memory/shared_state.hpp"
#include "../topology/numa.hpp"
#include "../topology/topology.hpp"
#include "dma_lane_policy.hpp"
#include "h2h_engine.hpp"
#include "../util/debug_flags.hpp"
#include "../util/inflight_drain.hpp"
#include "../util/log.hpp"
#include "../util/prof.hpp"
#include "../util/trace_writer.hpp"

namespace {

bool skipDma() {
  static int v = -1;
  if (v < 0) {
    const char* e = std::getenv("PCCL_SKIP_DMA");
    v = (e && std::atoi(e));
  }
  return v > 0;
}

int executorTimeoutMs() {
  const char* e = std::getenv("PCCL_EXECUTOR_TIMEOUT_MS");
  if (!e || e[0] == '\0')
    return 120000;
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed <= 0)
    return 120000;
  return static_cast<int>(parsed);
}

int executorQueueDepthFromEnv() {
  const char* e = std::getenv("PCCL_EXECUTOR_QUEUE_DEPTH");
  if (!e || e[0] == '\0')
    return pccl::PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed <= 0)
    return pccl::PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  return static_cast<int>(std::max<long>(1, std::min<long>(pccl::PCCL_EXECUTOR_QUEUE_DEPTH_MAX, parsed)));
}

// Number of leading collectives that are submitted synchronously (drained to
// completion before submit() returns). This diagnostic knob defaults to off.
int coldStartDrainFromEnv() {
  const char* e = std::getenv("PCCL_COLD_START_DRAIN");
  if (!e || e[0] == '\0')
    return 0;
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed < 0)
    return 0;
  return static_cast<int>(parsed);
}

int forcedHostDmaLanesFromEnv() {
  static int value = -2;
  if (value != -2)
    return value;
  const char* e = std::getenv("PCCL_FORCE_DMA_LANES");
  if (!e || e[0] == '\0') {
    value = -1;
    return value;
  }
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed <= 0) {
    value = -1;
    return value;
  }
  value = static_cast<int>(parsed);
  return value;
}

bool h2hRangeFastPathEnabled() {
  static int enabled = -1;
  if (enabled < 0) {
    // The range scheduler is experimental: its persistent worker-side
    // progress state is not yet safe when an epoch is reused across mixed
    // collective types.  Keep the regular sharded H2H path as the default;
    // it still uses all NUMA workers and supports concurrent communicators.
    const char* e = std::getenv("PCCL_H2H_RANGE_ENABLE");
    enabled = (e && std::atoi(e) != 0) ? 1 : 0;
  }
  return enabled != 0;
}

const char* collectiveDebugOpName(pccl::CollectiveDebugOp op) {
  switch (op) {
    case pccl::CollectiveDebugOp::AllReduce:
      return "AllReduce";
    case pccl::CollectiveDebugOp::AllGather:
      return "AllGather";
    case pccl::CollectiveDebugOp::ReduceScatter:
      return "ReduceScatter";
    case pccl::CollectiveDebugOp::AllToAll:
      return "AllToAll";
    case pccl::CollectiveDebugOp::Broadcast:
      return "Broadcast";
    case pccl::CollectiveDebugOp::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

const char* opName(pccl::OpCode op) {
  switch (op) {
    case pccl::OpCode::NOP:
      return "NOP";
    case pccl::OpCode::D2D:
      return "D2D";
    case pccl::OpCode::D2H:
      return "D2H";
    case pccl::OpCode::H2D:
      return "H2D";
    case pccl::OpCode::H2H:
      return "H2H";
    case pccl::OpCode::D2DReduce:
      return "D2DReduce";
  }
  return "Unknown";
}

const char* spaceName(pccl::ProgressSpace space) {
  switch (space) {
    case pccl::ProgressSpace::HOST:
      return "HOST";
    case pccl::ProgressSpace::INPUT:
      return "INPUT";
    case pccl::ProgressSpace::OUTPUT:
      return "OUTPUT";
    case pccl::ProgressSpace::SCRATCH:
      return "SCRATCH";
    case pccl::ProgressSpace::INTERNAL:
      return "INTERNAL";
  }
  return "Unknown";
}

bool shouldLogProgressKey(const pccl::ProgressKey& key) {
  if (!pccl::debugProgressKeysEnabled())
    return false;
  if (pccl::debugProgressHostOnly() && key.space != pccl::ProgressSpace::HOST)
    return false;
  return pccl::debugProgressChunkSelected(key.chunk);
}

bool graphUsesHostStaging(const pccl::CompiledGraph& graph) {
  for (const auto& key : graph.active_progress_keys) {
    if (key.space == pccl::ProgressSpace::HOST)
      return true;
  }
  return false;
}

}  // namespace

namespace pccl {

GraphExecutor::GraphExecutor(int rank, int world_size, int device_id, SharedStateManager* state_mgr,
                             CompletionFlag* completion_flag, H2HEngine* h2h_engine, int comm_id,
                             const LocalTopology* topo)
    : rank_(rank),
      world_size_(world_size),
      device_id_(device_id),
      comm_id_(comm_id),
      state_mgr_(state_mgr),
      completion_flag_(completion_flag),
      h2h_engine_(h2h_engine),
      topo_(topo) {
  queue_depth_ = completion_flag_ ? completion_flag_->queueDepth() : executorQueueDepthFromEnv();
  cold_start_drain_ = coldStartDrainFromEnv();
  if (devSetDevice(device_id_) != devSuccess) {
    LOG_ERROR("GraphExecutor: devSetDevice failed for device=%d", device_id_);
    last_result_ = pcclDeviceError;
  }
  for (int i = 0; i < DMA_LANE_COUNT; ++i) {
    if (devCreateHighPriorityStream(&streams_[i]) != devSuccess) {
      LOG_ERROR("GraphExecutor: devCreateHighPriorityStream failed for lane %d", i);
      last_result_ = pcclDeviceError;
    }
  }
  for (int i = 0; i < DMA_LANE_COUNT; ++i) {
    if (event_rings_[i].initialize() != 0) {
      last_result_ = pcclDeviceError;
    }
  }
  std::memset(slot_info_rings_, 0, sizeof(slot_info_rings_));
  inflightRegister(this);
  thread_ = std::thread(&GraphExecutor::executorThreadFunc, this);
  LOG_INFO("GraphExecutor: Started (rank=%d device=%d comm_id=%d queue_depth=%d)",
           rank_, device_id_, comm_id_, queue_depth_);
}

GraphExecutor::~GraphExecutor() {
  drain();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable())
    thread_.join();
  for (int i = 0; i < DMA_LANE_COUNT; ++i)
    devDestroyStream(streams_[i]);
  for (int i = 0; i < DMA_LANE_COUNT; ++i)
    event_rings_[i].cleanup();
  inflightUnregister(this);
  LOG_INFO("GraphExecutor: Stopped (rank=%d)", rank_);
}

bool GraphExecutor::hasInflight() {
  std::lock_guard<std::mutex> lock(mutex_);
  return busy_ || !queue_.empty();
}

pcclResult_t GraphExecutor::submit(const CompiledGraph& graph, const RuntimeBinding& binding, devStream user_stream,
                                   CollectiveDebugOp op, size_t count, size_t dtype, const void* comm) {
  std::lock_guard<std::mutex> submit_lock(submit_mutex_);

  if (debugAsyncSubmitEnabled()) {
    bool busy = false;
    size_t queued = 0;
    uint64_t work_seq_counter = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      busy = busy_;
      queued = queue_.size();
      work_seq_counter = work_seq_counter_;
    }
    LOG_INFO("ASYNCDBG executor submit enter rank=%d device=%d graph=%p op=%s nodes=%zu subchunks=%zu "
             "user_stream=%p busy=%d queued=%zu work_seq_counter=%llu",
             rank_,
             device_id_,
             static_cast<const void*>(&graph),
             collectiveDebugOpName(op),
             graph.nodes.size(),
             graph.num_subchunks,
             user_stream.handle,
             busy ? 1 : 0,
             queued,
             static_cast<unsigned long long>(work_seq_counter));
  }

  int64_t submit_t0 = trace::nowMicros();
  uint64_t work_seq = 0;
  int epoch = 0;
  int slot = 0;
  size_t queued_after = 0;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const int timeout_ms = executorTimeoutMs();
    bool ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
      size_t active = busy_ ? 1 : 0;
      return stop_ || last_result_ != pcclSuccess || active + queue_.size() < static_cast<size_t>(queue_depth_);
    });
    if (!ready) {
      LOG_ERROR("GraphExecutor: submit timeout rank=%d queued=%zu busy=%d queue_depth=%d timeout_ms=%d",
                rank_,
                queue_.size(),
                busy_ ? 1 : 0,
                queue_depth_,
                timeout_ms);
      logGraphProgress("submit_queue_full");
      last_result_ = pcclTimeout;
      return pcclTimeout;
    }
    if (stop_)
      return pcclInternalError;
    if (last_result_ != pcclSuccess)
      return last_result_;
    work_seq = ++work_seq_counter_;
    epoch = static_cast<int>(work_seq & 1);
    slot = static_cast<int>((work_seq - 1) % static_cast<uint64_t>(queue_depth_));
    queued_after = queue_.size() + 1;
  }

  if (!completion_flag_) {
    LOG_ERROR("GraphExecutor: completion flag is not initialized");
    return pcclInternalError;
  }
  if (completion_flag_->prepareUserWait(slot, user_stream, op, rank_, work_seq, count, dtype, comm,
                                        user_stream.handle) != 0) {
    return pcclDeviceError;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(WorkItem{&graph, binding, user_stream, epoch, work_seq, slot, op, count, dtype, comm});
  }
  cv_.notify_one();

  if (trace::isActive()) {
    char label[32];
    std::snprintf(label, sizeof(label), "Rank %d Submit", rank_);
    trace::emitComplete(label, "sched", submit_t0, trace::nowMicros() - submit_t0, PCCL_PROF_COLOR_SUBMIT);
  }

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor submit exit rank=%d epoch=%d work_seq=%llu slot=%d op=%s queued=%zu",
             rank_,
             epoch,
             static_cast<unsigned long long>(work_seq),
             slot,
             collectiveDebugOpName(op),
             queued_after);
  }

  // Cold-start drain is diagnostic only and defaults to off.
  if (cold_start_drain_ > 0 && work_seq <= static_cast<uint64_t>(cold_start_drain_)) {
    pcclResult_t drain_result = waitForIdle("cold-start-drain");
    devStatus sync_status = devSuccess;
    if (drain_result == pcclSuccess && user_stream.handle)
      sync_status = devSynchronizeStream(user_stream);
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG executor cold_start_drain rank=%d work_seq=%llu cold_start_drain=%d result=%d sync=%d",
               rank_,
               static_cast<unsigned long long>(work_seq),
               cold_start_drain_,
               static_cast<int>(drain_result),
               static_cast<int>(sync_status));
    }
    if (drain_result != pcclSuccess)
      return drain_result;
    if (sync_status != devSuccess) {
      LOG_ERROR("GraphExecutor: cold-start user-stream sync failed rank=%d work_seq=%llu status=%d",
                rank_, static_cast<unsigned long long>(work_seq), static_cast<int>(sync_status));
      return pcclDeviceError;
    }
  }
  return pcclSuccess;
}

bool GraphExecutor::drain() {
  waitForIdle("drain");
  return last_result_ == pcclSuccess;
}

void GraphExecutor::synchronizeInternalStreams() {
  drain();
}

pcclResult_t GraphExecutor::waitForIdle(const char* where) {
  std::unique_lock<std::mutex> lock(mutex_);
  const int timeout_ms = executorTimeoutMs();
  bool done = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
    return !busy_ && queue_.empty();
  });
  if (!done) {
    LOG_ERROR("GraphExecutor: waitForIdle timeout where=%s rank=%d epoch=%d busy=%d queued=%zu last_result=%d "
              "timeout_ms=%d",
              where ? where : "unknown",
              rank_,
              current_epoch_,
              busy_ ? 1 : 0,
              queue_.size(),
              static_cast<int>(last_result_),
              timeout_ms);
    logGraphProgress(where);
    last_result_ = pcclTimeout;
    return pcclTimeout;
  }
  return last_result_;
}

void GraphExecutor::executorThreadFunc() {
  inflightMarkThreadInternal();
  if (devSetDevice(device_id_) != devSuccess) {
    LOG_ERROR("GraphExecutor: worker devSetDevice failed for device=%d", device_id_);
    finishCurrentWork(pcclDeviceError);
    return;
  }

  if (topo_) {
    int logical = topo_->getNUMAForDevice(device_id_);
    int physical = topo_->getPhysicalNUMA(logical);
    pinThreadToNuma(physical);
  }

  if (trace::isActive()) {
    trace::setThreadTid(trace::rankH2DTid(device_id_));
    char self_name[32];
    std::snprintf(self_name, sizeof(self_name), "Rank %d Graph", device_id_);
    trace::emitThreadName(self_name);
    for (int i = 0; i < DMA_LANE_COUNT; ++i) {
      char name[40];
      std::snprintf(name, sizeof(name), "Rank %d Graph Lane %d", device_id_, i);
      trace::emitThreadName(name, trace::deviceStreamTid(device_id_, 1, i));
    }
  }

  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !queue_.empty() || stop_; });
      if (stop_ && queue_.empty())
        break;
      item = queue_.front();
      queue_.pop_front();
      graph_ = item.graph;
      binding_ = item.binding;
      user_stream_ = item.user_stream;
      current_epoch_ = item.epoch;
      current_work_seq_ = item.work_seq;
      current_slot_ = item.slot;
      current_op_ = item.op;
      current_count_ = item.count;
      current_dtype_ = item.dtype;
      active_dma_lanes_per_direction_ = chooseActiveHostDmaLanes();
      for (const auto& key : graph_->active_progress_keys) {
        state_mgr_->setNumReaders(comm_id_, key, 0);
      }
      for (const auto& chunk : graph_->reader_assignment.chunks) {
        state_mgr_->setNumReaders(comm_id_, chunk.key, chunk.num_readers);
      }
      d2h_lead_window_enabled_ = graphUsesMultipleHostNumas(*graph_);
      initial_inputs_published_ = false;
      node_states_.assign(graph_->nodes.size(), NodeState{});
      h2h_completion_pending_.assign(graph_->nodes.size(), std::vector<uint8_t>(graph_->num_subchunks, 0));
      d2h_lead_block_log_signature_.assign(graph_->nodes.size(), UINT64_MAX);
      busy_ = true;
    }
    cv_.notify_all();
    runGraph();
  }
}

void GraphExecutor::finishCurrentWork(pcclResult_t result) {
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor finishCurrentWork rank=%d epoch=%d work_seq=%llu slot=%d result=%d graph=%p",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             current_slot_,
             static_cast<int>(result),
             static_cast<const void*>(graph_));
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    last_result_ = result;
    graph_ = nullptr;
    node_states_.clear();
    h2h_completion_pending_.clear();
    d2h_lead_block_log_signature_.clear();
    for (int i = 0; i < DMA_LANE_COUNT; ++i)
      event_rings_[i].reset();
    if (result != pcclSuccess) {
      if (completion_flag_ && current_work_seq_ != 0)
        completion_flag_->signalComplete(current_slot_, current_work_seq_);
      for (const auto& item : queue_) {
        if (completion_flag_)
          completion_flag_->signalComplete(item.slot, item.work_seq);
      }
      queue_.clear();
    }
    busy_ = false;
  }
  cv_.notify_all();
}

void GraphExecutor::logGraphDebugDump(const char* where) const {
  if (!graph_)
    return;
  if (!debugProgressKeysEnabled() || !debugProgressGenerationSelected(current_work_seq_))
    return;

  LOG_INFO("PROGDBG executor graph_dump where=%s rank=%d epoch=%d work_seq=%llu nodes=%zu subchunks=%zu "
           "initial_inputs=%zu terminal_outputs=%zu active_keys=%zu reader_chunks=%zu d2h_lead_window=%d",
           where ? where : "unknown",
           rank_,
           current_epoch_,
           static_cast<unsigned long long>(current_work_seq_),
           graph_->nodes.size(),
           graph_->num_subchunks,
           graph_->initial_ready_inputs.size(),
           graph_->terminal_outputs.size(),
           graph_->active_progress_keys.size(),
           graph_->reader_assignment.chunks.size(),
           d2h_lead_window_enabled_ ? 1 : 0);

  for (const auto& key : graph_->initial_ready_inputs) {
    LOG_INFO("PROGDBG executor graph_dump initial rank=%d work_seq=%llu key=(space=%s/%d owner=%d chunk=%d "
             "version=%d)",
             rank_,
             static_cast<unsigned long long>(current_work_seq_),
             spaceName(key.space),
             static_cast<int>(key.space),
             key.owner,
             key.chunk,
             key.version);
  }
  for (const auto& terminal : graph_->terminal_outputs) {
    LOG_INFO("PROGDBG executor graph_dump terminal rank=%d work_seq=%llu key=(space=%s/%d owner=%d chunk=%d "
             "version=%d)",
             rank_,
             static_cast<unsigned long long>(current_work_seq_),
             spaceName(terminal.space),
             static_cast<int>(terminal.space),
             terminal.owner,
             terminal.chunk,
             terminal.version);
  }
  for (const auto& assignment : graph_->reader_assignment.chunks) {
    if (!shouldLogProgressKey(assignment.key))
      continue;
    LOG_INFO("PROGDBG executor graph_dump readers rank=%d work_seq=%llu key=(space=%s/%d owner=%d chunk=%d "
             "version=%d) num_readers=%d ring=%d",
             rank_,
             static_cast<unsigned long long>(current_work_seq_),
             spaceName(assignment.key.space),
             static_cast<int>(assignment.key.space),
             assignment.key.owner,
             assignment.key.chunk,
             assignment.key.version,
             assignment.num_readers,
             assignment.ring_buffered ? 1 : 0);
  }
  for (size_t i = 0; i < graph_->nodes.size(); ++i) {
    const auto& node = graph_->nodes[i];
    bool log_node = shouldLogProgressKey(node.output);
    for (const auto& input : node.inputs)
      log_node = log_node || shouldLogProgressKey(input.key);
    if (!log_node)
      continue;
    LOG_INFO("PROGDBG executor graph_dump node rank=%d work_seq=%llu node=%zu op=%s output=(space=%s/%d owner=%d "
             "chunk=%d version=%d ring=%d) inputs=%zu src_rank=%d dst_rank=%d overwrite_guard=%d",
             rank_,
             static_cast<unsigned long long>(current_work_seq_),
             i,
             opName(node.op),
             spaceName(node.output.space),
             static_cast<int>(node.output.space),
             node.output.owner,
             node.output.chunk,
             node.output.version,
             node.output_ring_buffered ? 1 : 0,
             node.inputs.size(),
             node.source_rank,
             node.destination_rank,
             node.has_overwrite_guard ? 1 : 0);
    for (size_t input_idx = 0; input_idx < node.inputs.size(); ++input_idx) {
      const auto& input = node.inputs[input_idx];
      LOG_INFO("PROGDBG executor graph_dump node_input rank=%d work_seq=%llu node=%zu input_idx=%zu key=(space=%s/%d "
               "owner=%d chunk=%d version=%d) reader=%d ring=%d",
               rank_,
               static_cast<unsigned long long>(current_work_seq_),
               i,
               input_idx,
               spaceName(input.key.space),
               static_cast<int>(input.key.space),
               input.key.owner,
               input.key.chunk,
               input.key.version,
               input.reader_id,
               input.ring_buffered ? 1 : 0);
    }
    if (node.has_overwrite_guard) {
      const auto& key = node.overwrite_guard;
      LOG_INFO("PROGDBG executor graph_dump overwrite_guard rank=%d work_seq=%llu node=%zu key=(space=%s/%d "
               "owner=%d chunk=%d version=%d)",
               rank_,
               static_cast<unsigned long long>(current_work_seq_),
               i,
               spaceName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version);
    }
  }
}

void GraphExecutor::logGraphProgress(const char* where) const {
  if (!graph_) {
    LOG_ERROR("GraphExecutor: progress where=%s rank=%d no active graph", where ? where : "unknown", rank_);
    return;
  }
  LOG_ERROR("GraphExecutor: progress where=%s rank=%d epoch=%d work_seq=%llu slot=%d nodes=%zu subchunks=%zu "
            "initial_inputs=%d last_result=%d",
            where ? where : "unknown",
            rank_,
            current_epoch_,
            static_cast<unsigned long long>(current_work_seq_),
            current_slot_,
            graph_->nodes.size(),
            graph_->num_subchunks,
            initial_inputs_published_ ? 1 : 0,
            static_cast<int>(last_result_));
  for (int lane = 0; lane < DMA_LANE_COUNT; ++lane) {
    const auto& ring = event_rings_[lane];
    if (ring.inflight() == 0)
      continue;
    size_t ring_idx = ring.tailIndex();
    const auto& info = slot_info_rings_[lane][ring_idx];
    LOG_ERROR("GraphExecutor: lane=%d inflight=%zu oldest_slot=%zu node=%zu subchunk=%zu bytes=%zu submit_ts=%lld",
              lane,
              ring.inflight(),
              ring_idx,
              info.node_idx,
              info.subchunk_idx,
              info.size,
              static_cast<long long>(info.submit_ts));
  }
  for (size_t i = 0; i < node_states_.size(); ++i) {
    const auto& state = node_states_[i];
    const auto& node = graph_->nodes[i];
    LOG_ERROR("GraphExecutor: node=%zu op=%s(%d) next=%zu completed=%zu output=(space=%s/%d owner=%d "
              "chunk=%d version=%d)",
              i,
              opName(node.op),
              static_cast<int>(node.op),
              state.next_subchunk,
              state.completed_subchunks,
              spaceName(node.output.space),
              static_cast<int>(node.output.space),
              node.output.owner,
              node.output.chunk,
              node.output.version);
    if (state.completed_subchunks < graph_->num_subchunks) {
      size_t sc = std::min(state.next_subchunk, graph_->num_subchunks - 1);
      logNodeBlockers(i, sc);
    }
  }
  for (const auto& key : graph_->terminal_outputs) {
    uint64_t progress = state_mgr_->getWriteProgress(comm_id_, key, current_epoch_);
    if (progress < graph_->num_subchunks) {
      LOG_ERROR("GraphExecutor: terminal blocked key=(space=%s/%d owner=%d chunk=%d version=%d) progress=%llu "
                "needed=%zu",
                spaceName(key.space),
                static_cast<int>(key.space),
                key.owner,
                key.chunk,
                key.version,
                static_cast<unsigned long long>(progress),
                graph_->num_subchunks);
    }
  }
}

void GraphExecutor::logNodeBlockers(size_t node_idx, size_t subchunk_idx) const {
  if (!graph_ || node_idx >= graph_->nodes.size())
    return;

  const auto& node = graph_->nodes[node_idx];
  if (node_states_[node_idx].next_subchunk >= graph_->num_subchunks) {
    LOG_ERROR("GraphExecutor: node=%zu waiting_for_completion submitted=%zu completed=%zu",
              node_idx,
              node_states_[node_idx].next_subchunk,
              node_states_[node_idx].completed_subchunks);
    return;
  }

  uint64_t needed = static_cast<uint64_t>(subchunk_idx) + 1;
  for (const auto& input : node.inputs) {
    uint64_t progress = state_mgr_->getWriteProgress(comm_id_, input.key, current_epoch_);
    if (progress < needed) {
      LOG_ERROR("GraphExecutor: node=%zu blocked input key=(space=%s/%d owner=%d chunk=%d version=%d) "
                "progress=%llu needed=%llu reader_id=%d ring=%d",
                node_idx,
                spaceName(input.key.space),
                static_cast<int>(input.key.space),
                input.key.owner,
                input.key.chunk,
                input.key.version,
                static_cast<unsigned long long>(progress),
                static_cast<unsigned long long>(needed),
                input.reader_id,
                input.ring_buffered ? 1 : 0);
    }
  }

  if (!node.has_overwrite_guard)
    return;

  uint64_t guard_write = state_mgr_->getWriteProgress(comm_id_, node.overwrite_guard, current_epoch_);
  int readers = state_mgr_->getNumReaders(comm_id_, node.overwrite_guard);
  uint64_t consumed = state_mgr_->getConsumeProgress(comm_id_, node.overwrite_guard, current_epoch_);
  if ((node.overwrite_guard.version > 0 && guard_write < needed) || (readers > 0 && consumed < needed)) {
    LOG_ERROR("GraphExecutor: node=%zu blocked overwrite_guard key=(space=%s/%d owner=%d chunk=%d version=%d) "
              "write=%llu consume=%llu readers=%d needed=%llu",
              node_idx,
              spaceName(node.overwrite_guard.space),
              static_cast<int>(node.overwrite_guard.space),
              node.overwrite_guard.owner,
              node.overwrite_guard.chunk,
              node.overwrite_guard.version,
              static_cast<unsigned long long>(guard_write),
              static_cast<unsigned long long>(consumed),
              readers,
              static_cast<unsigned long long>(needed));
  }
}

size_t GraphExecutor::subchunkSize(const InstructionNode& node, size_t subchunk_idx) const {
  size_t offset = subchunk_idx * SUBCHUNK_SLICE_SIZE;
  if (offset >= node.work_bytes)
    return 0;
  return std::min(SUBCHUNK_SLICE_SIZE, node.work_bytes - offset);
}

char* GraphExecutor::resolve(const ProgressKey& key, size_t subchunk_idx, bool ring_buffered) const {
  const LayoutDesc& layout = binding_.layout[static_cast<int>(key.space)];
  char* base = nullptr;
  if (key.space == ProgressSpace::HOST) {
    if (key.owner < 0 || key.owner >= MAX_NUMA_NODES)
      return nullptr;
    base = binding_.host_base[key.owner];
    if (!base)
      return nullptr;
    const auto& remap = binding_.host_chunk_remap[key.owner];
    if (key.chunk < 0 || static_cast<size_t>(key.chunk) >= remap.size())
      return nullptr;
    const int physical_chunk = remap[static_cast<size_t>(key.chunk)];
    if (physical_chunk < 0)
      return nullptr;
    size_t chunk_offset = static_cast<size_t>(physical_chunk) * graph_->chunk_size;
    size_t sub_offset = ring_buffered ? ((subchunk_idx & 1) * SUBCHUNK_SLICE_SIZE)
                                      : (subchunk_idx * SUBCHUNK_SLICE_SIZE);
    return base + chunk_offset + sub_offset;
  }

  base = binding_.device_base[static_cast<int>(key.space)][key.owner];
  if (!base)
    return nullptr;
  int chunks_per_segment = std::max(1, layout.chunks_per_segment);
  int seg = key.chunk / chunks_per_segment;
  int chunk_in_seg = key.chunk % chunks_per_segment;
  size_t chunk_offset = static_cast<size_t>(seg) * layout.segment_size + layout.stride_offset;
  if (key.chunk >= 0 && static_cast<size_t>(key.chunk) < layout.chunk_offsets.size()) {
    chunk_offset += layout.chunk_offsets[static_cast<size_t>(key.chunk)];
  } else {
    chunk_offset += static_cast<size_t>(chunk_in_seg) * layout.chunk_size;
  }
  return base + chunk_offset + subchunk_idx * SUBCHUNK_SLICE_SIZE;
}

bool GraphExecutor::inputsReady(const InstructionNode& node, size_t subchunk_idx) const {
  for (const auto& input : node.inputs) {
    if (state_mgr_->getWriteProgress(comm_id_, input.key, current_epoch_) < subchunk_idx + 1)
      return false;
  }
  return true;
}

bool GraphExecutor::overwriteReady(const InstructionNode& node, size_t subchunk_idx) const {
  if (!node.has_overwrite_guard)
    return true;
  if (node.overwrite_guard.version > 0 &&
      state_mgr_->getWriteProgress(comm_id_, node.overwrite_guard, current_epoch_) < subchunk_idx + 1) {
    return false;
  }
  if (state_mgr_->getNumReaders(comm_id_, node.overwrite_guard) == 0)
    return true;
  return state_mgr_->getConsumeProgress(comm_id_, node.overwrite_guard, current_epoch_) >= subchunk_idx + 1;
}

void GraphExecutor::publishInputReaders(const InstructionNode& node, size_t subchunk_idx) {
  for (const auto& input : node.inputs) {
    if (input.reader_id >= 0) {
      if (shouldLogProgressKey(input.key) && debugProgressGenerationSelected(current_work_seq_)) {
        LOG_INFO("PROGDBG executor publish_reader rank=%d epoch=%d work_seq=%llu op=%s input_key=(space=%s/%d "
                 "owner=%d chunk=%d version=%d) reader=%d sc=%zu new=%zu",
                 rank_,
                 current_epoch_,
                 static_cast<unsigned long long>(current_work_seq_),
                 opName(node.op),
                 spaceName(input.key.space),
                 static_cast<int>(input.key.space),
                 input.key.owner,
                 input.key.chunk,
                 input.key.version,
                 input.reader_id,
                 subchunk_idx,
                 subchunk_idx + 1);
      }
      state_mgr_->advanceReaderDone(comm_id_, input.key, input.reader_id, subchunk_idx + 1, current_epoch_);
    }
  }
}

void GraphExecutor::publishNodeProgress(const InstructionNode& node, size_t subchunk_idx) {
  if (shouldLogProgressKey(node.output) && debugProgressGenerationSelected(current_work_seq_)) {
    LOG_INFO("PROGDBG executor publish_write rank=%d epoch=%d work_seq=%llu op=%s output_key=(space=%s/%d owner=%d "
             "chunk=%d version=%d) sc=%zu new=%zu",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             opName(node.op),
             spaceName(node.output.space),
             static_cast<int>(node.output.space),
             node.output.owner,
             node.output.chunk,
             node.output.version,
             subchunk_idx,
             subchunk_idx + 1);
  }
  state_mgr_->advanceWriteProgress(comm_id_, node.output, subchunk_idx + 1, current_epoch_);
}

pcclResult_t GraphExecutor::waitForUserReadyAndPublishInputs() {
  if (initial_inputs_published_)
    return pcclSuccess;

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor wait_user_ready begin rank=%d epoch=%d work_seq=%llu slot=%d ready_seq=%llu",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             current_slot_,
             static_cast<unsigned long long>(completion_flag_ ? completion_flag_->readySequence(current_slot_)
                                                              : 0));
  }

  if (!completion_flag_) {
    LOG_ERROR("GraphExecutor: completion flag is not initialized");
    return pcclInternalError;
  }

  const int timeout_ms = executorTimeoutMs();
  auto start = std::chrono::steady_clock::now();
  uint64_t spins = 0;
  while (!completion_flag_->isReady(current_slot_, current_work_seq_)) {
    if ((++spins & 0x3ff) == 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (elapsed_ms > timeout_ms) {
        LOG_ERROR("GraphExecutor: user-ready doorbell timeout rank=%d epoch=%d work_seq=%llu slot=%d ready_seq=%llu "
                  "timeout_ms=%d",
                  rank_,
                  current_epoch_,
                  static_cast<unsigned long long>(current_work_seq_),
                  current_slot_,
                  static_cast<unsigned long long>(completion_flag_->readySequence(current_slot_)),
                  timeout_ms);
        logGraphProgress("user-ready-doorbell");
        return pcclTimeout;
      }
      std::this_thread::yield();
    } else {
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
      _mm_pause();
#endif
    }
  }

  state_mgr_->publishUserReadySeq(comm_id_, rank_, current_slot_, current_work_seq_);

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor publish_initial_inputs rank=%d epoch=%d work_seq=%llu slot=%d ready_seq=%llu",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             current_slot_,
             static_cast<unsigned long long>(completion_flag_->readySequence(current_slot_)));
  }

  for (const auto& key : graph_->initial_ready_inputs) {
    if (shouldLogProgressKey(key) && debugProgressGenerationSelected(current_work_seq_)) {
      LOG_INFO("PROGDBG executor initial_input rank=%d epoch=%d work_seq=%llu key=(space=%s/%d owner=%d chunk=%d "
               "version=%d) value=%zu",
               rank_,
               current_epoch_,
               static_cast<unsigned long long>(current_work_seq_),
               spaceName(key.space),
               static_cast<int>(key.space),
               key.owner,
               key.chunk,
               key.version,
               graph_->num_subchunks);
    }
    state_mgr_->setWriteProgress(comm_id_, key, graph_->num_subchunks, current_epoch_);
  }
  initial_inputs_published_ = true;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor wait_user_ready done rank=%d epoch=%d work_seq=%llu initial_inputs=%zu",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             graph_->initial_ready_inputs.size());
  }
  return pcclSuccess;
}

void GraphExecutor::pollDeviceEvents(bool block) {
  for (int lane = 0; lane < DMA_LANE_COUNT; ++lane)
    pollDeviceLane(lane, block);
}

size_t GraphExecutor::pollH2HCompletions() {
  if (!h2h_engine_)
    return 0;
  std::vector<H2HEngine::Completion> completions;
  size_t count = h2h_engine_->pollCompletions(this, current_work_seq_, completions);
  for (const auto& completion : completions) {
    if (!graph_ || completion.node_idx >= graph_->nodes.size())
      continue;
    if (completion.subchunk_idx == H2HEngine::kRangeCompletionSubchunk) {
      auto& state = node_states_[completion.node_idx];
      state.completed_subchunks = graph_->num_subchunks;
      state.next_subchunk = graph_->num_subchunks;
      if (trace::isActive()) {
        trace::TraceArg args[] = {
            trace::TraceArg::Int("bytes", static_cast<int64_t>(completion.bytes)),
            trace::TraceArg::Int("src_numa", completion.src_numa),
            trace::TraceArg::Int("dst_numa", completion.dst_numa),
        };
        trace::emitComplete("Graph H2H Range", "h2h", completion.submit_ts,
                            completion.complete_ts - completion.submit_ts,
                            PCCL_PROF_COLOR_COPY_OUTER, 0, args, 3);
      }
      continue;
    }
    if (completion.node_idx >= h2h_completion_pending_.size() ||
        completion.subchunk_idx >= h2h_completion_pending_[completion.node_idx].size())
      continue;
    h2h_completion_pending_[completion.node_idx][completion.subchunk_idx] = 1;
    if (trace::isActive()) {
      trace::TraceArg args[] = {
          trace::TraceArg::Int("bytes", static_cast<int64_t>(completion.bytes)),
          trace::TraceArg::Int("src_numa", completion.src_numa),
          trace::TraceArg::Int("dst_numa", completion.dst_numa),
      };
      trace::emitComplete("Graph H2H", "h2h", completion.submit_ts,
                          completion.complete_ts - completion.submit_ts,
                          PCCL_PROF_COLOR_COPY_OUTER, 0, args, 3);
    }

    auto& state = node_states_[completion.node_idx];
    auto& pending = h2h_completion_pending_[completion.node_idx];
    while (state.completed_subchunks < graph_->num_subchunks &&
           state.completed_subchunks < pending.size() &&
           pending[state.completed_subchunks]) {
      size_t publish_sc = state.completed_subchunks;
      pending[publish_sc] = 0;
      const auto& node = graph_->nodes[completion.node_idx];
      publishNodeProgress(node, publish_sc);
      publishInputReaders(node, publish_sc);
      state.completed_subchunks++;
    }
  }
  return count;
}

void GraphExecutor::pollDeviceLane(int lane_idx, bool block) {
  auto& ring = event_rings_[lane_idx];
  auto* slot_info = slot_info_rings_[lane_idx];
  auto block_start = std::chrono::steady_clock::now();
  const int timeout_ms = executorTimeoutMs();
  while (ring.inflight() > 0) {
    devEvent oldest = ring.peekOldest();
    devEventStatus status;
    devStatus query_status = devQueryEvent(oldest, &status);
    if (query_status != devSuccess) {
      LOG_ERROR("GraphExecutor: devQueryEvent failed lane=%d status=%d", lane_idx, static_cast<int>(query_status));
      ring.releaseOldest();
      finishCurrentWork(pcclDeviceError);
      return;
    }
    if (status == devEventStatusComplete) {
      size_t ring_idx = ring.tailIndex();
      ring.releaseOldest();
      auto& info = slot_info[ring_idx];
      const auto& node = graph_->nodes[info.node_idx];
      publishNodeProgress(node, info.subchunk_idx);
      if (info.source_reader_pending)
        publishInputReaders(node, info.subchunk_idx);
      node_states_[info.node_idx].completed_subchunks++;

      if (trace::isActive()) {
        int64_t now = trace::nowMicros();
        trace::TraceArg args[] = {
            trace::TraceArg::Int("bytes", static_cast<int64_t>(info.size)),
            trace::TraceArg::Int("submit_ts", info.submit_ts),
            trace::TraceArg::Int("lane", info.lane_idx),
        };
        trace::emitComplete("Graph DMA", "device_dma", info.submit_ts, now - info.submit_ts,
                            PCCL_PROF_COLOR_H2D_EXEC, trace::deviceStreamTid(device_id_, 1, info.lane_idx),
                            args, 3);
      }
    } else {
      if (!block && !ring.isFull())
        break;

      auto now = std::chrono::steady_clock::now();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - block_start).count();
      if (elapsed_ms > timeout_ms) {
        size_t ring_idx = ring.tailIndex();
        auto& info = slot_info[ring_idx];
        LOG_ERROR("GraphExecutor: Ascend event wait timeout rank=%d epoch=%d work_seq=%llu lane=%d node=%zu "
                  "subchunk=%zu inflight=%zu block=%d elapsed_ms=%lld timeout_ms=%d",
                  rank_,
                  current_epoch_,
                  static_cast<unsigned long long>(current_work_seq_),
                  lane_idx,
                  info.node_idx,
                  info.subchunk_idx,
                  ring.inflight(),
                  block ? 1 : 0,
                  static_cast<long long>(elapsed_ms),
                  timeout_ms);
        logGraphProgress("ascend-event");
        last_result_ = pcclTimeout;
        return;
      }
      std::this_thread::yield();
    }
  }
}

int GraphExecutor::selectDmaLane(const InstructionNode& node) const {
  auto lane_offset = [](int a, int b, int lanes) {
    if (lanes <= 1)
      return 0;
    unsigned int x = static_cast<unsigned int>(a) * 0x9e3779b1u +
                     static_cast<unsigned int>(b) * 0x85ebca77u + 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<int>(x % static_cast<unsigned int>(lanes));
  };

  switch (node.op) {
    case OpCode::D2H:
      return DMA_LANE_D2H_BASE + lane_offset(node.output.owner, node.output.chunk, active_dma_lanes_per_direction_);
    case OpCode::H2D:
      if (!node.inputs.empty()) {
        return DMA_LANE_H2D_BASE +
               lane_offset(node.inputs[0].key.owner, node.inputs[0].key.chunk, active_dma_lanes_per_direction_);
      }
      return DMA_LANE_H2D_BASE + lane_offset(node.output.owner, node.output.chunk, active_dma_lanes_per_direction_);
    case OpCode::H2H:
      return DMA_LANE_H2D_BASE;
    case OpCode::D2D:
      return DMA_LANE_D2D_LOCAL;
    case OpCode::D2DReduce:
      return DMA_LANE_D2D_REDUCE;
    default:
      return DMA_LANE_H2D_BASE;
  }
}

int GraphExecutor::chooseActiveHostDmaLanes() const {
  size_t message_bytes = 0;
  if (graph_) {
    if (current_count_ > 0 && graph_->element_size > 0 &&
        current_count_ <= static_cast<size_t>(-1) / graph_->element_size) {
      message_bytes = current_count_ * graph_->element_size;
    } else {
      message_bytes = graph_->total_bytes;
    }
  }
  return selectHostDmaLaneCount(current_op_, message_bytes, DMA_LANES_PER_DIRECTION,
                                forcedHostDmaLanesFromEnv(), graph_ ? graph_->world_size : 0);
}

bool GraphExecutor::graphUsesMultipleHostNumas(const CompiledGraph& graph) const {
  bool seen[MAX_NUMA_NODES] = {};
  int count = 0;
  auto mark = [&](const ProgressKey& key) {
    if (key.space != ProgressSpace::HOST || key.owner < 0 || key.owner >= MAX_NUMA_NODES)
      return;
    if (!seen[key.owner]) {
      seen[key.owner] = true;
      count++;
    }
  };
  for (const auto& key : graph.active_progress_keys)
    mark(key);
  return count > 1;
}

bool GraphExecutor::trySubmitNode(size_t node_idx) {
  const auto& node = graph_->nodes[node_idx];
  auto& state = node_states_[node_idx];
  if (state.next_subchunk >= graph_->num_subchunks)
    return false;
  size_t sc = state.next_subchunk;
  if (!inputsReady(node, sc) || !overwriteReady(node, sc))
    return false;

  if (node.op == OpCode::NOP) {
    publishNodeProgress(node, sc);
    publishInputReaders(node, sc);
    state.next_subchunk++;
    state.completed_subchunks++;
    return true;
  }

  if (skipDma() && (node.op == OpCode::D2H || node.op == OpCode::H2D || node.op == OpCode::H2H ||
                    node.op == OpCode::D2D || node.op == OpCode::D2DReduce)) {
    publishNodeProgress(node, sc);
    publishInputReaders(node, sc);
    state.next_subchunk++;
    state.completed_subchunks++;
    return true;
  }

  size_t bytes = subchunkSize(node, sc);
  if (bytes == 0) {
    publishNodeProgress(node, sc);
    publishInputReaders(node, sc);
    state.next_subchunk++;
    state.completed_subchunks++;
    return true;
  }

  if (d2h_lead_window_enabled_ && node.op == OpCode::D2H && state_mgr_->getNumReaders(comm_id_, node.output) > 0) {
    uint64_t consumed = state_mgr_->getConsumeProgress(comm_id_, node.output, current_epoch_);
    if (static_cast<uint64_t>(sc) >= consumed + PCCL_D2H_LEAD_WINDOW) {
      if (shouldLogProgressKey(node.output) && debugProgressGenerationSelected(current_work_seq_)) {
        uint64_t write = state_mgr_->getWriteProgress(comm_id_, node.output, current_epoch_);
        uint64_t signature = (static_cast<uint64_t>(sc) << 48) ^ (write << 24) ^ consumed;
        if (node_idx >= d2h_lead_block_log_signature_.size() ||
            d2h_lead_block_log_signature_[node_idx] != signature) {
          if (node_idx < d2h_lead_block_log_signature_.size())
            d2h_lead_block_log_signature_[node_idx] = signature;
          LOG_INFO("PROGDBG executor d2h_lead_block rank=%d epoch=%d work_seq=%llu node=%zu output_key=(space=%s/%d "
                   "owner=%d chunk=%d version=%d) sc=%zu write=%llu consume=%llu readers=%d window=%d",
                   rank_,
                   current_epoch_,
                   static_cast<unsigned long long>(current_work_seq_),
                   node_idx,
                   spaceName(node.output.space),
                   static_cast<int>(node.output.space),
                   node.output.owner,
                   node.output.chunk,
                   node.output.version,
                   sc,
                   static_cast<unsigned long long>(write),
                   static_cast<unsigned long long>(consumed),
                   state_mgr_->getNumReaders(comm_id_, node.output),
                   PCCL_D2H_LEAD_WINDOW);
        }
      }
      return false;
    }
  }

  if (node.op == OpCode::H2H) {
    if (!h2h_engine_) {
      LOG_ERROR("GraphExecutor: H2H node requires H2HEngine node=%zu sc=%zu", node_idx, sc);
      finishCurrentWork(pcclInternalError);
      return false;
    }
    if (h2hRangeFastPathEnabled() && !state.h2h_range_submitted && sc == 0 &&
        node.inputs.size() == 1 && node.inputs[0].key.space == ProgressSpace::HOST &&
        node.output.space == ProgressSpace::HOST) {
      char* dst_base = resolve(node.output, 0, node.output_ring_buffered);
      char* src_base = resolve(node.inputs[0].key, 0, node.inputs[0].ring_buffered);
      if (dst_base && src_base && dst_base != src_base) {
        int h2h_to_dst = 0;
        for (const auto& graph_node : graph_->nodes) {
          if (graph_node.op == OpCode::H2H && graph_node.output.space == ProgressSpace::HOST &&
              graph_node.output.owner == node.output.owner) {
            h2h_to_dst++;
          }
        }
        int threads_per_task = std::max(1, std::min(16, h2h_engine_->workersPerNuma() /
                                                            std::max(1, h2h_to_dst)));
        H2HEngine::CopyRangeRequest request;
        request.owner = this;
        request.work_seq = current_work_seq_;
        request.node_idx = node_idx;
        request.dst_base = dst_base;
        request.src_base = src_base;
        request.bytes = node.work_bytes;
        request.subchunk_bytes = SUBCHUNK_SLICE_SIZE;
        request.num_subchunks = graph_->num_subchunks;
        request.input_key = node.inputs[0].key;
        request.output_key = node.output;
        request.input_ring_buffered = node.inputs[0].ring_buffered;
        request.output_ring_buffered = node.output_ring_buffered;
        request.input_reader_id = node.inputs[0].reader_id;
        request.has_overwrite_guard = node.has_overwrite_guard;
        request.overwrite_guard = node.overwrite_guard;
        request.comm_id = comm_id_;
        request.epoch = current_epoch_;
        request.src_numa = node.inputs[0].key.owner;
        request.dst_numa = node.output.owner;
        request.threads_per_task = threads_per_task;
        request.temporal_write = state_mgr_->getNumReaders(comm_id_, node.output) > 0;
        request.submit_ts = trace::nowMicros();
        if (h2h_engine_->submitCopyRange(request) != 0) {
          LOG_ERROR("GraphExecutor: H2H range submit failed node=%zu bytes=%zu src=%p dst=%p",
                    node_idx,
                    node.work_bytes,
                    src_base,
                    dst_base);
          finishCurrentWork(pcclInternalError);
          return false;
        }
        if (debugAsyncSubmitEnabled()) {
          LOG_INFO("ASYNCDBG executor h2h range submit rank=%d epoch=%d work_seq=%llu node=%zu bytes=%zu "
                   "subchunks=%zu threads=%d src=%p dst=%p src_numa=%d dst_numa=%d",
                   rank_,
                   current_epoch_,
                   static_cast<unsigned long long>(current_work_seq_),
                   node_idx,
                   node.work_bytes,
                   graph_->num_subchunks,
                   threads_per_task,
                   src_base,
                   dst_base,
                   request.src_numa,
                   request.dst_numa);
        }
        state.h2h_range_submitted = true;
        state.next_subchunk = graph_->num_subchunks;
        return true;
      }
    }
    char* dst = resolve(node.output, sc, node.output_ring_buffered);
    char* src = node.inputs.empty() ? nullptr : resolve(node.inputs[0].key, sc, node.inputs[0].ring_buffered);
    if (!dst || !src) {
      LOG_ERROR("GraphExecutor: failed to resolve H2H node=%zu sc=%zu src=%p dst=%p bytes=%zu",
                node_idx,
                sc,
                src,
                dst,
                bytes);
      finishCurrentWork(pcclInvalidArgument);
      return false;
    }
    if (src == dst) {
      publishNodeProgress(node, sc);
      publishInputReaders(node, sc);
      state.next_subchunk++;
      state.completed_subchunks++;
      return true;
    }
    H2HEngine::CopyRequest request;
    request.owner = this;
    request.work_seq = current_work_seq_;
    request.node_idx = node_idx;
    request.subchunk_idx = sc;
    request.dst = dst;
    request.src = src;
    request.bytes = bytes;
    request.src_numa = node.inputs[0].key.owner;
    request.dst_numa = node.output.owner;
    request.temporal_write = state_mgr_->getNumReaders(comm_id_, node.output) > 0;
    request.submit_ts = trace::nowMicros();
    if (h2h_engine_->submitCopy(request) != 0) {
      LOG_ERROR("GraphExecutor: H2H submit failed node=%zu sc=%zu bytes=%zu src=%p dst=%p",
                node_idx,
                sc,
                bytes,
                src,
                dst);
      finishCurrentWork(pcclInternalError);
      return false;
    }
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG executor h2h submit rank=%d epoch=%d work_seq=%llu node=%zu sc=%zu bytes=%zu src=%p dst=%p "
               "src_numa=%d dst_numa=%d",
               rank_,
               current_epoch_,
               static_cast<unsigned long long>(current_work_seq_),
               node_idx,
               sc,
               bytes,
               src,
               dst,
               request.src_numa,
               request.dst_numa);
    }
    state.next_subchunk++;
    return true;
  }

  int lane = selectDmaLane(node);
  auto& ring = event_rings_[lane];
  if (ring.isFull())
    return false;

  char* dst = resolve(node.output, sc, node.output_ring_buffered);
  char* src = node.inputs.empty() ? nullptr : resolve(node.inputs[0].key, sc, node.inputs[0].ring_buffered);
  if (!dst || !src || bytes == 0) {
    LOG_ERROR("GraphExecutor: failed to resolve DMA node=%zu sc=%zu src=%p dst=%p bytes=%zu", node_idx, sc, src, dst,
              bytes);
    finishCurrentWork(pcclInvalidArgument);
    return false;
  }

  if (node.op == OpCode::D2D && node.d2d_placement == D2DPlacement::Local && src == dst) {
    publishNodeProgress(node, sc);
    publishInputReaders(node, sc);
    state.next_subchunk++;
    state.completed_subchunks++;
    return true;
  }

  devStream stream = streams_[lane];
  devStatus status = devSuccess;
  devEvent ev = ring.nextSlot();
  slot_info_rings_[lane][ring.lastHeadIndex()] =
      DmaSlotInfo{node_idx, sc, bytes, lane, trace::nowMicros(), true};

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor dma submit begin rank=%d epoch=%d work_seq=%llu node=%zu op=%s sc=%zu lane=%d "
             "bytes=%zu src=%p dst=%p src_rank=%d dst_rank=%d",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             node_idx,
             opName(node.op),
             sc,
             lane,
             bytes,
             src,
             dst,
             node.source_rank,
             node.destination_rank);
  }
  if (debugProgressKeysEnabled() && debugProgressGenerationSelected(current_work_seq_)) {
    bool log_node = shouldLogProgressKey(node.output);
    for (const auto& input : node.inputs)
      log_node = log_node || shouldLogProgressKey(input.key);
    if (log_node) {
      LOG_INFO("PROGDBG executor dma_submit rank=%d epoch=%d work_seq=%llu node=%zu op=%s sc=%zu lane=%d "
               "bytes=%zu output=(space=%s/%d owner=%d chunk=%d version=%d) inputs=%zu src_rank=%d dst_rank=%d",
               rank_,
               current_epoch_,
               static_cast<unsigned long long>(current_work_seq_),
               node_idx,
               opName(node.op),
               sc,
               lane,
               bytes,
               spaceName(node.output.space),
               static_cast<int>(node.output.space),
               node.output.owner,
               node.output.chunk,
               node.output.version,
               node.inputs.size(),
               node.source_rank,
               node.destination_rank);
    }
  }
  switch (node.op) {
    case OpCode::D2H:
      status = devMemcpyAsync(dst, src, bytes, MemcpyDirection::D2H, stream);
      break;
    case OpCode::H2D:
      status = devMemcpyAsync(dst, src, bytes, MemcpyDirection::H2D, stream);
      break;
    case OpCode::D2D:
      status = devMemcpyAsync(dst, src, bytes, MemcpyDirection::D2D, stream);
      break;
    case OpCode::D2DReduce:
      status = devLaunchD2DReduce(dst, src, bytes, static_cast<int>(binding_.scalar_type),
                                  static_cast<int>(binding_.reduce_op), stream);
      break;
    default:
      break;
  }
  if (status != devSuccess) {
    LOG_ERROR("GraphExecutor: DMA submit failed node=%zu op=%d status=%d", node_idx, static_cast<int>(node.op),
              static_cast<int>(status));
    finishCurrentWork(pcclDeviceError);
    return false;
  }
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor dma submit end rank=%d epoch=%d work_seq=%llu node=%zu op=%s sc=%zu lane=%d "
             "status=%d",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             node_idx,
             opName(node.op),
             sc,
             lane,
             static_cast<int>(status));
  }
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor dma record_event begin rank=%d epoch=%d work_seq=%llu node=%zu sc=%zu lane=%d "
             "event=%p stream=%p",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             node_idx,
             sc,
             lane,
             ev.handle,
             stream.handle);
  }
  status = devRecordEvent(ev, stream);
  if (status != devSuccess) {
    LOG_ERROR("GraphExecutor: devRecordEvent failed status=%d", static_cast<int>(status));
    finishCurrentWork(pcclDeviceError);
    return false;
  }
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor dma record_event end rank=%d epoch=%d work_seq=%llu node=%zu sc=%zu lane=%d "
             "event=%p stream=%p",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             node_idx,
             sc,
             lane,
             ev.handle,
             stream.handle);
  }
  state.next_subchunk++;
  return true;
}

bool GraphExecutor::terminalsComplete() const {
  for (size_t i = 0; i < graph_->nodes.size(); ++i) {
    const auto& state = node_states_[i];
    if (state.completed_subchunks < graph_->num_subchunks)
      return false;
  }
  for (const auto& key : graph_->terminal_outputs) {
    if (state_mgr_->getWriteProgress(comm_id_, key, current_epoch_) < graph_->num_subchunks)
      return false;
  }
  return true;
}

void GraphExecutor::runGraph() {
  bool host_staging_region = false;
  if (graph_ && graphUsesHostStaging(*graph_)) {
    size_t required[MAX_NUMA_NODES] = {};
    size_t capacity[MAX_NUMA_NODES] = {};
    size_t offsets[MAX_NUMA_NODES] = {};
    for (int numa = 0; numa < MAX_NUMA_NODES; ++numa) {
      required[numa] =
          static_cast<size_t>(std::count_if(binding_.host_chunk_remap[numa].begin(),
                                            binding_.host_chunk_remap[numa].end(),
                                            [](int slot) { return slot >= 0; })) *
          graph_->chunk_size;
      capacity[numa] = binding_.host_capacity[numa];
    }
    if (!state_mgr_->acquireHostStagingRegion(
            comm_id_, world_size_, rank_, required, capacity, offsets)) {
      LOG_ERROR("GraphExecutor: failed to acquire host staging region rank=%d comm_id=%d work_seq=%llu",
                rank_,
                comm_id_,
                static_cast<unsigned long long>(current_work_seq_));
      finishCurrentWork(pcclTimeout);
      return;
    }
    for (int numa = 0; numa < MAX_NUMA_NODES; ++numa) {
      if (binding_.host_base[numa])
        binding_.host_base[numa] += offsets[numa];
    }
    host_staging_region = true;
  }

  // Progress slots are reused by epoch.  Reset at the beginning of the work,
  // then synchronize the communicator before any rank publishes new progress.
  // Resetting at the end lets a faster rank enter the next (possibly different)
  // graph while a slower rank is still clearing overlapping keys.
  state_mgr_->resetEpochKeys(comm_id_, current_epoch_, graph_->active_progress_keys);
  if (!state_mgr_->barrier(
          comm_id_, world_size_, "GraphExecutor::start", rank_, current_epoch_)) {
    LOG_ERROR("GraphExecutor: start barrier failed rank=%d comm_id=%d work_seq=%llu",
              rank_,
              comm_id_,
              static_cast<unsigned long long>(current_work_seq_));
    if (host_staging_region)
      state_mgr_->releaseHostStagingRegion(comm_id_, world_size_, rank_);
    finishCurrentWork(pcclTimeout);
    return;
  }

  int64_t t0 = trace::nowMicros();
  pcclResult_t result = pcclSuccess;
  const int timeout_ms = executorTimeoutMs();
  auto run_start = std::chrono::steady_clock::now();
  uint64_t idle_spins = 0;

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor runGraph begin rank=%d epoch=%d work_seq=%llu slot=%d op=%s active_dma_lanes=%d "
             "graph=%p nodes=%zu subchunks=%zu",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             current_slot_,
             collectiveDebugOpName(current_op_),
             active_dma_lanes_per_direction_,
             static_cast<const void*>(graph_),
             graph_ ? graph_->nodes.size() : 0,
             graph_ ? graph_->num_subchunks : 0);
  }
  logGraphDebugDump("runGraph_begin");

  while (true) {
    if (!initial_inputs_published_) {
      result = waitForUserReadyAndPublishInputs();
      if (result != pcclSuccess)
        break;
    }
    pollDeviceEvents(false);
    size_t h2h_completed = pollH2HCompletions();
    if (last_result_ != pcclSuccess) {
      result = last_result_;
      break;
    }

    bool progressed = false;
    for (size_t i = 0; i < graph_->nodes.size(); ++i) {
      progressed = trySubmitNode(i) || progressed;
      if (last_result_ != pcclSuccess) {
        result = last_result_;
        break;
      }
    }
    if (result != pcclSuccess)
      break;

    if (terminalsComplete())
      break;

    if (!progressed && h2h_completed == 0) {
      if ((++idle_spins & 0xfff) == 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - run_start).count();
        if (elapsed_ms > timeout_ms) {
          LOG_ERROR("GraphExecutor: runGraph timeout rank=%d epoch=%d work_seq=%llu elapsed_ms=%lld "
                    "timeout_ms=%d",
                    rank_,
                    current_epoch_,
                    static_cast<unsigned long long>(current_work_seq_),
                    static_cast<long long>(elapsed_ms),
                    timeout_ms);
          logGraphProgress("runGraph");
          result = pcclTimeout;
          break;
        }
        std::this_thread::yield();
      } else {
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
        _mm_pause();
#endif
      }
    } else {
      idle_spins = 0;
    }
  }

  pollDeviceEvents(true);
  while (pollH2HCompletions() > 0) {}
  if (last_result_ != pcclSuccess)
    result = last_result_;
  if (result == pcclSuccess) {
    int64_t b0 = trace::nowMicros();
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG executor barrier begin rank=%d epoch=%d work_seq=%llu",
               rank_,
               current_epoch_,
               static_cast<unsigned long long>(current_work_seq_));
    }
    if (!state_mgr_->barrier(comm_id_, world_size_, "GraphExecutor::runGraph", rank_, current_epoch_)) {
      result = pcclTimeout;
    }
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG executor barrier end rank=%d epoch=%d work_seq=%llu result=%d",
               rank_,
               current_epoch_,
               static_cast<unsigned long long>(current_work_seq_),
               static_cast<int>(result));
    }
    if (trace::isActive()) {
      char lbl[48];
      std::snprintf(lbl, sizeof(lbl), "Rank %d Barrier", rank_);
      trace::emitComplete(lbl, "sync", b0, trace::nowMicros() - b0, 0xFF8800AA);
    }
    if (result == pcclSuccess) {
      if (debugAsyncSubmitEnabled()) {
        LOG_INFO("ASYNCDBG executor signal_complete begin rank=%d epoch=%d work_seq=%llu active_keys=%zu",
                 rank_,
                 current_epoch_,
                 static_cast<unsigned long long>(current_work_seq_),
                 graph_->active_progress_keys.size());
      }
      if (completion_flag_ && completion_flag_->signalComplete(current_slot_, current_work_seq_) != 0)
        result = pcclDeviceError;
      if (debugAsyncSubmitEnabled()) {
        LOG_INFO("ASYNCDBG executor signal_complete end rank=%d epoch=%d work_seq=%llu result=%d",
                 rank_,
                 current_epoch_,
                 static_cast<unsigned long long>(current_work_seq_),
                 static_cast<int>(result));
      }
    }
  }

  if (trace::isActive()) {
    char label[32];
    std::snprintf(label, sizeof(label), "Rank %d Graph Batch", rank_);
    trace::emitComplete(label, "graph", t0, trace::nowMicros() - t0, PCCL_PROF_COLOR_SUBMIT);
  }

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG executor runGraph end rank=%d epoch=%d work_seq=%llu result=%d",
             rank_,
             current_epoch_,
             static_cast<unsigned long long>(current_work_seq_),
             static_cast<int>(result));
  }
  finishCurrentWork(result);
  if (host_staging_region)
    state_mgr_->releaseHostStagingRegion(comm_id_, world_size_, rank_);
}

}  // namespace pccl
