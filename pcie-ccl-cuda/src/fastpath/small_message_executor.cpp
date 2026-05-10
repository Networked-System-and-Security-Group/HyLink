#include "small_message_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "../core/config.hpp"
#include "../core/process_state.hpp"
#include "../memory/host_pool.hpp"
#include "../memory/shared_state.hpp"
#include "../util/debug_flags.hpp"
#include "../util/log.hpp"

namespace pccl {

namespace {

constexpr size_t kDefaultThresholdBytes = 1024 * 1024;

int queueDepthFromEnv() {
  const char* e = std::getenv("PCCL_EXECUTOR_QUEUE_DEPTH");
  if (!e || e[0] == '\0')
    return PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed <= 0)
    return PCCL_EXECUTOR_QUEUE_DEPTH_DEFAULT;
  return static_cast<int>(std::max<long>(1, std::min<long>(PCCL_EXECUTOR_QUEUE_DEPTH_MAX, parsed)));
}

int timeoutMsFromEnv(const char* name, int fallback) {
  const char* e = std::getenv(name);
  if (!e || e[0] == '\0')
    return fallback;
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed <= 0)
    return fallback;
  return static_cast<int>(parsed);
}

bool fastpathEnvEnabled() {
  const char* e = std::getenv("PCCL_SMALL_MSG_FASTPATH");
  return !(e && e[0] == '0');
}

bool byteRangesOverlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  if (!a || !b || a_bytes == 0 || b_bytes == 0)
    return false;
  uintptr_t a_begin = reinterpret_cast<uintptr_t>(a);
  uintptr_t b_begin = reinterpret_cast<uintptr_t>(b);
  uintptr_t a_end = a_begin + a_bytes;
  uintptr_t b_end = b_begin + b_bytes;
  if (a_end < a_begin)
    a_end = UINTPTR_MAX;
  if (b_end < b_begin)
    b_end = UINTPTR_MAX;
  return a_begin < b_end && b_begin < a_end;
}

}  // namespace

SmallMessageExecutor::SmallMessageExecutor(int rank, int world_size, int device_id,
                                           SharedStateManager* state_mgr, ProcessState* process_state,
                                           CompletionFlag* completion_flag,
                                           std::unique_ptr<HostMemoryPool> staging_pool,
                                           size_t threshold_bytes, int comm_id)
    : rank_(rank),
      world_size_(world_size),
      device_id_(device_id),
      comm_id_(comm_id),
      state_mgr_(state_mgr),
      process_state_(process_state),
      completion_flag_(completion_flag),
      staging_pool_(std::move(staging_pool)),
      threshold_bytes_(threshold_bytes ? threshold_bytes : kDefaultThresholdBytes),
      enabled_(fastpathEnvEnabled()),
      initialized_(false),
      queue_depth_(completion_flag ? completion_flag->queueDepth() : queueDepthFromEnv()) {
  for (auto& stream : streams_)
    stream.handle = nullptr;
  for (auto& event : stream_events_)
    event.handle = nullptr;
  for (auto& ptr : scratch_)
    ptr = nullptr;

  if (!enabled_)
    return;
  if (!state_mgr_ || !process_state_ || !completion_flag_ || !staging_pool_ || !staging_pool_->isValid()) {
    LOG_WARN("SmallMessageExecutor: unavailable rank=%d missing shared resources", rank_);
    return;
  }
  if (!process_state_->isMultiProcess()) {
    LOG_INFO("SmallMessageExecutor: disabled in single-process mode rank=%d", rank_);
    return;
  }
  if (world_size_ <= 1 || world_size_ > MAX_RANKS) {
    LOG_WARN("SmallMessageExecutor: unsupported world_size=%d rank=%d", world_size_, rank_);
    return;
  }

#if defined(PCCL_DEVICE_CUDA)
  if (devSetDevice(device_id_) != devSuccess) {
    LOG_ERROR("SmallMessageExecutor: devSetDevice failed rank=%d device=%d", rank_, device_id_);
    return;
  }
  for (int i = 0; i < STREAM_COUNT; ++i) {
    if (devCreateHighPriorityStream(&streams_[i]) != devSuccess) {
      LOG_ERROR("SmallMessageExecutor: stream create failed rank=%d lane=%d", rank_, i);
      return;
    }
    if (devCreateEvent(&stream_events_[i]) != devSuccess) {
      LOG_ERROR("SmallMessageExecutor: event create failed rank=%d lane=%d", rank_, i);
      return;
    }
  }
  for (int i = 0; i < queue_depth_; ++i) {
    if (devMalloc(&scratch_[i], threshold_bytes_) != devSuccess || !scratch_[i]) {
      LOG_ERROR("SmallMessageExecutor: scratch allocation failed rank=%d slot=%d bytes=%zu",
                rank_, i, threshold_bytes_);
      return;
    }
  }
  initialized_ = true;
  thread_ = std::thread(&SmallMessageExecutor::threadFunc, this);
  LOG_INFO("SmallMessageExecutor: Started rank=%d device=%d comm_id=%d queue_depth=%d threshold=%zu",
           rank_, device_id_, comm_id_, queue_depth_, threshold_bytes_);
#else
  LOG_INFO("SmallMessageExecutor: disabled because build backend is not CUDA");
#endif
}

SmallMessageExecutor::~SmallMessageExecutor() {
  drain();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable())
    thread_.join();
  for (int i = 0; i < queue_depth_; ++i) {
    if (scratch_[i]) {
      devFree(scratch_[i]);
      scratch_[i] = nullptr;
    }
  }
  for (auto& event : stream_events_) {
    if (event.handle) {
      devDestroyEvent(event);
      event.handle = nullptr;
    }
  }
  for (auto& stream : streams_) {
    if (stream.handle) {
      devDestroyStream(stream);
      stream.handle = nullptr;
    }
  }
  LOG_INFO("SmallMessageExecutor: Stopped rank=%d", rank_);
}

bool SmallMessageExecutor::envEnabled() const {
  return fastpathEnvEnabled();
}

int SmallMessageExecutor::timeoutMs() const {
  return timeoutMsFromEnv("PCCL_SMALL_MSG_FASTPATH_TIMEOUT_MS", timeoutMsFromEnv("PCCL_EXECUTOR_TIMEOUT_MS", 120000));
}

const char* SmallMessageExecutor::kindName(CollectiveKind kind) const {
  switch (kind) {
    case CollectiveKind::AllGather:
      return "AllGather";
    case CollectiveKind::AllReduce:
      return "AllReduce";
    case CollectiveKind::ReduceScatter:
      return "ReduceScatter";
    case CollectiveKind::Broadcast:
      return "Broadcast";
  }
  return "Unknown";
}

size_t SmallMessageExecutor::logicalBytes(const Request& req) const {
  size_t max_segments = static_cast<size_t>(std::max(req.send_segs, req.recv_segs));
  return req.seg_count * req.element_size * max_segments;
}

bool SmallMessageExecutor::canHandle(const Request& req, size_t stride) const {
#if !defined(PCCL_DEVICE_CUDA)
  (void)req;
  (void)stride;
  return false;
#else
  if (!initialized_ || !enabled_ || !envEnabled())
    return false;
  if (!state_mgr_ || !process_state_ || !process_state_->isMultiProcess())
    return false;
  if (rank_ < 0 || rank_ >= MAX_RANKS || world_size_ <= 1 || world_size_ > MAX_RANKS)
    return false;
  if (!req.sendbuff || !req.recvbuff || req.seg_count == 0 || req.element_size == 0)
    return false;
  if (stride != 0)
    return false;
  if (logicalBytes(req) >= threshold_bytes_)
    return false;
  if (req.seg_count * req.element_size > threshold_bytes_)
    return false;
  if (req.send_segs <= 0 || req.recv_segs <= 0)
    return false;
  if (req.kind == CollectiveKind::ReduceScatter) {
    const size_t seg_bytes = req.seg_count * req.element_size;
    const size_t send_bytes = seg_bytes * static_cast<size_t>(world_size_);
    const auto* expected_in_place =
        static_cast<const char*>(req.sendbuff) + static_cast<size_t>(rank_) * seg_bytes;
    if (byteRangesOverlap(req.sendbuff, send_bytes, req.recvbuff, seg_bytes) &&
        req.recvbuff != expected_in_place)
      return false;
  }
  if (req.kind == CollectiveKind::AllReduce && !collectiveTopologyCanAllDirect(req))
    return false;
  switch (req.kind) {
    case CollectiveKind::AllGather:
      return req.send_segs == 1 && req.recv_segs == world_size_;
    case CollectiveKind::AllReduce:
      return req.send_segs == 1 && req.recv_segs == 1;
    case CollectiveKind::ReduceScatter:
      return req.send_segs == world_size_ && req.recv_segs == 1;
    case CollectiveKind::Broadcast:
      return req.send_segs == 1 && req.recv_segs == 1 &&
             req.broadcast_root >= 0 && req.broadcast_root < world_size_;
  }
  return false;
#endif
}

pcclResult_t SmallMessageExecutor::submit(const Request& req) {
  std::lock_guard<std::mutex> submit_lock(submit_mutex_);
  if (!completion_flag_)
    return pcclInternalError;

  uint64_t work_seq = 0;
  int slot = 0;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const int timeout_ms = timeoutMs();
    bool ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
      size_t active = busy_ ? 1 : 0;
      return stop_ || last_result_ != pcclSuccess || active + queue_.size() < static_cast<size_t>(queue_depth_);
    });
    if (!ready) {
      LOG_ERROR("SmallMessageExecutor: submit timeout rank=%d queued=%zu busy=%d queue_depth=%d timeout_ms=%d",
                rank_, queue_.size(), busy_ ? 1 : 0, queue_depth_, timeout_ms);
      last_result_ = pcclTimeout;
      return pcclTimeout;
    }
    if (stop_)
      return pcclInternalError;
    if (last_result_ != pcclSuccess)
      return last_result_;
    work_seq = ++work_seq_counter_;
    slot = static_cast<int>((work_seq - 1) % static_cast<uint64_t>(queue_depth_));
  }

  if (completion_flag_->prepareUserWait(slot, req.user_stream, req.debug_op, rank_, work_seq, req.debug_count,
                                        static_cast<size_t>(req.scalar_type), req.comm,
                                        req.user_stream.handle) != 0) {
    return pcclDeviceError;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(WorkItem{req, work_seq, slot});
  }
  cv_.notify_one();

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath submit rank=%d kind=%s work_seq=%llu slot=%d bytes=%zu",
             rank_, kindName(req.kind), static_cast<unsigned long long>(work_seq), slot, logicalBytes(req));
  }
  return pcclSuccess;
}

bool SmallMessageExecutor::drain() {
  waitForIdle("drain");
  return last_result_ == pcclSuccess;
}

void SmallMessageExecutor::synchronizeInternalStreams() {
  drain();
}

bool SmallMessageExecutor::hasInflight() {
  std::lock_guard<std::mutex> lock(mutex_);
  return busy_ || !queue_.empty();
}

pcclResult_t SmallMessageExecutor::waitForIdle(const char* where) {
  if (!initialized_)
    return pcclSuccess;
  std::unique_lock<std::mutex> lock(mutex_);
  const int timeout_ms = timeoutMs();
  bool done = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
    return !busy_ && queue_.empty();
  });
  if (!done) {
    LOG_ERROR("SmallMessageExecutor: waitForIdle timeout where=%s rank=%d queued=%zu busy=%d last_result=%d",
              where ? where : "unknown", rank_, queue_.size(), busy_ ? 1 : 0, static_cast<int>(last_result_));
    last_result_ = pcclTimeout;
    return pcclTimeout;
  }
  return last_result_;
}

void SmallMessageExecutor::threadFunc() {
  if (devSetDevice(device_id_) != devSuccess) {
    LOG_ERROR("SmallMessageExecutor: worker devSetDevice failed rank=%d device=%d", rank_, device_id_);
    return;
  }

  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty())
        break;
      item = queue_.front();
      queue_.pop_front();
      busy_ = true;
    }
    cv_.notify_all();
    runWork(item);
  }
}

void SmallMessageExecutor::runWork(const WorkItem& item) {
  pcclResult_t result = execute(item);
  finishWork(result, item);
}

void SmallMessageExecutor::finishWork(pcclResult_t result, const WorkItem& item) {
  if (result != pcclSuccess) {
    LOG_ERROR("SmallMessageExecutor: work failed rank=%d kind=%s slot=%d work_seq=%llu result=%d",
              rank_, kindName(item.req.kind), item.slot, static_cast<unsigned long long>(item.work_seq),
              static_cast<int>(result));
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    last_result_ = result;
    if (result != pcclSuccess) {
      if (completion_flag_)
        completion_flag_->signalComplete(item.slot, item.work_seq);
      for (const auto& queued : queue_) {
        if (completion_flag_)
          completion_flag_->signalComplete(queued.slot, queued.work_seq);
      }
      queue_.clear();
    }
    busy_ = false;
  }
  cv_.notify_all();
}

pcclResult_t SmallMessageExecutor::execute(const WorkItem& item) {
  const auto& req = item.req;
  const size_t seg_bytes = req.seg_count * req.element_size;
  if (seg_bytes == 0 || seg_bytes > threshold_bytes_)
    return pcclInvalidArgument;

  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath execute begin rank=%d kind=%s work_seq=%llu slot=%d seg_bytes=%zu",
             rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot, seg_bytes);
  }
  pcclResult_t result = waitForUserReadyAndPublish(item.slot, item.work_seq);
  if (result != pcclSuccess)
    return result;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath user_ready rank=%d kind=%s work_seq=%llu slot=%d",
             rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot);
  }

  bool all_direct = false;
  result = publishAndResolveDirectMode(req, item.slot, item.work_seq, &all_direct);
  if (result != pcclSuccess)
    return result;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath direct_mode rank=%d kind=%s work_seq=%llu slot=%d all_direct=%d",
             rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot,
             all_direct ? 1 : 0);
  }

  if (!all_direct) {
    result = stageLocalSource(req, item.slot, item.work_seq, seg_bytes);
    if (result != pcclSuccess)
      return result;
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG smallpath stage_done rank=%d kind=%s work_seq=%llu slot=%d",
               rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot);
    }
  } else {
    state_mgr_->publishFastpathStageSeq(comm_id_, rank_, item.slot, item.work_seq);
  }

  bool allow_remote_direct = true;
  if (!all_direct) {
    if (req.kind == CollectiveKind::AllGather) {
      const auto* expected_in_place =
          static_cast<const char*>(req.recvbuff) + static_cast<size_t>(rank_) * seg_bytes;
      if (req.sendbuff == expected_in_place)
        allow_remote_direct = false;
    } else if (req.kind == CollectiveKind::ReduceScatter) {
      allow_remote_direct = false;
    }
  }

  // Avoid mixed remote CUDA IPC and host-staged reads on paths that have shown ordering
  // hazards in mixed PIX/non-PIX topologies. Keep the faster mixed path for AG out-of-place.
  switch (req.kind) {
    case CollectiveKind::AllGather:
      result = executeAllGather(req, item.slot, item.work_seq, seg_bytes, allow_remote_direct);
      break;
    case CollectiveKind::Broadcast:
      result = executeBroadcast(req, item.slot, item.work_seq, seg_bytes, allow_remote_direct);
      break;
    case CollectiveKind::AllReduce:
      result = executeAllReduce(req, item.slot, item.work_seq, seg_bytes, allow_remote_direct);
      break;
    case CollectiveKind::ReduceScatter:
      result = executeReduceScatter(req, item.slot, item.work_seq, seg_bytes, allow_remote_direct);
      break;
  }
  if (result != pcclSuccess)
    return result;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath collective_done rank=%d kind=%s work_seq=%llu slot=%d",
             rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot);
  }

  if (!state_mgr_->barrier(comm_id_, world_size_, "SmallMessageExecutor::execute", rank_, -1))
    return pcclTimeout;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath barrier_done rank=%d kind=%s work_seq=%llu slot=%d",
             rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot);
  }

  if (completion_flag_ && completion_flag_->signalComplete(item.slot, item.work_seq) != 0)
    return pcclDeviceError;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath complete_signaled rank=%d kind=%s work_seq=%llu slot=%d",
             rank_, kindName(req.kind), static_cast<unsigned long long>(item.work_seq), item.slot);
  }
  return pcclSuccess;
}

pcclResult_t SmallMessageExecutor::waitForUserReadyAndPublish(int slot, uint64_t work_seq) {
  const int timeout_ms = timeoutMs();
  auto start = std::chrono::steady_clock::now();
  while (!completion_flag_->isReady(slot, work_seq)) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms) {
      LOG_ERROR("SmallMessageExecutor: user-ready timeout rank=%d slot=%d work_seq=%llu ready_seq=%llu",
                rank_, slot, static_cast<unsigned long long>(work_seq),
                static_cast<unsigned long long>(completion_flag_->readySequence(slot)));
      return pcclTimeout;
    }
    std::this_thread::yield();
  }
  state_mgr_->publishFastpathReadySeq(comm_id_, rank_, slot, work_seq);
  return pcclSuccess;
}

bool SmallMessageExecutor::sourceRankNeeded(const Request& req, int source_rank) const {
  if (source_rank == rank_)
    return false;
  switch (req.kind) {
    case CollectiveKind::AllGather:
    case CollectiveKind::AllReduce:
    case CollectiveKind::ReduceScatter:
      return source_rank >= 0 && source_rank < world_size_;
    case CollectiveKind::Broadcast:
      return source_rank == req.broadcast_root;
  }
  return false;
}

bool SmallMessageExecutor::peerDirectAllowed(int source_rank) const {
  return pairDirectAllowed(rank_, source_rank);
}

bool SmallMessageExecutor::pairDirectAllowed(int dst_rank, int source_rank) const {
  if (source_rank == dst_rank)
    return true;
  const Config* cfg = process_state_ ? process_state_->config() : nullptr;
  if (!cfg)
    return false;
  const auto& topo = cfg->topology;
  return topo.pairLcaIsPcieSwitch(dst_rank, source_rank) &&
         topo.pairDirectPeerAllowed(dst_rank, source_rank);
}

bool SmallMessageExecutor::collectiveTopologyCanAllDirect(const Request& req) const {
  for (int dst = 0; dst < world_size_; ++dst) {
    for (int source = 0; source < world_size_; ++source) {
      if (source == dst)
        continue;
      bool needed = false;
      switch (req.kind) {
        case CollectiveKind::AllGather:
        case CollectiveKind::AllReduce:
        case CollectiveKind::ReduceScatter:
          needed = true;
          break;
        case CollectiveKind::Broadcast:
          needed = (source == req.broadcast_root);
          break;
      }
      if (needed && !pairDirectAllowed(dst, source))
        return false;
    }
  }
  return true;
}

bool SmallMessageExecutor::localSourceNeededByAnyRank(const Request& req) const {
  switch (req.kind) {
    case CollectiveKind::AllGather:
    case CollectiveKind::AllReduce:
    case CollectiveKind::ReduceScatter:
      return true;
    case CollectiveKind::Broadcast:
      return rank_ == req.broadcast_root;
  }
  return true;
}

bool SmallMessageExecutor::localNeedsAllDirectSources(const Request& req) const {
  for (int source = 0; source < world_size_; ++source) {
    if (!sourceRankNeeded(req, source))
      continue;
    if (!process_state_->deviceIpcEnabled())
      return false;
    if (!peerDirectAllowed(source))
      return false;
    if (!process_state_->isRemoteDeviceBufferReady(comm_id_, rank_, ProcessState::IpcInput, source))
      return false;
    if (!process_state_->getRemoteDeviceBuffer(comm_id_, rank_, device_id_, ProcessState::IpcInput, source))
      return false;
  }
  return true;
}

pcclResult_t SmallMessageExecutor::publishAndResolveDirectMode(const Request& req, int slot, uint64_t work_seq,
                                                               bool* all_direct) {
  if (!all_direct)
    return pcclInvalidArgument;
  if (!collectiveTopologyCanAllDirect(req)) {
    *all_direct = false;
    return pcclSuccess;
  }
  bool local_all_direct = localNeedsAllDirectSources(req);
  state_mgr_->publishFastpathDirectStatus(comm_id_, rank_, slot, work_seq, local_all_direct);

  const int timeout_ms = timeoutMs();
  auto start = std::chrono::steady_clock::now();
  bool global_all_direct = true;
  for (int peer = 0; peer < world_size_; ++peer) {
    while (true) {
      int status = state_mgr_->getFastpathDirectStatus(comm_id_, peer, slot, work_seq);
      if (status >= 0) {
        if (status == 0)
          global_all_direct = false;
        break;
      }
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms) {
        LOG_ERROR("SmallMessageExecutor: direct-status timeout rank=%d peer=%d slot=%d work_seq=%llu",
                  rank_, peer, slot, static_cast<unsigned long long>(work_seq));
        return pcclTimeout;
      }
      std::this_thread::yield();
    }
  }
  *all_direct = global_all_direct;
  return pcclSuccess;
}

bool SmallMessageExecutor::waitForFastpathReady(int source_rank, int slot, uint64_t work_seq) {
  if (source_rank == rank_)
    return true;
  const int timeout_ms = timeoutMs();
  auto start = std::chrono::steady_clock::now();
  while (state_mgr_->getFastpathReadySeq(comm_id_, source_rank, slot) != work_seq) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms) {
      LOG_ERROR("SmallMessageExecutor: remote-ready timeout rank=%d source=%d slot=%d work_seq=%llu",
                rank_, source_rank, slot, static_cast<unsigned long long>(work_seq));
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

bool SmallMessageExecutor::waitForFastpathStage(int source_rank, int slot, uint64_t work_seq) {
  if (source_rank == rank_)
    return true;
  const int timeout_ms = timeoutMs();
  auto start = std::chrono::steady_clock::now();
  while (state_mgr_->getFastpathStageSeq(comm_id_, source_rank, slot) != work_seq) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms) {
      LOG_ERROR("SmallMessageExecutor: remote-stage timeout rank=%d source=%d slot=%d work_seq=%llu",
                rank_, source_rank, slot, static_cast<unsigned long long>(work_seq));
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

void* SmallMessageExecutor::stagingPtrForRank(int rank, size_t bytes) const {
  if (!staging_pool_)
    return nullptr;
  int numa = staging_pool_->getRankNUMANode(rank);
  return staging_pool_->getChunkBuffer(numa, rank, threshold_bytes_ >= bytes ? threshold_bytes_ : bytes);
}

void* SmallMessageExecutor::scratchPtrForSlot(int slot) const {
  if (slot < 0 || slot >= queue_depth_)
    return nullptr;
  return scratch_[slot];
}

pcclResult_t SmallMessageExecutor::stageLocalSource(const Request& req, int slot, uint64_t work_seq,
                                                    size_t seg_bytes) {
  if (!localSourceNeededByAnyRank(req)) {
    state_mgr_->publishFastpathStageSeq(comm_id_, rank_, slot, work_seq);
    return pcclSuccess;
  }

  void* host_dst = stagingPtrForRank(rank_, threshold_bytes_);
  if (!host_dst)
    return pcclInternalError;

  const char* src = static_cast<const char*>(req.sendbuff);
  size_t bytes = seg_bytes;
  if (req.kind == CollectiveKind::ReduceScatter) {
    bytes = seg_bytes * static_cast<size_t>(world_size_);
  }
  if (bytes == 0 || bytes > threshold_bytes_)
    return pcclInvalidArgument;

  devStatus status = devMemcpyAsync(host_dst, src, bytes, MemcpyDirection::D2H, streams_[STREAM_D2H]);
  if (status != devSuccess)
    return pcclDeviceError;
  status = devRecordEvent(stream_events_[STREAM_D2H], streams_[STREAM_D2H]);
  if (status != devSuccess)
    return pcclDeviceError;
  pcclResult_t wait_result = waitEvent(stream_events_[STREAM_D2H], "local-stage");
  if (wait_result != pcclSuccess)
    return wait_result;
  state_mgr_->publishFastpathStageSeq(comm_id_, rank_, slot, work_seq);
  return pcclSuccess;
}

pcclResult_t SmallMessageExecutor::buildSourceInfo(const Request& req, int source_rank, int slot,
                                                   uint64_t work_seq, size_t seg_bytes, int buffer_kind,
                                                   bool allow_remote_direct, SourceInfo* out) {
  if (!out)
    return pcclInvalidArgument;
  *out = SourceInfo{};
  if (!waitForFastpathReady(source_rank, slot, work_seq))
    return pcclTimeout;

  if (source_rank == rank_) {
    out->device_ptr = static_cast<const char*>(req.sendbuff);
    out->direct = true;
    return pcclSuccess;
  }

  if (allow_remote_direct && process_state_->deviceIpcEnabled()) {
    void* remote = nullptr;
    if (peerDirectAllowed(source_rank) &&
        process_state_->isRemoteDeviceBufferReady(comm_id_, rank_, buffer_kind, source_rank)) {
      remote = process_state_->getRemoteDeviceBuffer(comm_id_, rank_, device_id_, buffer_kind, source_rank);
    }
    if (remote) {
      out->device_ptr = static_cast<const char*>(remote);
      out->direct = true;
      return pcclSuccess;
    }
  }

  if (!waitForFastpathStage(source_rank, slot, work_seq))
    return pcclTimeout;
  out->host_ptr = static_cast<const char*>(stagingPtrForRank(source_rank, threshold_bytes_));
  if (!out->host_ptr)
    return pcclInternalError;
  (void)seg_bytes;
  return pcclSuccess;
}

pcclResult_t SmallMessageExecutor::copySourceToDevice(const SourceInfo& src, const char* src_offset, void* dst,
                                                      size_t bytes, devStream stream, int source_rank) {
  if (bytes == 0)
    return pcclSuccess;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath copy_submit rank=%d source=%d direct=%d bytes=%zu src=%p dst=%p",
             rank_, source_rank, src.direct ? 1 : 0, bytes, src_offset, dst);
  }
  devStatus status = devSuccess;
  if (src.direct) {
    status = (source_rank == rank_)
                 ? devMemcpyAsync(dst, src_offset, bytes, MemcpyDirection::D2D, stream)
                 : devMemcpyPeerAsync(dst, device_id_, src_offset, source_rank, bytes, stream);
  } else {
    status = devMemcpyAsync(dst, src_offset, bytes, MemcpyDirection::H2D, stream);
  }
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG smallpath copy_submitted rank=%d source=%d direct=%d status=%d",
             rank_, source_rank, src.direct ? 1 : 0, static_cast<int>(status));
  }
  return status == devSuccess ? pcclSuccess : pcclDeviceError;
}

pcclResult_t SmallMessageExecutor::executeAllGather(const Request& req, int slot, uint64_t work_seq,
                                                    size_t seg_bytes, bool allow_remote_direct) {
  char* recv = static_cast<char*>(req.recvbuff);
  for (int source = 0; source < world_size_; ++source) {
    SourceInfo src;
    pcclResult_t result = buildSourceInfo(req, source, slot, work_seq, seg_bytes, ProcessState::IpcInput,
                                          allow_remote_direct, &src);
    if (result != pcclSuccess)
      return result;
    const char* source_base = src.direct ? src.device_ptr : src.host_ptr;
    void* dst = recv + static_cast<size_t>(source) * seg_bytes;
    if (source_base != dst) {
      result = copySourceToDevice(src, source_base, dst, seg_bytes, streams_[STREAM_D2D], source);
      if (result != pcclSuccess)
        return result;
    }
  }
  return synchronizeStream(streams_[STREAM_D2D], "allgather");
}

pcclResult_t SmallMessageExecutor::executeBroadcast(const Request& req, int slot, uint64_t work_seq,
                                                    size_t seg_bytes, bool allow_remote_direct) {
  SourceInfo src;
  pcclResult_t result = buildSourceInfo(req, req.broadcast_root, slot, work_seq, seg_bytes,
                                        ProcessState::IpcInput, allow_remote_direct, &src);
  if (result != pcclSuccess)
    return result;
  const char* source_base = src.direct ? src.device_ptr : src.host_ptr;
  if (source_base == req.recvbuff)
    return pcclSuccess;
  result = copySourceToDevice(src, source_base, req.recvbuff, seg_bytes, streams_[STREAM_D2D],
                              req.broadcast_root);
  if (result != pcclSuccess)
    return result;
  return synchronizeStream(streams_[STREAM_D2D], "broadcast");
}

pcclResult_t SmallMessageExecutor::executeAllReduce(const Request& req, int slot, uint64_t work_seq,
                                                    size_t seg_bytes, bool allow_remote_direct) {
  char* recv = static_cast<char*>(req.recvbuff);
  int seed_source = rank_;
  if (req.sendbuff != req.recvbuff && world_size_ > 1) {
    for (int source = 0; source < world_size_; ++source) {
      if (source != rank_) {
        seed_source = source;
        break;
      }
    }
    SourceInfo src;
    pcclResult_t result = buildSourceInfo(req, seed_source, slot, work_seq, seg_bytes,
                                          ProcessState::IpcInput, allow_remote_direct, &src);
    if (result != pcclSuccess)
      return result;
    const char* source_base = src.direct ? src.device_ptr : src.host_ptr;
    result = copySourceToDevice(src, source_base, recv, seg_bytes, streams_[STREAM_REDUCE], seed_source);
    if (result != pcclSuccess)
      return result;
  } else if (req.sendbuff != req.recvbuff) {
    devStatus copy_status =
        devMemcpyAsync(recv, req.sendbuff, seg_bytes, MemcpyDirection::D2D, streams_[STREAM_REDUCE]);
    if (copy_status != devSuccess)
      return pcclDeviceError;
  }

  for (int source = 0; source < world_size_; ++source) {
    if (source == seed_source || (source == rank_ && req.sendbuff == req.recvbuff))
      continue;
    if (source == rank_) {
      if (devLaunchD2DReduce(recv, req.sendbuff, seg_bytes, static_cast<int>(req.scalar_type),
                             static_cast<int>(req.reduce_op), streams_[STREAM_REDUCE]) != devSuccess) {
        return pcclDeviceError;
      }
      continue;
    }
    SourceInfo src;
    pcclResult_t result = buildSourceInfo(req, source, slot, work_seq, seg_bytes, ProcessState::IpcInput,
                                          allow_remote_direct, &src);
    if (result != pcclSuccess)
      return result;
    const char* source_base = src.direct ? src.device_ptr : src.host_ptr;
    void* scratch = scratchPtrForSlot(slot);
    if (!scratch)
      return pcclInternalError;
    result = copySourceToDevice(src, source_base, scratch, seg_bytes, streams_[STREAM_REDUCE], source);
    if (result != pcclSuccess)
      return result;
    if (devLaunchD2DReduce(recv, scratch, seg_bytes, static_cast<int>(req.scalar_type),
                           static_cast<int>(req.reduce_op), streams_[STREAM_REDUCE]) != devSuccess) {
      return pcclDeviceError;
    }
  }
  return synchronizeStream(streams_[STREAM_REDUCE], "allreduce");
}

pcclResult_t SmallMessageExecutor::executeReduceScatter(const Request& req, int slot, uint64_t work_seq,
                                                        size_t seg_bytes, bool allow_remote_direct) {
  char* recv = static_cast<char*>(req.recvbuff);
  const size_t rank_offset = static_cast<size_t>(rank_) * seg_bytes;
  const char* local_src = static_cast<const char*>(req.sendbuff) + rank_offset;
  if (local_src != recv) {
    devStatus copy_status = devMemcpyAsync(recv, local_src, seg_bytes, MemcpyDirection::D2D,
                                           streams_[STREAM_REDUCE]);
    if (copy_status != devSuccess)
      return pcclDeviceError;
  }

  for (int source = 0; source < world_size_; ++source) {
    if (source == rank_)
      continue;
    SourceInfo src;
    pcclResult_t result = buildSourceInfo(req, source, slot, work_seq, seg_bytes, ProcessState::IpcInput,
                                          allow_remote_direct, &src);
    if (result != pcclSuccess)
      return result;
    const char* source_base = (src.direct ? src.device_ptr : src.host_ptr) + rank_offset;
    void* scratch = scratchPtrForSlot(slot);
    if (!scratch)
      return pcclInternalError;
    result = copySourceToDevice(src, source_base, scratch, seg_bytes, streams_[STREAM_REDUCE], source);
    if (result != pcclSuccess)
      return result;
    if (devLaunchD2DReduce(recv, scratch, seg_bytes, static_cast<int>(req.scalar_type),
                           static_cast<int>(req.reduce_op), streams_[STREAM_REDUCE]) != devSuccess) {
      return pcclDeviceError;
    }
  }
  return synchronizeStream(streams_[STREAM_REDUCE], "reducescatter");
}

pcclResult_t SmallMessageExecutor::waitEvent(devEvent event, const char* where) {
  const int timeout_ms = timeoutMs();
  auto start = std::chrono::steady_clock::now();
  while (true) {
    devEventStatus event_status = devEventStatusNotReady;
    devStatus query = devQueryEvent(event, &event_status);
    if (query != devSuccess) {
      LOG_ERROR("SmallMessageExecutor: event query failed where=%s rank=%d status=%d",
                where ? where : "unknown", rank_, static_cast<int>(query));
      return pcclDeviceError;
    }
    if (event_status == devEventStatusComplete)
      return pcclSuccess;
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms) {
      LOG_ERROR("SmallMessageExecutor: event wait timeout where=%s rank=%d timeout_ms=%d",
                where ? where : "unknown", rank_, timeout_ms);
      return pcclTimeout;
    }
    std::this_thread::yield();
  }
}

pcclResult_t SmallMessageExecutor::synchronizeStream(devStream stream, const char* where) {
  devStatus status = devSynchronizeStream(stream);
  if (status != devSuccess) {
    LOG_ERROR("SmallMessageExecutor: stream synchronize failed where=%s rank=%d status=%d",
              where ? where : "unknown", rank_, static_cast<int>(status));
    return pcclDeviceError;
  }
  return pcclSuccess;
}

}  // namespace pccl
