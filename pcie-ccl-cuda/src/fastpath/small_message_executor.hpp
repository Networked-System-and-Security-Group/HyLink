#ifndef PCCL_SMALL_MESSAGE_EXECUTOR_HPP
#define PCCL_SMALL_MESSAGE_EXECUTOR_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "../core/limits.hpp"
#include "../hal/device_rt.hpp"
#include "../ir/compiler.hpp"
#include "../memory/completion_flag.hpp"
#include "pccl/types.h"

namespace pccl {

class HostMemoryPool;
class ProcessState;
class SharedStateManager;

class SmallMessageExecutor {
 public:
  enum class CollectiveKind {
    AllGather,
    AllReduce,
    ReduceScatter,
    Broadcast,
  };

  struct Request {
    CollectiveKind kind = CollectiveKind::AllGather;
    const void* sendbuff = nullptr;
    void* recvbuff = nullptr;
    size_t seg_count = 0;
    int send_segs = 1;
    int recv_segs = 1;
    int broadcast_root = -1;
    size_t element_size = sizeof(float);
    ScalarDataType scalar_type = ScalarDataType::Float32;
    ReduceOpKind reduce_op = ReduceOpKind::Sum;
    devStream user_stream{nullptr};
    CollectiveDebugOp debug_op = CollectiveDebugOp::Unknown;
    size_t debug_count = 0;
    const void* comm = nullptr;
  };

  SmallMessageExecutor(int rank, int world_size, int device_id, SharedStateManager* state_mgr,
                       ProcessState* process_state, CompletionFlag* completion_flag,
                       std::unique_ptr<HostMemoryPool> staging_pool, size_t threshold_bytes,
                       int comm_id = 0);
  ~SmallMessageExecutor();

  SmallMessageExecutor(const SmallMessageExecutor&) = delete;
  SmallMessageExecutor& operator=(const SmallMessageExecutor&) = delete;

  bool canHandle(const Request& req, size_t stride) const;
  pcclResult_t submit(const Request& req);
  bool drain();
  pcclResult_t waitForIdle(const char* where);
  void synchronizeInternalStreams();
  bool hasInflight();
  size_t thresholdBytes() const { return threshold_bytes_; }
  bool initialized() const { return initialized_; }

 private:
  static constexpr int STREAM_D2H = 0;
  static constexpr int STREAM_D2D = 1;
  static constexpr int STREAM_REDUCE = 2;
  static constexpr int STREAM_COUNT = 3;

  struct WorkItem {
    Request req;
    uint64_t work_seq = 0;
    int slot = 0;
  };

  struct SourceInfo {
    const char* device_ptr = nullptr;
    const char* host_ptr = nullptr;
    bool direct = false;
  };

  void threadFunc();
  void runWork(const WorkItem& item);
  pcclResult_t execute(const WorkItem& item);
  void finishWork(pcclResult_t result, const WorkItem& item);
  pcclResult_t waitForUserReadyAndPublish(int slot, uint64_t work_seq);
  bool sourceRankNeeded(const Request& req, int source_rank) const;
  bool peerDirectAllowed(int source_rank) const;
  bool pairDirectAllowed(int dst_rank, int source_rank) const;
  bool collectiveTopologyCanAllDirect(const Request& req) const;
  bool localSourceNeededByAnyRank(const Request& req) const;
  bool localNeedsAllDirectSources(const Request& req) const;
  pcclResult_t publishAndResolveDirectMode(const Request& req, int slot, uint64_t work_seq, bool* all_direct);
  bool waitForFastpathReady(int source_rank, int slot, uint64_t work_seq);
  bool waitForFastpathStage(int source_rank, int slot, uint64_t work_seq);
  pcclResult_t stageLocalSource(const Request& req, int slot, uint64_t work_seq, size_t seg_bytes);
  pcclResult_t buildSourceInfo(const Request& req, int source_rank, int slot, uint64_t work_seq, size_t seg_bytes,
                               int buffer_kind, bool allow_remote_direct, SourceInfo* out);
  pcclResult_t copySourceToDevice(const SourceInfo& src, const char* src_offset, void* dst, size_t bytes,
                                  devStream stream, int source_rank);
  pcclResult_t executeAllGather(const Request& req, int slot, uint64_t work_seq, size_t seg_bytes,
                                bool allow_remote_direct);
  pcclResult_t executeBroadcast(const Request& req, int slot, uint64_t work_seq, size_t seg_bytes,
                                bool allow_remote_direct);
  pcclResult_t executeAllReduce(const Request& req, int slot, uint64_t work_seq, size_t seg_bytes,
                                bool allow_remote_direct);
  pcclResult_t executeReduceScatter(const Request& req, int slot, uint64_t work_seq, size_t seg_bytes,
                                    bool allow_remote_direct);
  pcclResult_t waitEvent(devEvent event, const char* where);
  pcclResult_t synchronizeStream(devStream stream, const char* where);
  void* stagingPtrForRank(int rank, size_t bytes) const;
  void* scratchPtrForSlot(int slot) const;
  size_t logicalBytes(const Request& req) const;
  bool envEnabled() const;
  int timeoutMs() const;
  const char* kindName(CollectiveKind kind) const;

  int rank_;
  int world_size_;
  int device_id_;
  int comm_id_;
  SharedStateManager* state_mgr_;
  ProcessState* process_state_;
  CompletionFlag* completion_flag_;
  std::unique_ptr<HostMemoryPool> staging_pool_;
  size_t threshold_bytes_;
  bool enabled_;
  bool initialized_;
  int queue_depth_;

  devStream streams_[STREAM_COUNT];
  devEvent stream_events_[STREAM_COUNT];
  void* scratch_[PCCL_EXECUTOR_QUEUE_DEPTH_MAX];

  std::thread thread_;
  std::mutex submit_mutex_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool busy_ = false;
  pcclResult_t last_result_ = pcclSuccess;
  uint64_t work_seq_counter_ = 0;
  std::deque<WorkItem> queue_;
};

}  // namespace pccl

#endif /* PCCL_SMALL_MESSAGE_EXECUTOR_HPP */
