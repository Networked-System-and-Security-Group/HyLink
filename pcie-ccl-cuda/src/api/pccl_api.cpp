#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <exception>
#include <mutex>
#include <string>
#include <stdexcept>

#include "../core/comm.hpp"
#include "../core/config.hpp"
#include "../core/error.hpp"
#include "../core/process_state.hpp"
#include "../executor/graph_executor.hpp"
#include "../fastpath/small_message_executor.hpp"
#include "../ir/compiled_graph.hpp"
#include "../ir/compiler.hpp"
#include "../ir/ir.hpp"
#include "../ir/ir_loader.hpp"
#include "../memory/completion_flag.hpp"
#include "../memory/host_pool.hpp"
#include "../memory/shared_state.hpp"
#include "../util/debug_flags.hpp"
#include "../util/log.hpp"
#include "../util/trace_writer.hpp"
#include "pccl/pccl.h"

using namespace pccl;

static Config s_config;
static bool s_initialized = false;
static std::mutex s_init_mutex;

static bool isUniqueIdZero(const pcclUniqueId& id) {
  for (int i = 0; i < PCCL_UNIQUE_ID_BYTES; i++) {
    if (id.internal[i] != 0) return false;
  }
  return true;
}

static uint64_t hashCommKey(const pcclUniqueId& id, int nranks) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : id.internal) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ull;
  }
  for (int shift = 0; shift < 32; shift += 8) {
    h ^= static_cast<uint64_t>((nranks >> shift) & 0xff);
    h *= 1099511628211ull;
  }
  return h ? h : 1;
}

static bool isBroadcastRootCollective(const std::string& collective);

static pcclResult_t pcclInitInternal(int nranks, const pcclUniqueId& uniqueId, int rank) {
  std::lock_guard<std::mutex> lock(s_init_mutex);
  if (s_initialized) {
    LOG_WARN("pcclInit: Already initialized");
    return pcclSuccess;
  }

  // Initialize trace early — before spawning worker threads so they can
  // register their thread names on startup.
  const char* trace_path = std::getenv("PCCL_TRACE_FILE");
  if (trace_path) {
    trace::initialize(trace_path);
    LOG_INFO("pcclInit: Trace output enabled → %s", trace_path);
  }

  pcclConfig_t c_config = PCCL_CONFIG_DEFAULT;
  c_config.nranks = nranks;
  s_config = Config::load(&c_config);

  // Multi-process mode: set unique_id and local_rank from the caller
  if (!isUniqueIdZero(uniqueId)) {
    s_config.multi_process = true;
    s_config.unique_id = std::string(uniqueId.internal, strnlen(uniqueId.internal, PCCL_UNIQUE_ID_BYTES));
    s_config.local_rank = rank;
    LOG_INFO("pcclInit: Multi-process mode, unique_id=%s, local_rank=%d", s_config.unique_id.c_str(), rank);
  }

  int ret = ProcessState::instance().initialize(s_config);
  if (ret != 0) {
    LOG_ERROR("pcclInit: ProcessState initialization failed");
    return pcclInternalError;
  }

  s_initialized = true;

  LOG_INFO("pcclInit: Success (nranks=%d, multi_process=%d)", s_config.nranks, s_config.multi_process);
  return pcclSuccess;
}

// Messages up to this size (logical bytes, as nccl-tests reports them) use the
// "<collective>_<pattern>_small.xml" IR when one sits next to the selected
// non-direct IR. Small variants trade pipelining depth for fewer dependent host
// round trips (e.g. one-shot ReduceScatter/AllReduce on 2 NUMA nodes); above a
// few MB the regular IR wins. Override with PCCL_SMALL_IR_MAX_BYTES (0 disables).
static size_t smallIRMaxBytesFromEnv() {
  static const size_t value = []() -> size_t {
    const char* env = std::getenv("PCCL_SMALL_IR_MAX_BYTES");
    if (!env || env[0] == '\0')
      return static_cast<size_t>(4) << 20;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 10);
    return end == env ? (static_cast<size_t>(4) << 20) : static_cast<size_t>(parsed);
  }();
  return value;
}

static std::string smallIRVariant(const std::string& ir_file, const CompileOptions* options) {
  if (!options)
    return std::string();
  const size_t segs = static_cast<size_t>(std::max(options->d2h_segments, options->h2d_segments));
  const size_t bytes = options->seg_count * options->element_size * segs;
  const std::string suffix = ".xml";
  const std::string direct_suffix = "_direct.xml";
  if (bytes == 0 || bytes > smallIRMaxBytesFromEnv() || ir_file.size() <= suffix.size())
    return std::string();
  if (ir_file.size() > direct_suffix.size() &&
      ir_file.compare(ir_file.size() - direct_suffix.size(), direct_suffix.size(), direct_suffix) == 0)
    return std::string();
  std::string small = ir_file.substr(0, ir_file.size() - suffix.size()) + "_small.xml";
  struct stat st;
  return stat(small.c_str(), &st) == 0 ? small : std::string();
}

// Load IR program for a given collective and communicator.
static IRProgram loadIR(pcclComm_t comm, const std::string& collective, const CompileOptions* options = nullptr) {
  auto* ps = comm->process_state;
  std::string ir_file = isBroadcastRootCollective(collective)
                            ? IRLoader::findRelayIRFile(collective, ps->config()->topology,
                                                        ps->config()->ir_path)
                            : IRLoader::findIRFile(collective, ps->config()->topology,
                                                   ps->config()->ir_path);
  if (!isBroadcastRootCollective(collective)) {
    std::string small = smallIRVariant(ir_file, options);
    if (!small.empty())
      ir_file = small;
  }
  IRProgram program = IRLoader::loadFromFile(ir_file, comm->rank, &ps->config()->topology);
  if ((collective == "allreduce" || collective == "reducescatter") && options &&
      (options->scalar_type == ScalarDataType::Float16 || options->scalar_type == ScalarDataType::Bfloat16)) {
    for (const auto& inst : program.all_instructions) {
      if (inst.op == OpCode::H2HReduce) {
        throw std::runtime_error("16-bit floating-point reduction is not supported by PCCL host-reduction IR");
      }
    }
  }
  return program;
}

static bool usesUnsupportedPeerD2D(const IRProgram& program, const Config& cfg, int current_rank,
                                   std::string* reason) {
  for (const auto& inst : program.instructions) {
    if (inst.op != OpCode::D2D)
      continue;
    int src_rank = inst.src_rank;
    if (src_rank < 0)
      continue;
    if (src_rank == current_rank)
      continue;
    if (cfg.multi_process) {
#if defined(PCCL_DEVICE_CUDA)
      continue;
#else
      if (reason) {
        *reason = "peer D2D in multi-process mode requires CUDA IPC device-pointer exchange";
      }
      return true;
#endif
    }
    if (!cfg.topology.pairDirectPeerAllowed(current_rank, src_rank)) {
      if (reason) {
        *reason = "cudaDeviceCanAccessPeer failed for dst rank " + std::to_string(current_rank) +
                  " reading src rank " + std::to_string(src_rank);
      }
      return true;
    }
    if (!cfg.topology.pairLcaIsPcieSwitch(current_rank, src_rank)) {
      if (reason) {
        *reason = "peer D2D is only allowed for PIX pairs; dst rank " + std::to_string(current_rank) +
                  " reading src rank " + std::to_string(src_rank) + " is not PIX";
      }
      return true;
    }
  }
  return false;
}

static bool programUsesPeerD2D(const IRProgram& program) {
  for (const auto& inst : program.all_instructions) {
    if (inst.op != OpCode::D2D && inst.op != OpCode::D2DReduce)
      continue;
    int src_rank = (inst.src_rank >= 0) ? inst.src_rank : inst.rank;
    if (src_rank != inst.rank)
      return true;
  }
  return false;
}

static bool allreduceDirectOverwriteSafe(const IRProgram& program, int world_size) {
  if (world_size <= 0 || world_size > MAX_RANKS)
    return false;
  if (program.input_chunk_count <= 0 || program.output_chunk_count <= 0)
    return false;
  if (programUsesPeerD2D(program))
    return false;

  bool input_read[MAX_RANKS][MAX_CHUNKS] = {};
  for (const auto& inst : program.all_instructions) {
    if (inst.op == OpCode::D2H && inst.src_buf == DeviceBufferKind::INPUT) {
      if (inst.rank >= 0 && inst.rank < world_size && inst.src.chunk_idx >= 0 &&
          inst.src.chunk_idx < MAX_CHUNKS) {
        input_read[inst.rank][inst.src.chunk_idx] = true;
      }
      continue;
    }
    if ((inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
        inst.src_buf == DeviceBufferKind::INPUT) {
      int src_rank = (inst.src_rank >= 0) ? inst.src_rank : inst.rank;
      if (src_rank >= 0 && src_rank < world_size && inst.src.chunk_idx >= 0 &&
          inst.src.chunk_idx < MAX_CHUNKS) {
        input_read[src_rank][inst.src.chunk_idx] = true;
      }
      continue;
    }

    bool writes_output = (inst.op == OpCode::H2D || inst.op == OpCode::D2D ||
                          inst.op == OpCode::D2DReduce) &&
                         inst.dst_buf == DeviceBufferKind::OUTPUT;
    if (!writes_output)
      continue;
    if (inst.rank < 0 || inst.rank >= world_size || inst.dst.chunk_idx < 0)
      return false;
    if (inst.dst.chunk_idx >= program.input_chunk_count)
      continue;
    if (inst.dst.chunk_idx >= MAX_CHUNKS || !input_read[inst.rank][inst.dst.chunk_idx])
      return false;
  }

  return true;
}

static bool isCollectiveInPlace(pcclComm_t comm, const std::string& collective, const void* sendbuff,
                                void* recvbuff, size_t seg_count, size_t stride, size_t element_size) {
  if (collective == "allgather") {
    const size_t segment_bytes = seg_count * element_size;
    const size_t active_offset = stride * element_size;
    const auto* expected =
        static_cast<const char*>(recvbuff) + static_cast<size_t>(comm->rank) * segment_bytes + active_offset;
    return sendbuff == expected;
  }
  return sendbuff == recvbuff;
}

static bool isBroadcastRootCollective(const std::string& collective) {
  const char prefix[] = "broadcast_root";
  return collective.compare(0, sizeof(prefix) - 1, prefix) == 0;
}

static int parseBroadcastRoot(const std::string& collective) {
  const char prefix[] = "broadcast_root";
  if (collective.compare(0, sizeof(prefix) - 1, prefix) != 0)
    return -1;
  const char* p = collective.c_str() + sizeof(prefix) - 1;
  if (*p == '\0')
    return -1;
  char* end = nullptr;
  long root = std::strtol(p, &end, 10);
  if (end == p || *end != '\0' || root < 0 || root > 1024)
    return -1;
  return static_cast<int>(root);
}

static CollectiveDebugOp collectiveDebugOp(const std::string& collective) {
  if (collective == "allreduce")
    return CollectiveDebugOp::AllReduce;
  if (collective == "allgather")
    return CollectiveDebugOp::AllGather;
  if (collective == "reducescatter")
    return CollectiveDebugOp::ReduceScatter;
  if (isBroadcastRootCollective(collective))
    return CollectiveDebugOp::Broadcast;
  return CollectiveDebugOp::Unknown;
}

static size_t smallMessageThresholdBytesFromEnv() {
  const char* env = std::getenv("PCCL_SMALL_MSG_THRESHOLD_BYTES");
  if (!env || env[0] == '\0')
    return 1024 * 1024;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed == 0)
    return 1024 * 1024;
  return static_cast<size_t>(parsed);
}

static bool smallMessageExecutorClusterReady(pcclComm_t comm, size_t threshold) {
  if (!comm || !comm->process_state || !comm->process_state->isMultiProcess())
    return false;
  auto* state_mgr = comm->process_state->sharedState();
  if (!state_mgr || comm->world_size <= 1 || comm->world_size > MAX_RANKS)
    return false;
  for (int rank = 0; rank < comm->world_size; ++rank) {
    if (!state_mgr->isFastpathRankAvailable(comm->comm_id, rank))
      return false;
    uint64_t peer_threshold = state_mgr->getFastpathThresholdBytes(comm->comm_id, rank);
    if (peer_threshold != static_cast<uint64_t>(threshold))
      return false;
  }
  return true;
}

static bool smallMessageFastpathEnabledFromEnv() {
  const char* env = std::getenv("PCCL_SMALL_MSG_FASTPATH");
  return !(env && env[0] == '0');
}

static bool makeSmallMessageRequest(pcclComm_t comm, const std::string& collective, const void* sendbuff,
                                    void* recvbuff, size_t seg_count, int send_segs, int recv_segs,
                                    pcclStream_t stream, const CompileOptions& opts,
                                    SmallMessageExecutor::Request* out) {
  if (!out)
    return false;
  SmallMessageExecutor::Request req;
  if (collective == "allgather") {
    req.kind = SmallMessageExecutor::CollectiveKind::AllGather;
  } else if (collective == "allreduce") {
    req.kind = SmallMessageExecutor::CollectiveKind::AllReduce;
  } else if (collective == "reducescatter") {
    req.kind = SmallMessageExecutor::CollectiveKind::ReduceScatter;
  } else if (isBroadcastRootCollective(collective)) {
    return false;
  } else {
    return false;
  }
  req.sendbuff = sendbuff;
  req.recvbuff = recvbuff;
  if (req.kind == SmallMessageExecutor::CollectiveKind::Broadcast && comm->rank != req.broadcast_root)
    req.sendbuff = recvbuff;
  req.seg_count = seg_count;
  req.send_segs = send_segs;
  req.recv_segs = recv_segs;
  req.element_size = opts.element_size;
  req.scalar_type = opts.scalar_type;
  req.reduce_op = opts.reduce_op;
  req.user_stream.handle = stream;
  req.debug_op = collectiveDebugOp(collective);
  req.debug_count = seg_count;
  req.comm = comm;
  *out = req;
  return true;
}

class ScopedDevice {
 public:
  explicit ScopedDevice(int device) : previous_(-1), restore_(false), ok_(true) {
    if (devGetDevice(&previous_) == devSuccess)
      restore_ = true;
    if (device >= 0 && (!restore_ || previous_ != device)) {
      if (devSetDevice(device) != devSuccess) {
        ok_ = false;
      }
    }
  }

  ~ScopedDevice() {
    if (restore_)
      devSetDevice(previous_);
  }

  bool ok() const { return ok_; }

 private:
  int previous_;
  bool restore_;
  bool ok_;
};

// Submit a collective with IR/plan caching.
//
//   collective — name used for IR file lookup (e.g. "alltoall")
//   seg_count  — per-segment element count (fast + pcie combined)
//   stride     — fast-path element count per segment (0 = contiguous)
//   send_segs  — number of per-rank segments in sendbuf
//   recv_segs  — number of per-rank segments in recvbuf
static pcclResult_t submitCollective(pcclComm_t comm, const std::string& collective, const void* sendbuff,
                                     void* recvbuff, size_t seg_count, size_t stride, int send_segs, int recv_segs,
                                     pcclStream_t stream, const CompileOptions* compile_options = nullptr) {
  if (!comm || !sendbuff || !recvbuff || seg_count == 0)
    return pcclInvalidArgument;

  ScopedDevice device_guard(comm->device_id);
  if (!device_guard.ok()) {
    LOG_ERROR("submitCollective: failed to set CUDA device=%d for collective=%s rank=%d",
              comm->device_id, collective.c_str(), comm->rank);
    return pcclDeviceError;
  }

  auto* ps = comm->process_state;
  devStream dev_stream;
  dev_stream.handle = stream;

  CompileOptions base_opts;
  if (compile_options)
    base_opts = *compile_options;
  base_opts.comm_id = comm->comm_id;
  base_opts.local_rank = comm->rank;
  base_opts.local_device_id = comm->device_id;
  if (base_opts.element_size == 0)
    return pcclInvalidArgument;

  const bool is_in_place =
      isCollectiveInPlace(comm, collective, sendbuff, recvbuff, seg_count, stride, base_opts.element_size);
  PlanCacheKey key{collective, seg_count, base_opts.element_size, base_opts.scalar_type,
                   base_opts.reduce_op, stride, send_segs, recv_segs, is_in_place};
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective enter collective=%s rank=%d nranks=%d comm=%p count=%zu elem=%zu dtype=%d "
             "stride=%zu send_segs=%d recv_segs=%d in_place=%d stream=%p cache_size=%zu send=%p recv=%p",
             collective.c_str(),
             comm->rank,
             comm->world_size,
             static_cast<void*>(comm),
             seg_count,
             base_opts.element_size,
             static_cast<int>(base_opts.scalar_type),
             stride,
             send_segs,
             recv_segs,
             is_in_place ? 1 : 0,
             stream,
             comm->plan_cache.size(),
             sendbuff,
             recvbuff);
  }
  if (!ps->registerDeviceBuffers(comm->comm_id, comm->rank, comm->device_id, sendbuff, recvbuff))
    return pcclInvalidArgument;

  if (comm->small_message_executor) {
    SmallMessageExecutor::Request small_req;
    if (makeSmallMessageRequest(comm, collective, sendbuff, recvbuff, seg_count, send_segs, recv_segs,
                                stream, base_opts, &small_req) &&
        smallMessageExecutorClusterReady(comm, comm->small_message_executor->thresholdBytes()) &&
        comm->small_message_executor->canHandle(small_req, stride)) {
      pcclResult_t small_result = comm->small_message_executor->submit(small_req);
      if (debugAsyncSubmitEnabled()) {
        LOG_INFO("ASYNCDBG submitCollective smallpath collective=%s rank=%d comm=%p result=%d bytes=%zu",
                 collective.c_str(),
                 comm->rank,
                 static_cast<void*>(comm),
                 static_cast<int>(small_result),
                 small_req.seg_count * small_req.element_size *
                     static_cast<size_t>(std::max(small_req.send_segs, small_req.recv_segs)));
      }
      if (small_result != pcclSuccess)
        return small_result;
      return pcclSuccess;
    }
  }

  if (comm->small_message_executor && comm->small_message_executor->hasInflight()) {
    pcclResult_t idle_result = comm->small_message_executor->waitForIdle("submitCollective.graph_fallback");
    if (idle_result != pcclSuccess)
      return idle_result;
  }

  auto it = comm->plan_cache.find(key);
  if (it != comm->plan_cache.end()) {
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG submitCollective cache_hit collective=%s rank=%d comm=%p nodes=%zu subchunks=%zu "
               "internal_bytes=%zu remote_d2d=%d",
               collective.c_str(),
               comm->rank,
               static_cast<void*>(comm),
               it->second.graph.nodes.size(),
               it->second.graph.num_subchunks,
               it->second.graph.internal_bytes,
               it->second.graph.has_remote_d2d ? 1 : 0);
    }
    if (it->second.graph.internal_bytes > 0 &&
        ps->ensureInternalBuffer(comm->rank, comm->device_id, it->second.graph.internal_bytes) != 0) {
      return pcclInvalidArgument;
    }
    if (it->second.graph.has_remote_d2d &&
        !ps->sharedState()->barrier(comm->comm_id, comm->world_size, "submitCollective.cached.remote_d2d",
                                    comm->rank, -1))
      return pcclTimeout;
    CompileOptions bind_opts = base_opts;
    bind_opts.seg_count = seg_count;
    bind_opts.stride = stride;
    bind_opts.d2h_segments = send_segs;
    bind_opts.h2d_segments = recv_segs;
    bind_opts.is_in_place = is_in_place;
    bind_opts.in_place_alias_mode = (collective == "allgather") ? InPlaceAliasMode::AllGather
                                                                 : InPlaceAliasMode::SameBase;
    bind_opts.process_state = ps;
    bind_opts.comm_id = comm->comm_id;
    bind_opts.local_rank = comm->rank;
    bind_opts.local_device_id = comm->device_id;
    RuntimeBinding binding;
    try {
      binding = Compiler::makeRuntimeBinding(it->second.graph, bind_opts, const_cast<void*>(sendbuff), recvbuff,
                                             ps->hostPool());
    } catch (const std::exception& e) {
      LOG_ERROR("submitCollective: runtime binding failed collective=%s rank=%d: %s", collective.c_str(),
                comm->rank, e.what());
      return pcclTimeout;
    }
    pcclResult_t submit_result = comm->graph_executor->submit(
        it->second.graph, binding, dev_stream, collectiveDebugOp(collective), seg_count,
        static_cast<size_t>(base_opts.scalar_type), comm);
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG submitCollective graph_submit_return collective=%s rank=%d comm=%p result=%d",
               collective.c_str(),
               comm->rank,
               static_cast<void*>(comm),
               static_cast<int>(submit_result));
    }
    if (submit_result != pcclSuccess)
      return submit_result;
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG submitCollective exit collective=%s rank=%d comm=%p result=%d cache_hit=1",
               collective.c_str(),
               comm->rank,
               static_cast<void*>(comm),
               static_cast<int>(pcclSuccess));
    }
    return pcclSuccess;
  }

  CompileOptions opts = base_opts;
  opts.seg_count = seg_count;
  opts.stride = stride;
  opts.d2h_segments = send_segs;
  opts.h2d_segments = recv_segs;
  opts.is_in_place = is_in_place;
  opts.in_place_alias_mode = (collective == "allgather") ? InPlaceAliasMode::AllGather
                                                         : InPlaceAliasMode::SameBase;
  opts.process_state = ps;
  opts.comm_id = comm->comm_id;
  opts.local_rank = comm->rank;
  opts.local_device_id = comm->device_id;

  IRProgram program;
  try {
    program = loadIR(comm, collective, &opts);
  } catch (const std::exception& e) {
    LOG_ERROR("submitCollective: IR load failed collective=%s rank=%d: %s", collective.c_str(), comm->rank,
              e.what());
    return pcclInvalidArgument;
  }

  std::string unsupported_reason;
  if (usesUnsupportedPeerD2D(program, *ps->config(), comm->rank, &unsupported_reason)) {
    LOG_ERROR("Unsupported peer D2D in selected IR for collective=%s rank=%d: %s", collective.c_str(), comm->rank,
              unsupported_reason.c_str());
    return pcclDeviceError;
  }

  GraphCompileOptions graph_opts;
  graph_opts.legacy_layout = opts;
  bool direct_overwrite =
      collective == "allgather" || collective == "reducescatter" ||
      isBroadcastRootCollective(collective) ||
      (collective == "allreduce" && allreduceDirectOverwriteSafe(program, comm->world_size));
  graph_opts.staging_policy =
      direct_overwrite ? StagingPolicy::DirectOverwrite : StagingPolicy::Conservative;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective staging_policy collective=%s rank=%d policy=%s",
             collective.c_str(),
             comm->rank,
             direct_overwrite ? "DirectOverwrite" : "Conservative");
  }
  graph_opts.prioritize_input_d2d = true;
  CompiledGraph graph = Compiler::compileGraph(program, comm->rank, comm->world_size, graph_opts);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective cache_miss collective=%s rank=%d comm=%p nodes=%zu subchunks=%zu "
             "internal_bytes=%zu remote_d2d=%d terminal_outputs=%zu active_keys=%zu",
             collective.c_str(),
             comm->rank,
             static_cast<void*>(comm),
             graph.nodes.size(),
             graph.num_subchunks,
             graph.internal_bytes,
             graph.has_remote_d2d ? 1 : 0,
             graph.terminal_outputs.size(),
             graph.active_progress_keys.size());
  }
  if (graph.internal_bytes > 0 &&
      ps->ensureInternalBuffer(comm->rank, comm->device_id, graph.internal_bytes) != 0) {
    return pcclInvalidArgument;
  }
  if (graph.has_remote_d2d &&
      !ps->sharedState()->barrier(comm->comm_id, comm->world_size, "submitCollective.compile.remote_d2d",
                                  comm->rank, -1))
    return pcclTimeout;
  RuntimeBinding binding;
  try {
    binding = Compiler::makeRuntimeBinding(graph, opts, const_cast<void*>(sendbuff), recvbuff, ps->hostPool());
  } catch (const std::exception& e) {
    LOG_ERROR("submitCollective: runtime binding failed collective=%s rank=%d: %s", collective.c_str(), comm->rank,
              e.what());
    return pcclTimeout;
  }

  auto inserted = comm->plan_cache.emplace(key, PlanCacheEntry{std::move(program), std::move(graph)});
  pcclResult_t submit_result = comm->graph_executor->submit(
      inserted.first->second.graph, binding, dev_stream, collectiveDebugOp(collective), seg_count,
      static_cast<size_t>(base_opts.scalar_type), comm);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective graph_submit_return collective=%s rank=%d comm=%p result=%d",
             collective.c_str(),
             comm->rank,
             static_cast<void*>(comm),
             static_cast<int>(submit_result));
  }
  if (submit_result != pcclSuccess)
    return submit_result;
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective exit collective=%s rank=%d comm=%p result=%d cache_hit=0",
             collective.c_str(),
             comm->rank,
             static_cast<void*>(comm),
             static_cast<int>(pcclSuccess));
  }
  return pcclSuccess;
}

extern "C" {

pcclResult_t pcclFinalize(void) {
  std::lock_guard<std::mutex> lock(s_init_mutex);
  if (!s_initialized)
    return pcclSuccess;

  // Shut down executors and engines first (joins all worker threads)
  ProcessState::instance().finalize();

  // Now safe to flush trace — all producer threads have exited
  trace::finalize();

  s_initialized = false;
  LOG_INFO("pcclFinalize: Success");
  return pcclSuccess;
}

pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId) {
  if (!uniqueId) return pcclInvalidArgument;
  std::memset(uniqueId, 0, sizeof(*uniqueId));

  // Generate random hex string for shm naming
  std::FILE* f = std::fopen("/dev/urandom", "r");
  if (!f) {
    LOG_ERROR("pcclGetUniqueId: Cannot open /dev/urandom");
    return pcclSystemError;
  }
  unsigned char bytes[16];
  if (std::fread(bytes, 1, sizeof(bytes), f) != sizeof(bytes)) {
    std::fclose(f);
    LOG_ERROR("pcclGetUniqueId: Failed to read /dev/urandom");
    return pcclSystemError;
  }
  std::fclose(f);

  // Encode as 32-char hex string
  for (int i = 0; i < 16; i++) {
    std::snprintf(uniqueId->internal + i * 2, 3, "%02x", bytes[i]);
  }

  LOG_INFO("pcclGetUniqueId: %s", uniqueId->internal);
  return pcclSuccess;
}

pcclResult_t pcclCommInitRank(pcclComm_t* comm, int nranks, pcclUniqueId uniqueId, int rank) {
  if (!comm || rank < 0 || rank >= nranks)
    return pcclInvalidArgument;

  int requested_device = -1;
  devGetDevice(&requested_device);

  // Initialize process state on first communicator creation
  if (!s_initialized) {
    pcclResult_t ret = pcclInitInternal(nranks, uniqueId, rank);
    if (ret != pcclSuccess)
      return ret;
  }

  auto* c = new pcclComm();
  c->rank = rank;
  c->world_size = nranks;
  c->device_id = requested_device >= 0 ? requested_device : rank;
  c->process_state = &ProcessState::instance();
  c->comm_id = 0;
  auto* state_mgr = c->process_state->sharedState();
  if (c->process_state->isMultiProcess()) {
    c->comm_id = state_mgr->userReadyCommSlot(hashCommKey(uniqueId, nranks));
    if (c->comm_id < 0) {
      delete c;
      return pcclInternalError;
    }
  }

  // Initialize per-GPU resources
  if (devSetDevice(c->device_id) != devSuccess) {
    delete c;
    return pcclDeviceError;
  }
  if (devPrepareD2DReduce() != devSuccess) {
    LOG_ERROR("pcclCommInitRank: failed to prepare D2DReduce kernel for device=%d", c->device_id);
    delete c;
    return pcclDeviceError;
  }
  if (devPrepareCollectiveKernel() != devSuccess) {
    LOG_ERROR("pcclCommInitRank: failed to prepare collective control kernels for device=%d", c->device_id);
    delete c;
    return pcclDeviceError;
  }
  // Context warm-up (default on; disable with PCCL_WARMUP_KERNEL=0).
  //
  // A freshly created CUDA context whose first GPU work is a pure-DMA collective
  // -- no compute kernel has ever executed and no device synchronize has occurred
  // -- deadlocks: the copy engine schedules a couple of subchunk DMAs then
  // permanently stops being scheduled, and the recorded completion events never
  // fire, so every rank's executor parks forever in cudaEventQuery (the
  // nccl-tests warmup / back-to-back hang). Executing one real compute kernel
  // followed by a full device synchronize activates the context's execution path;
  // afterwards arbitrarily deep DMA-only collective bursts run to completion. This
  // is exactly what a benchmark's initData() (a prepare kernel + cudaDeviceSync)
  // happens to do before its first timed collective -- we make the library robust
  // to applications that issue collectives on a cold context (e.g. a warmup
  // burst) instead of relying on that. One-time per communicator; no steady-state
  // cost. NB: draining/synchronizing the first collective alone does NOT cure this
  // -- that path carries no compute kernel, and the cold collective's own DMAs are
  // precisely what deadlock (see PCCL_COLD_START_DRAIN, now off by default).
  const char* warmup_env = std::getenv("PCCL_WARMUP_KERNEL");
  bool warmup_enabled = !(warmup_env && warmup_env[0] == '0');
  if (warmup_enabled) {
    if (devWarmupKernel() != devSuccess) {
      LOG_ERROR("pcclCommInitRank: warmup kernel failed for device=%d", c->device_id);
      delete c;
      return pcclDeviceError;
    }
    LOG_INFO("pcclCommInitRank: context warmup kernel done for device=%d", c->device_id);
  }
  if (c->process_state->initializeInternalBuffer(c->comm_id, c->rank, c->device_id) != 0) {
    LOG_ERROR("pcclCommInitRank: failed to initialize internal buffer for rank=%d device=%d", c->rank,
              c->device_id);
    delete c;
    return pcclDeviceError;
  }
  for (int peer = 0; peer < nranks; ++peer) {
    if (peer == c->device_id)
      continue;
    if (s_config.topology.pairDirectPeerAllowed(c->device_id, peer)) {
      devStatus status = devEnablePeerAccess(peer);
      if (status != devSuccess) {
        LOG_ERROR("pcclCommInitRank: devEnablePeerAccess failed for device=%d peer=%d status=%d", c->device_id,
                  peer, static_cast<int>(status));
        delete c;
        return pcclDeviceError;
      }
    }
  }

  c->completion_flag = std::make_unique<CompletionFlag>();
  if (c->completion_flag->initialize(c->process_state->isMultiProcess()) != 0) {
    LOG_ERROR("pcclCommInitRank: failed to initialize collective doorbell for device=%d", c->device_id);
    delete c;
    return pcclDeviceError;
  }

  c->graph_executor =
      std::make_unique<GraphExecutor>(rank, nranks, c->device_id, state_mgr, c->process_state->h2hEngine(),
                                      c->completion_flag.get(), c->comm_id);

  const bool small_fastpath_enabled = smallMessageFastpathEnabledFromEnv();
  if (small_fastpath_enabled) {
    c->small_completion_flag = std::make_unique<CompletionFlag>();
    if (c->small_completion_flag->initialize(c->process_state->isMultiProcess()) != 0) {
      LOG_WARN("pcclCommInitRank: small-message doorbell unavailable for device=%d", c->device_id);
      c->small_completion_flag.reset();
    }
  }

  if (small_fastpath_enabled && c->process_state->isMultiProcess() && nranks > 1 && c->small_completion_flag) {
    const size_t threshold_bytes = smallMessageThresholdBytesFromEnv();
    bool is_creator = (c->rank == 0);
    std::string small_pool_unique_id(uniqueId.internal,
                                     strnlen(uniqueId.internal, PCCL_UNIQUE_ID_BYTES));
    if (small_pool_unique_id.empty())
      small_pool_unique_id = s_config.unique_id;
    std::unique_ptr<HostMemoryPool> small_pool;
    small_pool = std::make_unique<HostMemoryPool>(nranks, threshold_bytes * static_cast<size_t>(nranks),
                                                  &s_config.topology, small_pool_unique_id, is_creator,
                                                  "_smallfast");
    c->small_message_executor = std::make_unique<SmallMessageExecutor>(
        rank, nranks, c->device_id, state_mgr, c->process_state, c->small_completion_flag.get(),
        std::move(small_pool), threshold_bytes, c->comm_id);
    state_mgr->publishFastpathStatus(c->comm_id, c->rank,
                                     c->small_message_executor && c->small_message_executor->initialized(),
                                     threshold_bytes);
  } else if (state_mgr) {
    state_mgr->publishFastpathStatus(c->comm_id, c->rank, false, smallMessageThresholdBytesFromEnv());
  }

  if (c->process_state->isMultiProcess() &&
      !state_mgr->barrier(c->comm_id, c->world_size, "pcclCommInitRank.small_fastpath", c->rank, -1)) {
    LOG_ERROR("pcclCommInitRank: small fastpath init barrier timeout rank=%d comm_id=%d", c->rank, c->comm_id);
    delete c;
    return pcclTimeout;
  }

  // Synchronize all ranks at the end of per-communicator setup. The process-wide
  // ProcessState::initialize barrier only covers shared-state attach on the first
  // communicator; the per-communicator resources above (internal buffer, completion
  // doorbell, comm_id slot, executor thread) are not. Without this barrier a rank can
  // return and issue its very first collective before its peers have finished setting
  // up, so the first host-relay collective waits forever on a peer's cross-rank progress
  // signal that is never produced (the nccl-tests warmup / back-to-back hang). The
  // application's initData() path happens to mask the race with its cudaDeviceSynchronize,
  // but a warmup burst issued immediately after init does not. One barrier per
  // communicator at init time has no steady-state cost.
  if (c->process_state->isMultiProcess() &&
      !state_mgr->barrier(c->comm_id, c->world_size, "pcclCommInitRank", c->rank, -1)) {
    LOG_ERROR("pcclCommInitRank: init barrier timeout rank=%d comm_id=%d", c->rank, c->comm_id);
    delete c;
    return pcclTimeout;
  }

  *comm = c;
  LOG_INFO("pcclCommInitRank: rank=%d, nranks=%d, device=%d, comm_id=%d", rank, nranks, c->device_id,
           c->comm_id);
  return pcclSuccess;
}

pcclResult_t pcclCommDestroy(pcclComm_t comm) {
  if (!comm)
    return pcclInvalidArgument;
  delete comm;
  return pcclSuccess;
}

pcclResult_t pcclCommRank(pcclComm_t comm, int* rank) {
  if (!comm || !rank)
    return pcclInvalidArgument;
  *rank = comm->rank;
  return pcclSuccess;
}

pcclResult_t pcclCommSize(pcclComm_t comm, int* size) {
  if (!comm || !size)
    return pcclInvalidArgument;
  *size = comm->world_size;
  return pcclSuccess;
}

pcclResult_t pcclAllReduce(const void* sendbuff, void* recvbuff, size_t count, pcclComm_t comm, pcclStream_t stream) {
  return submitCollective(comm, "allreduce", sendbuff, recvbuff, count, 0, 1, 1, stream);
}

pcclResult_t pcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, pcclComm_t comm,
                           pcclStream_t stream) {
  return submitCollective(comm, "allgather", sendbuff, recvbuff, sendcount, 0, 1, comm->world_size, stream);
}

pcclResult_t pcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, pcclComm_t comm,
                               pcclStream_t stream) {
  int nranks = comm->world_size;
  // recvcount = per-rank element count (matches NCCL semantics).
  // sendbuf has nranks segments of recvcount elements; recvbuf has 1 segment.
  return submitCollective(comm, "reducescatter", sendbuff, recvbuff, recvcount, 0, nranks, 1, stream);
}

pcclResult_t pcclAllToAll(const void* sendbuff, void* recvbuff, size_t count, pcclComm_t comm, pcclStream_t stream) {
  int nranks = comm->world_size;
  return submitCollective(comm, "alltoall", sendbuff, recvbuff, count / nranks, 0, nranks, nranks, stream);
}

pcclResult_t pcclAllGatherStrided(const void* sendbuff, void* recvbuff, size_t count, size_t stride, pcclComm_t comm,
                                  pcclStream_t stream) {
  return submitCollective(comm, "allgather", sendbuff, recvbuff, count, stride, 1, comm->world_size, stream);
}

pcclResult_t pcclReduceScatterStrided(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                      pcclComm_t comm, pcclStream_t stream) {
  int nranks = comm->world_size;
  // count = total element count.  seg_count = per-rank element count.
  // sendbuf has nranks segments; recvbuf has 1 segment.
  return submitCollective(comm, "reducescatter", sendbuff, recvbuff, count / nranks, stride, nranks, 1, stream);
}

pcclResult_t pcclAllToAllStrided(const void* sendbuff, void* recvbuff, size_t count, size_t stride, pcclComm_t comm,
                                 pcclStream_t stream) {
  int nranks = comm->world_size;
  return submitCollective(comm, "alltoall", sendbuff, recvbuff, count / nranks, stride, nranks, nranks, stream);
}

const char* pcclGetErrorString(pcclResult_t result) {
  return pccl::pcclGetErrorString(result);
}

}  // extern "C"

// C++ IR submission API
namespace pccl {

pcclResult_t pcclSubmitTypedCollective(pcclComm_t comm, const char* collective, const void* sendbuff, void* recvbuff,
                                       size_t seg_count, size_t stride, int send_segs, int recv_segs,
                                       pcclStream_t stream, const CompileOptions* options) {
  if (!collective || !options)
    return pcclInvalidArgument;
  return submitCollective(comm, collective, sendbuff, recvbuff, seg_count, stride, send_segs, recv_segs, stream,
                          options);
}

pcclResult_t pcclSubmitIR(pcclComm_t comm, const IRProgram& program, void* sendbuff, void* recvbuff, size_t count,
                          pcclStream_t stream, const CompileOptions* options) {
  if (!comm || !sendbuff || !recvbuff || count == 0)
    return pcclInvalidArgument;

  auto* ps = comm->process_state;
  devStream dev_stream;
  dev_stream.handle = stream;

  CompileOptions opts;
  if (options)
    opts = *options;
  if (opts.seg_count == 0)
    opts.seg_count = count;
  opts.is_in_place = (sendbuff == recvbuff);
  opts.in_place_alias_mode = InPlaceAliasMode::SameBase;
  opts.process_state = ps;
  opts.comm_id = comm->comm_id;
  opts.local_rank = comm->rank;
  opts.local_device_id = comm->device_id;
  if (!ps->registerDeviceBuffers(comm->comm_id, comm->rank, comm->device_id, sendbuff, recvbuff))
    return pcclInvalidArgument;

  GraphCompileOptions graph_opts;
  graph_opts.legacy_layout = opts;
  graph_opts.staging_policy = StagingPolicy::Conservative;
  CompiledGraph graph = Compiler::compileGraph(program, comm->rank, comm->world_size, graph_opts);
  if (graph.internal_bytes > 0 &&
      ps->ensureInternalBuffer(comm->rank, comm->device_id, graph.internal_bytes) != 0) {
    return pcclInvalidArgument;
  }
  RuntimeBinding binding = Compiler::makeRuntimeBinding(graph, opts, sendbuff, recvbuff, ps->hostPool());

  return comm->graph_executor->submit(graph, binding, dev_stream);
}

}  // namespace pccl
