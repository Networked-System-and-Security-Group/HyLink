#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <stdexcept>

#include "../core/comm.hpp"
#include "../core/config.hpp"
#include "../core/error.hpp"
#include "../core/process_state.hpp"
#include "../executor/dma_lane_policy.hpp"
#include "../executor/graph_executor.hpp"
#include "../ir/compiled_graph.hpp"
#include "../ir/compiler.hpp"
#include "../ir/ir.hpp"
#include "../ir/ir_loader.hpp"
#include "../memory/completion_flag.hpp"
#include "../memory/host_pool.hpp"
#include "../memory/shared_state.hpp"
#include "../topology/comm_topology.hpp"
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

// Load IR program for a given collective and communicator.
static IRProgram loadIR(pcclComm_t comm, const std::string& collective, const CompileOptions* options = nullptr) {
  auto* ps = comm->process_state;
  const LocalTopology comm_topology =
      projectCommunicatorTopology(ps->config()->topology, comm->world_size, comm->device_id, comm->rank);
  std::string ir_file = isBroadcastRootCollective(collective)
                            ? IRLoader::findRelayIRFile(collective, comm_topology,
                                                        ps->config()->ir_path)
                            : IRLoader::findIRFile(collective, comm_topology,
                                                   ps->config()->ir_path);
  LOG_INFO("IRLoader: communicator topology collective=%s comm_size=%d device=%d local_rank=%d devices=%d numas=%d",
           collective.c_str(),
           comm->world_size,
           comm->device_id,
           comm->rank,
           comm_topology.num_devices,
           comm_topology.num_numa_nodes);
  IRProgram program = IRLoader::loadFromFile(ir_file, comm->rank, &comm_topology);
  if (program.nranks != comm->world_size) {
    throw std::runtime_error("IR communicator size mismatch for " + collective + ": file nranks=" +
                             std::to_string(program.nranks) + " communicator nranks=" +
                             std::to_string(comm->world_size));
  }
  return program;
}

static bool programUsesCrossRankDeviceSource(const IRProgram& program) {
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
  if (programUsesCrossRankDeviceSource(program))
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

static bool collectiveUsesD2DReduce(const std::string& collective) {
  return collective == "allreduce" || collective == "reducescatter";
}

static pcclResult_t ensureCommInternalBuffer(pcclComm_t comm, size_t bytes) {
  if (!comm || bytes == 0)
    return pcclSuccess;
  auto* ps = comm->process_state;
  const size_t current = ps->getInternalBufferSize(comm->comm_hash, comm->comm_id, comm->rank, comm->device_id);
  const size_t required = InternalBufferAllocationBytes(bytes);
  if (required == 0)
    return pcclInvalidArgument;
  if (required > current && comm->graph_executor && !comm->graph_executor->drain())
    return pcclTimeout;
  return ps->ensureInternalBuffer(comm->comm_hash, comm->comm_id, comm->rank, comm->device_id, bytes) == 0
             ? pcclSuccess
             : pcclInvalidArgument;
}

static bool makePublicTypedOptions(pcclDataType_t datatype, pcclReduceOp_t op,
                                   CompileOptions* options) {
  if (!options)
    return false;
  *options = CompileOptions();
  switch (datatype) {
    case pcclInt8:
      options->scalar_type = ScalarDataType::Int8;
      options->element_size = 1;
      break;
    case pcclUint8:
      options->scalar_type = ScalarDataType::Uint8;
      options->element_size = 1;
      break;
    case pcclInt32:
      options->scalar_type = ScalarDataType::Int32;
      options->element_size = 4;
      break;
    case pcclUint32:
      options->scalar_type = ScalarDataType::Uint32;
      options->element_size = 4;
      break;
    case pcclInt64:
      options->scalar_type = ScalarDataType::Int64;
      options->element_size = 8;
      break;
    case pcclUint64:
      options->scalar_type = ScalarDataType::Uint64;
      options->element_size = 8;
      break;
    case pcclFloat16:
      options->scalar_type = ScalarDataType::Float16;
      options->element_size = 2;
      break;
    case pcclFloat32:
      options->scalar_type = ScalarDataType::Float32;
      options->element_size = 4;
      break;
    case pcclFloat64:
      options->scalar_type = ScalarDataType::Float64;
      options->element_size = 8;
      break;
    case pcclBfloat16:
      options->scalar_type = ScalarDataType::Bfloat16;
      options->element_size = 2;
      break;
    case pcclInvalidDataType:
      return false;
  }

  switch (op) {
    case pcclReduceSum:
      options->reduce_op = ReduceOpKind::Sum;
      break;
    case pcclReduceMax:
      options->reduce_op = ReduceOpKind::Max;
      break;
    case pcclReduceMin:
      options->reduce_op = ReduceOpKind::Min;
      break;
    case pcclInvalidReduceOp:
      return false;
  }
  return true;
}

static pcclResult_t submitPublicTypedCollective(
    pcclComm_t comm, const char* collective, const void* sendbuff, void* recvbuff,
    size_t seg_count, size_t stride, int send_segs, int recv_segs,
    pcclDataType_t datatype, pcclReduceOp_t op, pcclStream_t stream) {
  if (!comm || !collective || !sendbuff || !recvbuff || seg_count == 0 ||
      stride >= seg_count)
    return pcclInvalidArgument;
  CompileOptions options;
  if (!makePublicTypedOptions(datatype, op, &options))
    return pcclInvalidArgument;
  return pccl::pcclSubmitTypedCollective(
      comm, collective, sendbuff, recvbuff, seg_count, stride,
      send_segs, recv_segs, stream, &options);
}

static bool allreduceRsAgDisabled() {
  const char* value = std::getenv("PCCL_ALLREDUCE_RSAG_DISABLE");
  return value && std::atoi(value) != 0;
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
  if (collective == "alltoall")
    return CollectiveDebugOp::AllToAll;
  if (isBroadcastRootCollective(collective))
    return CollectiveDebugOp::Broadcast;
  return CollectiveDebugOp::Unknown;
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
//   seg_count  — per-segment element count including any reserved prefix
//   reserved_offset — leading elements skipped in each segment
//   send_segs  — number of per-rank segments in sendbuf
//   recv_segs  — number of per-rank segments in recvbuf
static pcclResult_t submitCollective(pcclComm_t comm, const std::string& collective, const void* sendbuff,
                                     void* recvbuff, size_t seg_count, size_t reserved_offset, int send_segs, int recv_segs,
                                     pcclStream_t stream, const CompileOptions* compile_options = nullptr,
                                     CollectiveDebugOp debug_op_override = CollectiveDebugOp::Unknown,
                                     size_t logical_count_override = 0) {
  if (!comm || !sendbuff || !recvbuff || seg_count == 0)
    return pcclInvalidArgument;
  if (reserved_offset >= seg_count)
    return pcclInvalidArgument;

  ScopedDevice device_guard(comm->device_id);
  if (!device_guard.ok()) {
    LOG_ERROR("submitCollective: failed to set Ascend device=%d for collective=%s rank=%d",
              comm->device_id, collective.c_str(), comm->rank);
    return pcclDeviceError;
  }

  auto* ps = comm->process_state;
  devStream dev_stream;
  dev_stream.handle = stream;

  CompileOptions base_opts;
  if (compile_options)
    base_opts = *compile_options;
  CollectiveDebugOp submit_op = debug_op_override == CollectiveDebugOp::Unknown ? collectiveDebugOp(collective)
                                                                                : debug_op_override;
  size_t submit_count = logical_count_override > 0 ? logical_count_override : seg_count;
  if (logical_count_override == 0 && collective == "alltoall" &&
      !allToAllLogicalElementCount(seg_count, comm->world_size, &submit_count))
    return pcclInvalidArgument;
  base_opts.comm_hash = comm->comm_hash;
  base_opts.comm_id = comm->comm_id;
  base_opts.local_rank = comm->rank;
  base_opts.local_device_id = comm->device_id;
  if (base_opts.element_size == 0)
    return pcclInvalidArgument;
  if (collectiveUsesD2DReduce(collective) &&
      !devD2DReduceSupported(static_cast<int>(base_opts.scalar_type), static_cast<int>(base_opts.reduce_op))) {
    LOG_ERROR("submitCollective: unsupported D2DReduce dtype/op collective=%s dtype=%d op=%d",
              collective.c_str(),
              static_cast<int>(base_opts.scalar_type),
              static_cast<int>(base_opts.reduce_op));
    return pcclInvalidArgument;
  }

  const bool is_in_place =
      isCollectiveInPlace(comm, collective, sendbuff, recvbuff, seg_count, reserved_offset, base_opts.element_size);
  PlanCacheKey key{collective, seg_count, base_opts.element_size, base_opts.scalar_type,
                   base_opts.reduce_op, reserved_offset, send_segs, recv_segs, is_in_place};
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective enter collective=%s rank=%d nranks=%d comm=%p count=%zu elem=%zu dtype=%d "
             "send_segs=%d recv_segs=%d in_place=%d stream=%p cache_size=%zu send=%p recv=%p",
             collective.c_str(),
             comm->rank,
             comm->world_size,
             static_cast<void*>(comm),
             seg_count,
             base_opts.element_size,
             static_cast<int>(base_opts.scalar_type),
             send_segs,
             recv_segs,
             is_in_place ? 1 : 0,
             stream,
             comm->plan_cache.size(),
             sendbuff,
             recvbuff);
  }
  if (!ps->registerDeviceBuffers(comm->comm_hash, comm->comm_id, comm->rank, comm->device_id,
                                  sendbuff, recvbuff))
    return pcclInvalidArgument;

  auto it = comm->plan_cache.find(key);
  if (it != comm->plan_cache.end()) {
    if (debugAsyncSubmitEnabled()) {
      LOG_INFO("ASYNCDBG submitCollective cache_hit collective=%s rank=%d comm=%p nodes=%zu subchunks=%zu "
               "internal_bytes=%zu",
               collective.c_str(),
               comm->rank,
               static_cast<void*>(comm),
               it->second.graph.nodes.size(),
               it->second.graph.num_subchunks,
               it->second.graph.internal_bytes);
    }
    pcclResult_t internal_result = ensureCommInternalBuffer(comm, it->second.graph.internal_bytes);
    if (internal_result != pcclSuccess)
      return internal_result;
    CompileOptions bind_opts = base_opts;
    bind_opts.seg_count = seg_count;
    bind_opts.stride = reserved_offset;
    bind_opts.d2h_segments = send_segs;
    bind_opts.h2d_segments = recv_segs;
    bind_opts.is_in_place = is_in_place;
    bind_opts.in_place_alias_mode = (collective == "allgather") ? InPlaceAliasMode::AllGather
                                                                 : InPlaceAliasMode::SameBase;
    bind_opts.process_state = ps;
    bind_opts.comm_hash = comm->comm_hash;
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
        it->second.graph, binding, dev_stream, submit_op, submit_count,
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
  opts.stride = reserved_offset;
  opts.d2h_segments = send_segs;
  opts.h2d_segments = recv_segs;
  opts.is_in_place = is_in_place;
  opts.in_place_alias_mode = (collective == "allgather") ? InPlaceAliasMode::AllGather
                                                         : InPlaceAliasMode::SameBase;
  opts.process_state = ps;
  opts.comm_hash = comm->comm_hash;
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
  CompiledGraph graph = Compiler::compileGraph(program, comm->rank, comm->world_size, graph_opts);
  if (debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG submitCollective cache_miss collective=%s rank=%d comm=%p nodes=%zu subchunks=%zu "
             "internal_bytes=%zu terminal_outputs=%zu active_keys=%zu",
             collective.c_str(),
             comm->rank,
             static_cast<void*>(comm),
             graph.nodes.size(),
             graph.num_subchunks,
             graph.internal_bytes,
             graph.terminal_outputs.size(),
             graph.active_progress_keys.size());
  }
  pcclResult_t internal_result = ensureCommInternalBuffer(comm, graph.internal_bytes);
  if (internal_result != pcclSuccess)
    return internal_result;
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
      inserted.first->second.graph, binding, dev_stream, submit_op, submit_count,
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

static bool canSubmitAllReduceAsRsAg(pcclComm_t comm, const void* sendbuff, const void* recvbuff,
                                     size_t count) {
  if (allreduceRsAgDisabled())
    return false;
  if (!comm || !sendbuff || !recvbuff || comm->world_size <= 1 || count == 0)
    return false;
  if (sendbuff == recvbuff)
    return false;
  return count % static_cast<size_t>(comm->world_size) == 0;
}

static pcclResult_t submitAllReduceAsRsAg(pcclComm_t comm, const void* sendbuff, void* recvbuff,
                                          size_t count, pcclStream_t stream,
                                          const CompileOptions* compile_options = nullptr) {
  if (!canSubmitAllReduceAsRsAg(comm, sendbuff, recvbuff, count))
    return pcclInvalidArgument;

  CompileOptions opts;
  if (compile_options)
    opts = *compile_options;
  if (opts.element_size == 0)
    return pcclInvalidArgument;

  const size_t chunk_count = count / static_cast<size_t>(comm->world_size);
  char* local_chunk = static_cast<char*>(recvbuff) +
                      static_cast<size_t>(comm->rank) * chunk_count * opts.element_size;

  pcclResult_t result = submitCollective(comm, "reducescatter", sendbuff, local_chunk,
                                         chunk_count, 0, comm->world_size, 1, stream,
                                         compile_options, CollectiveDebugOp::AllReduce, count);
  if (result != pcclSuccess)
    return result;

  return submitCollective(comm, "allgather", local_chunk, recvbuff, chunk_count, 0, 1,
                          comm->world_size, stream, compile_options, CollectiveDebugOp::AllReduce, count);
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
  c->comm_hash = hashCommKey(uniqueId, nranks);
  auto* state_mgr = c->process_state->sharedState();
  if (c->process_state->isMultiProcess()) {
    c->comm_id = state_mgr->userReadyCommSlot(c->comm_hash);
    if (c->comm_id < 0) {
      delete c;
      return pcclInternalError;
    }
  }

  auto release_slot = [&]() {
    if (c->process_state->isMultiProcess() && c->comm_id >= 0) {
      state_mgr->releaseUserReadyCommSlot(c->comm_id, c->comm_hash);
      c->comm_id = -1;
    }
  };

  // Initialize per-device resources
  if (devSetDevice(c->device_id) != devSuccess) {
    release_slot();
    delete c;
    return pcclDeviceError;
  }
  if (devPrepareD2DReduce() != devSuccess) {
    LOG_ERROR("pcclCommInitRank: failed to prepare D2DReduce runtime path for device=%d", c->device_id);
    release_slot();
    delete c;
    return pcclDeviceError;
  }
  c->completion_flag = std::make_unique<CompletionFlag>();
  if (c->completion_flag->initialize(c->process_state->isMultiProcess()) != 0) {
    LOG_ERROR("pcclCommInitRank: failed to initialize collective doorbell for device=%d", c->device_id);
    release_slot();
    delete c;
    return pcclDeviceError;
  }

  c->graph_executor =
      std::make_unique<GraphExecutor>(rank, nranks, c->device_id, state_mgr,
                                      c->completion_flag.get(), c->process_state->h2hEngine(), c->comm_id,
                                      &c->process_state->config()->topology);

  // Synchronize all ranks at the end of per-communicator setup. The process-wide
  // ProcessState::initialize barrier only covers shared-state attach on the first
  // communicator; the per-communicator resources above (internal buffer, completion
  // doorbell, comm_id slot, executor thread) are not. Without this barrier a rank can
  // return and issue its very first collective before its peers have finished setting
  // up, so the first host-relay collective waits forever on a peer's cross-rank progress
  // signal that is never produced. One barrier per communicator at init time has
  // no steady-state cost.
  if (c->process_state->isMultiProcess() &&
      !state_mgr->barrier(c->comm_id, c->world_size, "pcclCommInitRank", c->rank, -1)) {
    LOG_ERROR("pcclCommInitRank: init barrier timeout rank=%d comm_id=%d", c->rank, c->comm_id);
    release_slot();
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
  auto* state_mgr = comm->process_state != nullptr ? comm->process_state->sharedState() : nullptr;
  const bool multi_process = comm->process_state != nullptr && comm->process_state->isMultiProcess();
  const int comm_id = comm->comm_id;
  const uint64_t comm_hash = comm->comm_hash;
  delete comm;
  if (multi_process && state_mgr != nullptr && comm_id >= 0)
    state_mgr->releaseUserReadyCommSlot(comm_id, comm_hash);
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
  if (canSubmitAllReduceAsRsAg(comm, sendbuff, recvbuff, count))
    return submitAllReduceAsRsAg(comm, sendbuff, recvbuff, count, stream);
  return submitCollective(comm, "allreduce", sendbuff, recvbuff, count, 0, 1, 1, stream);
}

pcclResult_t pcclAllReduceTyped(const void* sendbuff, void* recvbuff, size_t count,
                                pcclDataType_t datatype, pcclReduceOp_t op,
                                pcclComm_t comm, pcclStream_t stream) {
  return submitPublicTypedCollective(comm, "allreduce", sendbuff, recvbuff, count, 0, 1, 1,
                                     datatype, op, stream);
}

pcclResult_t pcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, pcclComm_t comm,
                           pcclStream_t stream) {
  if (!comm)
    return pcclInvalidArgument;
  return submitCollective(comm, "allgather", sendbuff, recvbuff, sendcount, 0, 1, comm->world_size, stream);
}

pcclResult_t pcclAllGatherTyped(const void* sendbuff, void* recvbuff, size_t sendcount,
                                pcclDataType_t datatype, pcclComm_t comm, pcclStream_t stream) {
  if (!comm)
    return pcclInvalidArgument;
  return submitPublicTypedCollective(comm, "allgather", sendbuff, recvbuff, sendcount, 0, 1,
                                     comm->world_size, datatype, pcclReduceSum, stream);
}

pcclResult_t pcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, pcclComm_t comm,
                               pcclStream_t stream) {
  if (!comm)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  // recvcount is the per-rank element count.
  // sendbuf has nranks segments of recvcount elements; recvbuf has 1 segment.
  return submitCollective(comm, "reducescatter", sendbuff, recvbuff, recvcount, 0, nranks, 1, stream);
}

pcclResult_t pcclReduceScatterTyped(const void* sendbuff, void* recvbuff, size_t recvcount,
                                    pcclDataType_t datatype, pcclReduceOp_t op,
                                    pcclComm_t comm, pcclStream_t stream) {
  if (!comm)
    return pcclInvalidArgument;
  return submitPublicTypedCollective(comm, "reducescatter", sendbuff, recvbuff, recvcount, 0,
                                     comm->world_size, 1, datatype, op, stream);
}

pcclResult_t pcclAllToAll(const void* sendbuff, void* recvbuff, size_t count, pcclComm_t comm, pcclStream_t stream) {
  if (!comm || count % static_cast<size_t>(comm->world_size) != 0)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  return submitCollective(comm, "alltoall", sendbuff, recvbuff, count / nranks, 0, nranks, nranks, stream);
}

pcclResult_t pcclAllToAllTyped(const void* sendbuff, void* recvbuff, size_t count,
                               pcclDataType_t datatype, pcclComm_t comm, pcclStream_t stream) {
  if (!comm || count % static_cast<size_t>(comm->world_size) != 0)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  return submitPublicTypedCollective(comm, "alltoall", sendbuff, recvbuff, count / nranks, 0,
                                     nranks, nranks, datatype, pcclReduceSum, stream);
}

pcclResult_t pcclAllGatherStrided(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                  pcclComm_t comm, pcclStream_t stream) {
  if (!comm)
    return pcclInvalidArgument;
  return submitCollective(comm, "allgather", sendbuff, recvbuff, count, stride, 1, comm->world_size, stream);
}

pcclResult_t pcclAllGatherStridedTyped(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                       pcclDataType_t datatype, pcclComm_t comm, pcclStream_t stream) {
  if (!comm)
    return pcclInvalidArgument;
  return submitPublicTypedCollective(comm, "allgather", sendbuff, recvbuff, count, stride, 1,
                                     comm->world_size, datatype, pcclReduceSum, stream);
}

pcclResult_t pcclReduceScatterStrided(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                      pcclComm_t comm, pcclStream_t stream) {
  if (!comm || count % static_cast<size_t>(comm->world_size) != 0)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  return submitCollective(comm, "reducescatter", sendbuff, recvbuff, count / static_cast<size_t>(nranks), stride,
                          nranks, 1, stream);
}

pcclResult_t pcclReduceScatterStridedTyped(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                           pcclDataType_t datatype, pcclReduceOp_t op,
                                           pcclComm_t comm, pcclStream_t stream) {
  if (!comm || count % static_cast<size_t>(comm->world_size) != 0)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  return submitPublicTypedCollective(comm, "reducescatter", sendbuff, recvbuff,
                                     count / static_cast<size_t>(nranks), stride,
                                     nranks, 1, datatype, op, stream);
}

pcclResult_t pcclAllToAllStrided(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                 pcclComm_t comm, pcclStream_t stream) {
  if (!comm || count % static_cast<size_t>(comm->world_size) != 0)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  return submitCollective(comm, "alltoall", sendbuff, recvbuff, count / static_cast<size_t>(nranks), stride,
                          nranks, nranks, stream);
}

pcclResult_t pcclAllToAllStridedTyped(const void* sendbuff, void* recvbuff, size_t count, size_t stride,
                                      pcclDataType_t datatype, pcclComm_t comm, pcclStream_t stream) {
  if (!comm || count % static_cast<size_t>(comm->world_size) != 0)
    return pcclInvalidArgument;
  int nranks = comm->world_size;
  return submitPublicTypedCollective(comm, "alltoall", sendbuff, recvbuff,
                                     count / static_cast<size_t>(nranks), stride,
                                     nranks, nranks, datatype, pcclReduceSum, stream);
}

pcclResult_t pcclBroadcastRoot(const void* sendbuff, void* recvbuff, size_t count, int root, pcclComm_t comm,
                               pcclStream_t stream) {
  if (!comm || root < 0 || root >= comm->world_size)
    return pcclInvalidArgument;
  if (root != 0)
    return pcclInvalidArgument;
  std::string collective = "broadcast_root" + std::to_string(root);
  const void* local_send = (comm->rank == root) ? sendbuff : recvbuff;
  return submitCollective(comm, collective, local_send, recvbuff, count, 0, 1, 1, stream);
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
  if (std::strcmp(collective, "allreduce") == 0 &&
      stride == 0 && send_segs == 1 && recv_segs == 1 &&
      canSubmitAllReduceAsRsAg(comm, sendbuff, recvbuff, seg_count)) {
    return submitAllReduceAsRsAg(comm, sendbuff, recvbuff, seg_count, stream, options);
  }
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
  opts.comm_hash = comm->comm_hash;
  opts.comm_id = comm->comm_id;
  opts.local_rank = comm->rank;
  opts.local_device_id = comm->device_id;
  if (!ps->registerDeviceBuffers(comm->comm_hash, comm->comm_id, comm->rank, comm->device_id,
                                 sendbuff, recvbuff))
    return pcclInvalidArgument;

  GraphCompileOptions graph_opts;
  graph_opts.legacy_layout = opts;
  graph_opts.staging_policy = StagingPolicy::Conservative;
  CompiledGraph graph = Compiler::compileGraph(program, comm->rank, comm->world_size, graph_opts);
  pcclResult_t internal_result = ensureCommInternalBuffer(comm, graph.internal_bytes);
  if (internal_result != pcclSuccess)
    return internal_result;
  RuntimeBinding binding = Compiler::makeRuntimeBinding(graph, opts, sendbuff, recvbuff, ps->hostPool());

  return comm->graph_executor->submit(graph, binding, dev_stream);
}

}  // namespace pccl
