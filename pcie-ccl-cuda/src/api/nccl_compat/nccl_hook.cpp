// nccl_hook.cpp - NCCL compatibility layer for LD_PRELOAD interception
//
// Usage: LD_PRELOAD=libpccl.so ./nccl_application
//
// This provides NCCL-compatible function symbols that redirect to PCCL.
// Types are defined inline to avoid requiring nccl.h at build time.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "nccl_compat_internal.hpp"

#include "../../core/comm.hpp"
#include "../../core/process_state.hpp"
#include "../../hal/device_rt.hpp"
#include <cuda_runtime.h>

#include "../../util/debug_flags.hpp"
#include "../../util/log.hpp"
#include "pccl/pccl.h"

#define PCCL_NCCL_ALIAS(name) __attribute__((alias(#name))) __attribute__((visibility("default")))

static thread_local bool t_group_active = false;
static thread_local int t_group_depth = 0;
static thread_local std::vector<NcclCompatPendingOp> t_group_ops;

static bool opWaveCompatible(const NcclCompatPendingOp& first, const NcclCompatPendingOp& next) {
  return first.type == next.type && first.count == next.count && first.datatype == next.datatype &&
         first.op == next.op && first.root == next.root;
}

static ncclResult_t getCommRankAndSize(ncclComm_t comm, int* rank, int* size) {
  ncclResult_t rr = from_pccl(pcclCommRank(comm, rank));
  if (rr != ncclSuccess)
    return rr;
  return from_pccl(pcclCommSize(comm, size));
}

static ncclResult_t runBroadcastOp(const NcclCompatPendingOp& op);
static ncclResult_t runReduceOp(const NcclCompatPendingOp& op);

static std::string broadcastCollectiveName(int root) {
  return "broadcast_root" + std::to_string(root);
}

static ncclResult_t submitNcclCompatCollective(ncclComm_t comm, const char* collective, const void* sendbuff,
                                               void* recvbuff, size_t seg_count, size_t stride, int send_segs,
                                               int recv_segs, cudaStream_t stream,
                                               const pccl::CompileOptions* options) {
  if (!comm)
    return ncclInvalidArgument;
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG compat submit enter collective=%s rank=%d nranks=%d count=%zu stride=%zu send_segs=%d "
             "recv_segs=%d dtype=%d elem=%zu stream=%p comm=%p group_active=%d",
             collective ? collective : "(null)",
             comm->rank,
             comm->world_size,
             seg_count,
             stride,
             send_segs,
             recv_segs,
             options ? static_cast<int>(options->scalar_type) : -1,
             options ? options->element_size : 0,
             stream,
             static_cast<void*>(comm),
             t_group_active ? 1 : 0);
  }
  ncclResult_t result = from_pccl(pccl::pcclSubmitTypedCollective(comm, collective, sendbuff, recvbuff, seg_count,
                                                                  stride, send_segs, recv_segs, stream, options));
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG compat submit exit collective=%s rank=%d result=%d comm=%p",
             collective ? collective : "(null)",
             comm->rank,
             static_cast<int>(result),
             static_cast<void*>(comm));
  }
  return result;
}

static ncclResult_t runPendingOp(const NcclCompatPendingOp& op) {
  if (op.device_id >= 0 && cudaSetDevice(op.device_id) != cudaSuccess) {
    LOG_ERROR("NCCL compat: failed to set device %d for grouped op", op.device_id);
    return ncclUnhandledCudaError;
  }
  int initial_rank = -1;
  int initial_nranks = 0;
  ncclResult_t initial_meta = getCommRankAndSize(op.comm, &initial_rank, &initial_nranks);
  if (initial_meta != ncclSuccess)
    return initial_meta;
  if (initial_nranks == 1 && op.type != NcclCompatOpType::Reduce)
    return runSingleRankCollective(op);
  ncclResult_t result = ncclInternalError;
  switch (op.type) {
    case NcclCompatOpType::AllReduce: {
      int nranks = 0;
      int rank = 0;
      result = getCommRankAndSize(op.comm, &rank, &nranks);
      size_t elem_size = datatypeSize(op.datatype);
      if (result == ncclSuccess && elem_size == 0)
        result = ncclInvalidArgument;
      pccl::CompileOptions options;
      if (result == ncclSuccess)
        result = makeCompileOptions(op.datatype, op.op, &options, true);
      if (result == ncclSuccess) {
        result = submitNcclCompatCollective(op.comm, "allreduce", op.sendbuff, op.recvbuff, op.count, 0, 1, 1,
                                            op.stream, &options);
      }
      break;
    }
    case NcclCompatOpType::AllGather: {
      int nranks = 0;
      int rank = 0;
      result = getCommRankAndSize(op.comm, &rank, &nranks);
      pccl::CompileOptions options;
      if (result == ncclSuccess)
        result = makeCompileOptions(op.datatype, ncclSum, &options, false);
      size_t elem_size = datatypeSize(op.datatype);
      if (result == ncclSuccess && elem_size == 0)
        result = ncclInvalidArgument;
      if (result == ncclSuccess) {
        result = submitNcclCompatCollective(op.comm, "allgather", op.sendbuff, op.recvbuff, op.count, 0, 1,
                                            nranks, op.stream, &options);
      }
      break;
    }
    case NcclCompatOpType::ReduceScatter: {
      int nranks = 0;
      int rank = 0;
      result = getCommRankAndSize(op.comm, &rank, &nranks);
      pccl::CompileOptions options;
      if (result == ncclSuccess)
        result = makeCompileOptions(op.datatype, op.op, &options, true);
      size_t elem_size = datatypeSize(op.datatype);
      if (result == ncclSuccess && elem_size == 0)
        result = ncclInvalidArgument;
      if (result == ncclSuccess) {
        result = submitNcclCompatCollective(op.comm, "reducescatter", op.sendbuff, op.recvbuff, op.count, 0,
                                            nranks, 1, op.stream, &options);
      }
      break;
    }
    case NcclCompatOpType::Broadcast:
      result = runBroadcastOp(op);
      break;
    case NcclCompatOpType::Reduce:
      result = runReduceOp(op);
      break;
  }
  return result;
}

static ncclResult_t runBroadcastOp(const NcclCompatPendingOp& op) {
  int rank = -1;
  int nranks = 0;
  ncclResult_t meta = getCommRankAndSize(op.comm, &rank, &nranks);
  if (meta != ncclSuccess)
    return meta;
  if (op.root < 0 || op.root >= nranks)
    return ncclInvalidArgument;
  size_t elem_size = datatypeSize(op.datatype);
  if (elem_size == 0)
    return ncclInvalidArgument;
  if (!op.recvbuff)
    return ncclInvalidArgument;

  const void* local_send = (rank == op.root) ? (op.sendbuff ? op.sendbuff : op.recvbuff)
                                             : (op.recvbuff ? op.recvbuff : op.sendbuff);
  if (!local_send)
    return ncclInvalidArgument;

  pccl::CompileOptions options;
  ncclResult_t opt_ret = makeCompileOptions(op.datatype, ncclSum, &options, false);
  if (opt_ret != ncclSuccess)
    return opt_ret;

  std::string collective = broadcastCollectiveName(op.root);
  return submitNcclCompatCollective(op.comm, collective.c_str(), local_send, op.recvbuff, op.count, 0, 1, 1,
                                    op.stream, &options);
}

static ncclResult_t runReduceOp(const NcclCompatPendingOp& op) {
  (void)op;
  LOG_WARN("NCCL compat: ncclReduce is not supported");
  return ncclInternalError;
}

static ncclResult_t validateReduceRoot(const NcclCompatPendingOp& op) {
  int rank = 0;
  int nranks = 0;
  ncclResult_t meta = getCommRankAndSize(op.comm, &rank, &nranks);
  if (meta != ncclSuccess)
    return meta;
  if (op.root < 0 || op.root >= nranks)
    return ncclInvalidArgument;
  return ncclSuccess;
}

static ncclResult_t runPendingWave(const std::vector<NcclCompatPendingOp>& ops, size_t begin, size_t end,
                                   size_t wave_idx) {
  if (begin >= end)
    return ncclSuccess;

  if (pccl::debugAsyncSubmitEnabled()) {
    int first_rank = -1;
    int first_size = 0;
    getCommRankAndSize(ops[begin].comm, &first_rank, &first_size);
    LOG_INFO("ASYNCDBG group wave enter wave=%zu type=%s ops=%zu first_rank=%d nranks=%d count=%zu dtype=%d root=%d",
             wave_idx,
             opTypeName(ops[begin].type),
             end - begin,
             first_rank,
             first_size,
             ops[begin].count,
             static_cast<int>(ops[begin].datatype),
             ops[begin].root);
  }

  LOG_DEBUG("NCCL compat: running group wave %zu type=%s ops=%zu", wave_idx,
            opTypeName(ops[begin].type), end - begin);

  std::vector<ncclResult_t> results(end - begin, ncclSuccess);
  std::vector<std::thread> threads;
  threads.reserve(end - begin);
  for (size_t i = begin; i < end; ++i) {
    threads.emplace_back([&, i] {
      results[i - begin] = runPendingOp(ops[i]);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (auto result : results) {
    if (result != ncclSuccess)
      return result;
  }
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG group wave exit wave=%zu type=%s result=%d", wave_idx, opTypeName(ops[begin].type),
             static_cast<int>(ncclSuccess));
  }
  return ncclSuccess;
}

static int deviceForOp(ncclComm_t comm) {
  if (comm && comm->device_id >= 0)
    return comm->device_id;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return -1;
  }
  return device;
}

// ============================================================================
// NCCL API implementation
// ============================================================================

extern "C" {

// --- Version & ID ---

ncclResult_t ncclGetVersion(int* version) {
  if (version)
    *version = 21701;  // Report as NCCL 2.17.1
  return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (!uniqueId)
    return ncclInvalidArgument;
  pcclUniqueId pcclId;
  pcclResult_t ret = pcclGetUniqueId(&pcclId);
  if (ret != pcclSuccess)
    return from_pccl(ret);
  static_assert(sizeof(ncclUniqueId) == sizeof(pcclUniqueId), "NCCL/PCCL unique ID sizes must match");
  std::memcpy(uniqueId, &pcclId, sizeof(ncclUniqueId));
  rememberLocallyGeneratedUniqueId(*uniqueId);
  LOG_INFO("NCCL compat: ncclGetUniqueId");
  return ncclSuccess;
}

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  if (!ptr)
    return ncclInvalidArgument;
  return (cudaMalloc(ptr, size) == cudaSuccess) ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t ncclMemFree(void* ptr) {
  return (cudaFree(ptr) == cudaSuccess) ? ncclSuccess : ncclUnhandledCudaError;
}

// --- Communicator lifecycle ---

// Track live communicator count so we can auto-finalize on last destroy
static std::atomic<int> s_nccl_comm_count{0};

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  LOG_INFO("NCCL compat: ncclCommInitRank(nranks=%d, rank=%d)", nranks, rank);

  pcclUniqueId pcclId;
  std::memset(&pcclId, 0, sizeof(pcclId));
  bool multi_process = false;
  uint64_t id_hash = hashUniqueId(commId);
  ncclResult_t mode_ret = detectCompatMultiProcessByRankPids(commId, nranks, rank, &multi_process);
  if (mode_ret != ncclSuccess)
    return mode_ret;

  static_assert(sizeof(ncclUniqueId) == sizeof(pcclUniqueId), "NCCL/PCCL unique ID sizes must match");
  if (multi_process) {
    std::memcpy(&pcclId, &commId, sizeof(pcclId));
    LOG_INFO("NCCL compat: multi-process communicator detected, passing PCCL unique ID through");
  } else {
    LOG_INFO("NCCL compat: single-process communicator detected, using zero PCCL unique ID");
  }

  ncclResult_t ret = from_pccl(pcclCommInitRank(comm, nranks, pcclId, rank));
  if (ret == ncclSuccess) {
    rememberCommState(*comm, id_hash, multi_process);
    s_nccl_comm_count.fetch_add(1, std::memory_order_relaxed);
  }
  return ret;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
                                    ncclConfig_t* /*config*/) {
  LOG_INFO("NCCL compat: ncclCommInitRankConfig(nranks=%d, rank=%d)", nranks, rank);
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* /*config*/) {
  if (!newcomm)
    return ncclInvalidArgument;
  *newcomm = nullptr;
  if (color < 0)
    return ncclSuccess;
  if (!comm)
    return ncclInvalidArgument;

  int nranks = 0;
  int rank = 0;
  ncclResult_t ret = getCommRankAndSize(comm, &rank, &nranks);
  if (ret != ncclSuccess)
    return ret;

  NcclCompatCommState parent_state;
  if (!getCommState(comm, &parent_state)) {
    parent_state.id_hash = reinterpret_cast<uintptr_t>(comm);
    parent_state.multi_process = comm->process_state && comm->process_state->isMultiProcess();
  }
  uint64_t split_seq = nextSplitSeq(comm);
  if (split_seq == 0)
    split_seq = 1;

  ncclUniqueId split_id;
  std::memset(&split_id, 0, sizeof(split_id));
  std::snprintf(split_id.internal, sizeof(split_id.internal), "pccl_split_%016llx_%llu_%d_%d",
                static_cast<unsigned long long>(parent_state.id_hash),
                static_cast<unsigned long long>(split_seq), color, nranks);
  if (!parent_state.multi_process)
    rememberLocallyGeneratedUniqueId(split_id);
  ret = ncclCommInitRank(newcomm, nranks, split_id, rank);
  if (ret == ncclSuccess && parent_state.multi_process) {
    NcclCompatCommState child_state;
    if (getCommState(*newcomm, &child_state)) {
      child_state.multi_process = true;
      rememberCommState(*newcomm, child_state.id_hash, true);
    } else {
      rememberCommState(*newcomm, hashUniqueId(split_id), true);
    }
  }
  return ret;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  LOG_INFO("NCCL compat: ncclCommDestroy");
  forgetCommState(comm);
  ncclResult_t ret = from_pccl(pcclCommDestroy(comm));
  // Auto-finalize when all communicators are destroyed
  if (s_nccl_comm_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    LOG_INFO("NCCL compat: Last comm destroyed, auto-finalizing");
    reapDeferredDeviceBuffers(true);
    pcclFinalize();
  }
  return ret;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  LOG_INFO("NCCL compat: ncclCommAbort (mapped to destroy)");
  return ncclCommDestroy(comm);
}

ncclResult_t ncclCommFinalize(ncclComm_t /*comm*/) {
  return ncclSuccess;
}

// --- Communicator queries ---

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  return from_pccl(pcclCommSize(const_cast<ncclComm_t>(comm), count));
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  return from_pccl(pcclCommRank(const_cast<ncclComm_t>(comm), rank));
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
  if (!comm || !device)
    return ncclInvalidArgument;
  *device = const_cast<ncclComm_t>(comm)->device_id;
  return ncclSuccess;
}

ncclResult_t ncclCommRegister(const ncclComm_t /*comm*/, void* buff, size_t /*size*/, void** handle) {
  if (!handle)
    return ncclInvalidArgument;
  *handle = buff;
  return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t /*comm*/, void* /*handle*/) {
  return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t /*comm*/, ncclResult_t* asyncError) {
  if (asyncError)
    *asyncError = ncclSuccess;
  return ncclSuccess;
}

// --- Error strings ---

const char* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (pccl: device error)";
    case ncclSystemError:
      return "unhandled system error";
    case ncclInternalError:
      return "internal error";
    case ncclInvalidArgument:
      return "invalid argument";
    case ncclInvalidUsage:
      return "invalid usage";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

const char* ncclGetLastError(ncclComm_t /*comm*/) {
  return "";
}

// --- Group semantics (no-op for single-process) ---

ncclResult_t ncclGroupStart() {
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclGroupStart depth=%d pending=%zu", t_group_depth, t_group_ops.size());
  }
  if (t_group_depth == 0) {
    t_group_active = true;
    t_group_ops.clear();
  }
  t_group_depth++;
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclGroupEnd enter depth=%d pending=%zu active=%d",
             t_group_depth,
             t_group_ops.size(),
             t_group_active ? 1 : 0);
  }
  if (!t_group_active || t_group_depth <= 0)
    return ncclInvalidUsage;

  t_group_depth--;
  if (t_group_depth > 0)
    return ncclSuccess;

  std::vector<NcclCompatPendingOp> ops = std::move(t_group_ops);
  t_group_ops.clear();
  t_group_active = false;

  if (ops.empty())
    return ncclSuccess;

  size_t wave_begin = 0;
  size_t wave_idx = 0;
  std::set<int> wave_ranks;
  int wave_comm_size = 0;
  int first_rank = -1;
  ncclResult_t meta_result = getCommRankAndSize(ops[0].comm, &first_rank, &wave_comm_size);
  if (meta_result != ncclSuccess)
    return meta_result;
  wave_ranks.insert(first_rank);

  for (size_t i = 1; i < ops.size(); ++i) {
    int rank = -1;
    int comm_size = 0;
    meta_result = getCommRankAndSize(ops[i].comm, &rank, &comm_size);
    if (meta_result != ncclSuccess)
      return meta_result;

    bool starts_next_wave = !opWaveCompatible(ops[wave_begin], ops[i]) ||
                            comm_size != wave_comm_size ||
                            wave_ranks.count(rank) != 0 ||
                            static_cast<int>(wave_ranks.size()) >= wave_comm_size;
    if (starts_next_wave) {
      ncclResult_t result = runPendingWave(ops, wave_begin, i, wave_idx++);
      if (result != ncclSuccess)
        return result;
      wave_begin = i;
      wave_ranks.clear();
      wave_comm_size = comm_size;
    }
    wave_ranks.insert(rank);
  }

  ncclResult_t result = runPendingWave(ops, wave_begin, ops.size(), wave_idx);
  if (result != ncclSuccess)
    return result;

  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclGroupEnd exit waves=%zu result=%d", wave_idx + 1, static_cast<int>(ncclSuccess));
  }
  return ncclSuccess;
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* /*simInfo*/) {
  return ncclGroupEnd();
}

// --- Reduction operator management (no-op stubs) ---

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* /*scalar*/, ncclDataType_t /*datatype*/,
                                      ncclScalarResidence_t /*residence*/, ncclComm_t /*comm*/) {
  if (op)
    *op = ncclSum;  // Return a valid op handle
  return ncclSuccess;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t /*op*/, ncclComm_t /*comm*/) {
  return ncclSuccess;
}

// --- Collective operations ---

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclAllReduce call count=%zu dtype=%d op=%d comm=%p stream=%p group_active=%d",
             count,
             static_cast<int>(datatype),
             static_cast<int>(op),
             static_cast<void*>(comm),
             stream,
             t_group_active ? 1 : 0);
  }
  ncclResult_t valid = validateDatatype(datatype);
  if (valid != ncclSuccess)
    return valid;
  pccl::CompileOptions options;
  valid = makeCompileOptions(datatype, op, &options, true);
  if (valid != ncclSuccess)
    return valid;
  if (t_group_active) {
    t_group_ops.push_back({NcclCompatOpType::AllReduce, sendbuff, recvbuff, count, datatype, op, 0, comm,
                           stream, deviceForOp(comm)});
    return ncclSuccess;
  }
  int nranks = 0;
  int rank = 0;
  ncclResult_t meta = getCommRankAndSize(comm, &rank, &nranks);
  if (meta != ncclSuccess)
    return meta;
  if (nranks == 1) {
    return runSingleRankCollective({NcclCompatOpType::AllReduce, sendbuff, recvbuff, count, datatype, op,
                                    0, comm, stream, deviceForOp(comm)});
  }
  return submitNcclCompatCollective(comm, "allreduce", sendbuff, recvbuff, count, 0, 1, 1, stream, &options);
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclAllGather call count=%zu dtype=%d comm=%p stream=%p group_active=%d",
             sendcount,
             static_cast<int>(datatype),
             static_cast<void*>(comm),
             stream,
             t_group_active ? 1 : 0);
  }
  ncclResult_t valid = validateDatatype(datatype);
  if (valid != ncclSuccess)
    return valid;
  pccl::CompileOptions options;
  valid = makeCompileOptions(datatype, ncclSum, &options, false);
  if (valid != ncclSuccess)
    return valid;
  int nranks = 0;
  int rank = 0;
  valid = getCommRankAndSize(comm, &rank, &nranks);
  if (valid != ncclSuccess)
    return valid;
  if (nranks == 1) {
    return runSingleRankCollective({NcclCompatOpType::AllGather, sendbuff, recvbuff, sendcount, datatype,
                                    ncclSum, 0, comm, stream, deviceForOp(comm)});
  }

  if (t_group_active) {
    t_group_ops.push_back({NcclCompatOpType::AllGather, sendbuff, recvbuff, sendcount, datatype, ncclSum, 0,
                           comm, stream, deviceForOp(comm)});
    return ncclSuccess;
  }
  return submitNcclCompatCollective(comm, "allgather", sendbuff, recvbuff, sendcount, 0, 1, nranks, stream,
                                    &options);
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclReduceScatter call count=%zu dtype=%d op=%d comm=%p stream=%p group_active=%d",
             recvcount,
             static_cast<int>(datatype),
             static_cast<int>(op),
             static_cast<void*>(comm),
             stream,
             t_group_active ? 1 : 0);
  }
  // NCCL and PCCL now share the same semantics: recvcount = per-rank element count.
  ncclResult_t valid = validateDatatype(datatype);
  if (valid != ncclSuccess)
    return valid;
  pccl::CompileOptions options;
  valid = makeCompileOptions(datatype, op, &options, true);
  if (valid != ncclSuccess)
    return valid;
  int nranks = 0;
  int rank = 0;
  valid = getCommRankAndSize(comm, &rank, &nranks);
  if (valid != ncclSuccess)
    return valid;
  if (nranks == 1) {
    return runSingleRankCollective({NcclCompatOpType::ReduceScatter, sendbuff, recvbuff, recvcount,
                                    datatype, op, 0, comm, stream, deviceForOp(comm)});
  }
  if (t_group_active) {
    t_group_ops.push_back({NcclCompatOpType::ReduceScatter, sendbuff, recvbuff, recvcount, datatype, op, 0,
                           comm, stream, deviceForOp(comm)});
    return ncclSuccess;
  }
  return submitNcclCompatCollective(comm, "reducescatter", sendbuff, recvbuff, recvcount, 0, nranks, 1, stream,
                                    &options);
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                           int root, ncclComm_t comm, cudaStream_t stream) {
  if (pccl::debugAsyncSubmitEnabled()) {
    LOG_INFO("ASYNCDBG ncclBroadcast call count=%zu dtype=%d root=%d comm=%p stream=%p group_active=%d",
             count,
             static_cast<int>(datatype),
             root,
             static_cast<void*>(comm),
             stream,
             t_group_active ? 1 : 0);
  }
  ncclResult_t valid = validateDatatype(datatype);
  if (valid != ncclSuccess)
    return valid;
  int nranks = 0;
  int rank = 0;
  ncclResult_t meta = getCommRankAndSize(comm, &rank, &nranks);
  if (meta != ncclSuccess)
    return meta;
  if (root < 0 || root >= nranks)
    return ncclInvalidArgument;
  if (!recvbuff)
    return ncclInvalidArgument;
  if (!(sendbuff ? sendbuff : recvbuff))
    return ncclInvalidArgument;
  if (t_group_active) {
    t_group_ops.push_back({NcclCompatOpType::Broadcast, sendbuff, recvbuff, count, datatype, ncclSum, root,
                           comm, stream, deviceForOp(comm)});
    return ncclSuccess;
  }
  if (nranks == 1) {
    return runSingleRankCollective({NcclCompatOpType::Broadcast, sendbuff, recvbuff, count, datatype,
                                    ncclSum, root, comm, stream, deviceForOp(comm)});
  }
  return runBroadcastOp({NcclCompatOpType::Broadcast, sendbuff, recvbuff, count, datatype, ncclSum, root,
                         comm, stream, deviceForOp(comm)});
}

ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                        ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)op;
  (void)stream;
  ncclResult_t valid = validateDatatype(datatype);
  if (valid != ncclSuccess)
    return valid;
  valid = validateReduceRoot({NcclCompatOpType::Reduce, nullptr, nullptr, count, datatype, op, root,
                              comm, stream, deviceForOp(comm)});
  if (valid != ncclSuccess)
    return valid;
  return runReduceOp({NcclCompatOpType::Reduce, sendbuff, recvbuff, count, datatype, op, root,
                      comm, stream, deviceForOp(comm)});
}

ncclResult_t ncclSend(const void* /*sendbuff*/, size_t /*count*/, ncclDataType_t /*datatype*/, int /*peer*/,
                      ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
  LOG_WARN("NCCL compat: ncclSend not supported");
  return ncclInternalError;
}

ncclResult_t ncclRecv(void* /*recvbuff*/, size_t /*count*/, ncclDataType_t /*datatype*/, int /*peer*/,
                      ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
  LOG_WARN("NCCL compat: ncclRecv not supported");
  return ncclInternalError;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  if (!comm || ndev <= 0)
    return ncclInvalidArgument;
  ncclUniqueId id;
  std::memset(&id, 0, sizeof(id));
  rememberLocallyGeneratedUniqueId(id);
  for (int r = 0; r < ndev; ++r) {
    int dev = devlist ? devlist[r] : r;
    if (cudaSetDevice(dev) != cudaSuccess)
      return ncclUnhandledCudaError;
    ncclResult_t ret = ncclCommInitRank(&comm[r], ndev, id, r);
    if (ret != ncclSuccess)
      return ret;
    comm[r]->device_id = dev;
  }
  return ncclSuccess;
}

ncclResult_t pncclGetVersion(int* version) PCCL_NCCL_ALIAS(ncclGetVersion);
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId) PCCL_NCCL_ALIAS(ncclGetUniqueId);
ncclResult_t pncclMemAlloc(void** ptr, size_t size) PCCL_NCCL_ALIAS(ncclMemAlloc);
ncclResult_t pncclMemFree(void* ptr) PCCL_NCCL_ALIAS(ncclMemFree);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
    PCCL_NCCL_ALIAS(ncclCommInitRank);
ncclResult_t pncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
                                     ncclConfig_t* config) PCCL_NCCL_ALIAS(ncclCommInitRankConfig);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) PCCL_NCCL_ALIAS(ncclCommInitAll);
ncclResult_t pncclCommFinalize(ncclComm_t comm) PCCL_NCCL_ALIAS(ncclCommFinalize);
ncclResult_t pncclCommDestroy(ncclComm_t comm) PCCL_NCCL_ALIAS(ncclCommDestroy);
ncclResult_t pncclCommAbort(ncclComm_t comm) PCCL_NCCL_ALIAS(ncclCommAbort);
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config)
    PCCL_NCCL_ALIAS(ncclCommSplit);
const char* pncclGetErrorString(ncclResult_t result) PCCL_NCCL_ALIAS(ncclGetErrorString);
const char* pncclGetLastError(ncclComm_t comm) PCCL_NCCL_ALIAS(ncclGetLastError);
ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)
    PCCL_NCCL_ALIAS(ncclCommGetAsyncError);
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count) PCCL_NCCL_ALIAS(ncclCommCount);
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device) PCCL_NCCL_ALIAS(ncclCommCuDevice);
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank) PCCL_NCCL_ALIAS(ncclCommUserRank);
ncclResult_t pncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle)
    PCCL_NCCL_ALIAS(ncclCommRegister);
ncclResult_t pncclCommDeregister(const ncclComm_t comm, void* handle) PCCL_NCCL_ALIAS(ncclCommDeregister);
ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                                       ncclScalarResidence_t residence, ncclComm_t comm)
    PCCL_NCCL_ALIAS(ncclRedOpCreatePreMulSum);
ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) PCCL_NCCL_ALIAS(ncclRedOpDestroy);
ncclResult_t pncclGroupStart() PCCL_NCCL_ALIAS(ncclGroupStart);
ncclResult_t pncclGroupEnd() PCCL_NCCL_ALIAS(ncclGroupEnd);
ncclResult_t pncclGroupSimulateEnd(ncclSimInfo_t* simInfo) PCCL_NCCL_ALIAS(ncclGroupSimulateEnd);
ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                            ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
    PCCL_NCCL_ALIAS(ncclAllReduce);
ncclResult_t pncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream) PCCL_NCCL_ALIAS(ncclAllGather);
ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
    PCCL_NCCL_ALIAS(ncclReduceScatter);
ncclResult_t pncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                            int root, ncclComm_t comm, cudaStream_t stream) PCCL_NCCL_ALIAS(ncclBroadcast);
ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) PCCL_NCCL_ALIAS(ncclBcast);
ncclResult_t pncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                         ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
    PCCL_NCCL_ALIAS(ncclReduce);
ncclResult_t pncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
                       ncclComm_t comm, cudaStream_t stream) PCCL_NCCL_ALIAS(ncclSend);
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                       ncclComm_t comm, cudaStream_t stream) PCCL_NCCL_ALIAS(ncclRecv);

}  // extern "C"
