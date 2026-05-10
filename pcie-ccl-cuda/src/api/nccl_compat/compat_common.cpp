#include "nccl_compat_internal.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../core/comm.hpp"
#include "../../util/log.hpp"

namespace {

std::mutex s_compat_mode_mutex;
std::unordered_set<uint64_t> s_locally_generated_unique_ids;
std::unordered_map<std::string, bool> s_compat_mode_cache;

std::mutex s_comm_state_mutex;
std::unordered_map<ncclComm_t, NcclCompatCommState> s_comm_states;

struct DeferredDeviceBuffer {
  void* ptr;
  cudaEvent_t done;
  int device;
};

std::mutex s_deferred_buffer_mutex;
std::vector<DeferredDeviceBuffer> s_deferred_buffers;

struct CompatModeRendezvousHeader {
  uint32_t magic;
  uint32_t version;
  int nranks;
};

struct CompatModeRendezvousSlot {
  int present;
  int pid;
};

constexpr uint32_t kCompatModeMagic = 0x50434d52u;  // PCMR
constexpr uint32_t kCompatModeVersion = 1;

std::string compatModeCacheKey(uint64_t id_hash, int nranks) {
  char key[64];
  std::snprintf(key, sizeof(key), "%016llx:%d", static_cast<unsigned long long>(id_hash), nranks);
  return std::string(key);
}

bool lookupCachedCompatMode(uint64_t id_hash, int nranks, bool* multi_process) {
  std::lock_guard<std::mutex> lock(s_compat_mode_mutex);
  auto it = s_compat_mode_cache.find(compatModeCacheKey(id_hash, nranks));
  if (it == s_compat_mode_cache.end())
    return false;
  *multi_process = it->second;
  return true;
}

void cacheCompatMode(uint64_t id_hash, int nranks, bool multi_process) {
  std::lock_guard<std::mutex> lock(s_compat_mode_mutex);
  s_compat_mode_cache[compatModeCacheKey(id_hash, nranks)] = multi_process;
}

bool isLocallyGeneratedUniqueId(uint64_t id_hash) {
  std::lock_guard<std::mutex> lock(s_compat_mode_mutex);
  return s_locally_generated_unique_ids.count(id_hash) != 0;
}

bool envForcesCompatMode(bool* multi_process) {
  const char* mode = std::getenv("PCCL_NCCL_COMPAT_MODE");
  if (!mode || mode[0] == '\0' || std::strcmp(mode, "auto") == 0)
    return false;
  if (std::strcmp(mode, "sp") == 0 || std::strcmp(mode, "single") == 0 ||
      std::strcmp(mode, "single-process") == 0) {
    *multi_process = false;
    return true;
  }
  if (std::strcmp(mode, "mp") == 0 || std::strcmp(mode, "multi") == 0 ||
      std::strcmp(mode, "multi-process") == 0) {
    *multi_process = true;
    return true;
  }
  LOG_WARN("NCCL compat: ignoring invalid PCCL_NCCL_COMPAT_MODE=%s (expected auto|sp|mp)", mode);
  return false;
}

size_t compatModeRendezvousBytes(int nranks) {
  return sizeof(CompatModeRendezvousHeader) + static_cast<size_t>(nranks) * sizeof(CompatModeRendezvousSlot);
}

CompatModeRendezvousSlot* compatModeSlots(void* base) {
  return reinterpret_cast<CompatModeRendezvousSlot*>(static_cast<char*>(base) + sizeof(CompatModeRendezvousHeader));
}

bool pidIsAlive(int pid) {
  if (pid <= 0)
    return false;
  if (pid == static_cast<int>(getpid()))
    return true;
  if (kill(static_cast<pid_t>(pid), 0) == 0)
    return true;
  return errno != ESRCH;
}

ncclResult_t copyOnStream(ncclComm_t comm, const void* src, void* dst, size_t bytes, cudaStream_t stream) {
  if (!comm || (!src && bytes > 0) || (!dst && bytes > 0))
    return ncclInvalidArgument;
  int current_device = -1;
  cudaGetDevice(&current_device);
  if (comm->device_id >= 0 && cudaSetDevice(comm->device_id) != cudaSuccess) {
    cudaGetLastError();
    return ncclUnhandledCudaError;
  }
  cudaError_t status = cudaSuccess;
  if (bytes > 0 && src != dst)
    status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
  if (current_device >= 0)
    cudaSetDevice(current_device);
  if (status != cudaSuccess) {
    LOG_ERROR("NCCL compat: single-rank device copy failed: %s", cudaGetErrorString(status));
    cudaGetLastError();
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t from_pccl(pcclResult_t r) {
  switch (r) {
    case pcclSuccess:
      return ncclSuccess;
    case pcclInvalidArgument:
      return ncclInvalidArgument;
    case pcclSystemError:
      return ncclSystemError;
    case pcclInternalError:
      return ncclInternalError;
    case pcclDeviceError:
      return ncclUnhandledCudaError;
    default:
      return ncclInternalError;
  }
}

long parsePositiveEnvLong(const char* name, long fallback) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0)
    return fallback;
  return parsed;
}

uint64_t hashUniqueId(const ncclUniqueId& commId) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : commId.internal) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ull;
  }
  return h;
}

void rememberLocallyGeneratedUniqueId(const ncclUniqueId& commId) {
  std::lock_guard<std::mutex> lock(s_compat_mode_mutex);
  s_locally_generated_unique_ids.insert(hashUniqueId(commId));
}

void rememberCommState(ncclComm_t comm, uint64_t id_hash, bool multi_process) {
  if (!comm)
    return;
  std::lock_guard<std::mutex> lock(s_comm_state_mutex);
  s_comm_states[comm] = NcclCompatCommState{id_hash, multi_process, 0, 0};
}

void forgetCommState(ncclComm_t comm) {
  if (!comm)
    return;
  std::lock_guard<std::mutex> lock(s_comm_state_mutex);
  s_comm_states.erase(comm);
}

bool getCommState(ncclComm_t comm, NcclCompatCommState* state) {
  if (!comm || !state)
    return false;
  std::lock_guard<std::mutex> lock(s_comm_state_mutex);
  auto it = s_comm_states.find(comm);
  if (it == s_comm_states.end())
    return false;
  *state = it->second;
  return true;
}

uint64_t nextBroadcastSeq(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(s_comm_state_mutex);
  auto it = s_comm_states.find(comm);
  if (it == s_comm_states.end())
    return 0;
  return ++it->second.broadcast_seq;
}

uint64_t nextSplitSeq(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(s_comm_state_mutex);
  auto it = s_comm_states.find(comm);
  if (it == s_comm_states.end())
    return 0;
  return ++it->second.split_seq;
}

bool waitForShmSize(int fd, size_t expected_size, int timeout_ms) {
  auto start = std::chrono::steady_clock::now();
  while (true) {
    struct stat st {};
    if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= expected_size)
      return true;
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    if (elapsed_ms > timeout_ms)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

ncclResult_t detectCompatMultiProcessByRankPids(const ncclUniqueId& commId, int nranks, int rank,
                                                bool* multi_process) {
  if (!multi_process)
    return ncclInvalidArgument;
  if (envForcesCompatMode(multi_process)) {
    LOG_INFO("NCCL compat: PCCL_NCCL_COMPAT_MODE forced %s mode", *multi_process ? "multi-process" : "single-process");
    return ncclSuccess;
  }
  if (nranks <= 1) {
    *multi_process = false;
    return ncclSuccess;
  }

  const uint64_t id_hash = hashUniqueId(commId);
  if (lookupCachedCompatMode(id_hash, nranks, multi_process))
    return ncclSuccess;
  *multi_process = true;
  cacheCompatMode(id_hash, nranks, true);
  LOG_INFO("NCCL compat: defaulting to multi-process mode rank=%d nranks=%d; set PCCL_NCCL_COMPAT_MODE=sp for "
           "single-process tests",
           rank, nranks);
  return ncclSuccess;
}

const char* opTypeName(NcclCompatOpType type) {
  switch (type) {
    case NcclCompatOpType::AllReduce:
      return "AllReduce";
    case NcclCompatOpType::AllGather:
      return "AllGather";
    case NcclCompatOpType::ReduceScatter:
      return "ReduceScatter";
    case NcclCompatOpType::Broadcast:
      return "Broadcast";
    case NcclCompatOpType::Reduce:
      return "Reduce";
  }
  return "Unknown";
}

size_t datatypeSize(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return 0;
  }
}

bool datatypeToScalar(ncclDataType_t datatype, pccl::ScalarDataType* out) {
  if (!out)
    return false;
  switch (datatype) {
    case ncclInt8:
      *out = pccl::ScalarDataType::Int8;
      return true;
    case ncclUint8:
      *out = pccl::ScalarDataType::Uint8;
      return true;
    case ncclInt32:
      *out = pccl::ScalarDataType::Int32;
      return true;
    case ncclUint32:
      *out = pccl::ScalarDataType::Uint32;
      return true;
    case ncclInt64:
      *out = pccl::ScalarDataType::Int64;
      return true;
    case ncclUint64:
      *out = pccl::ScalarDataType::Uint64;
      return true;
    case ncclFloat16:
      *out = pccl::ScalarDataType::Float16;
      return true;
    case ncclFloat32:
      *out = pccl::ScalarDataType::Float32;
      return true;
    case ncclFloat64:
      *out = pccl::ScalarDataType::Float64;
      return true;
    case ncclBfloat16:
      *out = pccl::ScalarDataType::Bfloat16;
      return true;
    default:
      return false;
  }
}

bool redOpToPccl(ncclRedOp_t op, pccl::ReduceOpKind* out) {
  if (!out)
    return false;
  switch (op) {
    case ncclSum:
      *out = pccl::ReduceOpKind::Sum;
      return true;
    case ncclMax:
      *out = pccl::ReduceOpKind::Max;
      return true;
    case ncclMin:
      *out = pccl::ReduceOpKind::Min;
      return true;
    default:
      return false;
  }
}

ncclResult_t makeCompileOptions(ncclDataType_t datatype, ncclRedOp_t op, pccl::CompileOptions* options,
                                bool reduction) {
  if (!options)
    return ncclInvalidArgument;
  size_t elem_size = datatypeSize(datatype);
  pccl::ScalarDataType scalar_type;
  if (elem_size == 0 || !datatypeToScalar(datatype, &scalar_type)) {
    LOG_ERROR("NCCL compat: unsupported datatype=%d", static_cast<int>(datatype));
    return ncclInvalidArgument;
  }
  pccl::ReduceOpKind reduce_op = pccl::ReduceOpKind::Sum;
  if (reduction && !redOpToPccl(op, &reduce_op)) {
    LOG_ERROR("NCCL compat: unsupported reduction op=%d", static_cast<int>(op));
    return ncclInvalidArgument;
  }
  *options = pccl::CompileOptions();
  options->element_size = elem_size;
  options->scalar_type = scalar_type;
  options->reduce_op = reduce_op;
  return ncclSuccess;
}

ncclResult_t validateDatatype(ncclDataType_t datatype) {
  if (datatypeSize(datatype) == 0) {
    LOG_ERROR("NCCL compat: unsupported datatype=%d", static_cast<int>(datatype));
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

void reapDeferredDeviceBuffers(bool block) {
  int current_device = -1;
  cudaGetDevice(&current_device);
  std::vector<DeferredDeviceBuffer> ready;
  {
    std::lock_guard<std::mutex> lock(s_deferred_buffer_mutex);
    auto it = s_deferred_buffers.begin();
    while (it != s_deferred_buffers.end()) {
      if (it->device >= 0)
        cudaSetDevice(it->device);
      cudaError_t status = block ? cudaEventSynchronize(it->done) : cudaEventQuery(it->done);
      if (status == cudaSuccess) {
        ready.push_back(*it);
        it = s_deferred_buffers.erase(it);
      } else if (status == cudaErrorNotReady && !block) {
        ++it;
      } else {
        cudaGetLastError();
        ready.push_back(*it);
        it = s_deferred_buffers.erase(it);
      }
    }
  }
  for (const auto& item : ready) {
    if (item.device >= 0)
      cudaSetDevice(item.device);
    cudaEventDestroy(item.done);
    cudaFree(item.ptr);
  }
  if (current_device >= 0)
    cudaSetDevice(current_device);
}

ncclResult_t deferDeviceFree(void* ptr, cudaStream_t stream, int device) {
  if (!ptr)
    return ncclSuccess;
  int current_device = -1;
  cudaGetDevice(&current_device);
  int owner_device = device;
  if (owner_device < 0)
    owner_device = current_device;
  if (owner_device >= 0 && cudaSetDevice(owner_device) != cudaSuccess)
    return ncclUnhandledCudaError;

  cudaEvent_t done = nullptr;
  cudaError_t status = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
  if (status == cudaSuccess)
    status = cudaEventRecord(done, stream);
  if (status != cudaSuccess) {
    if (done)
      cudaEventDestroy(done);
    cudaGetLastError();
    cudaStreamSynchronize(stream);
    cudaFree(ptr);
    if (current_device >= 0)
      cudaSetDevice(current_device);
    return ncclUnhandledCudaError;
  }
  {
    std::lock_guard<std::mutex> lock(s_deferred_buffer_mutex);
    s_deferred_buffers.push_back({ptr, done, owner_device});
  }
  if (current_device >= 0)
    cudaSetDevice(current_device);
  return ncclSuccess;
}

ncclResult_t runSingleRankCollective(const NcclCompatPendingOp& op) {
  size_t elem_size = datatypeSize(op.datatype);
  if (elem_size == 0)
    return ncclInvalidArgument;
  switch (op.type) {
    case NcclCompatOpType::AllReduce:
    case NcclCompatOpType::AllGather:
    case NcclCompatOpType::ReduceScatter:
      return copyOnStream(op.comm, op.sendbuff, op.recvbuff, op.count * elem_size, op.stream);
    case NcclCompatOpType::Broadcast: {
      const void* src = op.sendbuff ? op.sendbuff : op.recvbuff;
      return copyOnStream(op.comm, src, op.recvbuff, op.count * elem_size, op.stream);
    }
    case NcclCompatOpType::Reduce:
      if (op.root != 0)
        return ncclInvalidArgument;
      return copyOnStream(op.comm, op.sendbuff, op.recvbuff, op.count * elem_size, op.stream);
  }
  return ncclInternalError;
}
