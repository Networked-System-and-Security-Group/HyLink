#include "process_state.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>

#include "../executor/h2h_engine.hpp"
#include "../hal/device_rt.hpp"
#include "../memory/host_pool.hpp"
#include "../memory/shared_state.hpp"
#include "../transport/bootstrap.hpp"
#include "../transport/transport.hpp"
#include "../util/log.hpp"
#include "config.hpp"

#if defined(PCCL_DEVICE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace pccl {

namespace {

constexpr int kIpcInput = 0;
constexpr int kIpcOutput = 1;
constexpr int kIpcInternal = 2;
// Per-rank preallocated internal device buffer. Override with
// PCCL_INTERNAL_BUFFER_BYTES (bytes) when large collectives need more scratch.
const size_t kInternalBufferBytes = [] {
  const char* value = std::getenv("PCCL_INTERNAL_BUFFER_BYTES");
  if (value && value[0] != '\0') {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end != value && parsed > 0)
      return static_cast<size_t>(parsed);
  }
  return static_cast<size_t>(3ULL << 30);
}();

int remoteBufferTimeoutMs() {
  const char* value = std::getenv("PCCL_REMOTE_BUFFER_TIMEOUT_MS");
  if (!value || value[0] == '\0')
    return 120000;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0)
    return 120000;
  return static_cast<int>(parsed);
}

int commIndex(int comm_id) {
  return (comm_id >= 0 && comm_id < PCCL_MAX_CONCURRENT_COMMS) ? comm_id : 0;
}

}  // namespace

ProcessState::ProcessState() : initialized_(false), config_(nullptr) {}

ProcessState::~ProcessState() {
  finalize();
}

ProcessState& ProcessState::instance() {
  // Keep the process-wide state alive until process teardown.  The explicit
  // pcclFinalize path still releases CUDA resources; at exit, running the
  // static destructor can deadlock inside CUDA runtime cleanup after nccl-tests
  // has already concluded.
  static ProcessState* s_instance = new ProcessState();
  return *s_instance;
}

int ProcessState::initialize(const Config& config) {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    LOG_WARN("ProcessState: Already initialized");
    return 0;
  }

  config_ = &config;

  // Decide once whether device-side CUDA IPC buffers can ever be consumed. They
  // are only opened by a peer when a "_direct" IR is selected, and a direct IR is
  // only a candidate when some GPU pair allows direct peer access (see
  // IRLoader::findIRFile). On a pure-relay topology (no PCIe-switch P2P) every
  // collective stages through HOST memory, so the published IPC handles are dead
  // weight -- and republishing them per collective costs a cudaIpcGetMemHandle
  // each time (PyTorch packs many tensors per allocator segment, so the cheap
  // "same pointer" early-out rarely fires). Skip publishing in that case.
  {
    bool any_direct_peer = false;
    if (config.multi_process) {
      const auto& topo = config.topology;
      for (int i = 0; i < topo.num_gpus && !any_direct_peer; ++i)
        for (int j = 0; j < topo.num_gpus; ++j) {
          if (i != j && topo.pairDirectPeerAllowed(i, j)) {
            any_direct_peer = true;
            break;
          }
        }
    }
    device_ipc_needed_ = any_direct_peer;
    const char* disable_env = std::getenv("PCCL_DISABLE_DEVICE_IPC");
    if (disable_env && std::atoi(disable_env) != 0)
      device_ipc_needed_ = false;
    const char* force_env = std::getenv("PCCL_FORCE_DEVICE_IPC");
    if (force_env && std::atoi(force_env) != 0)
      device_ipc_needed_ = true;
    LOG_INFO("ProcessState: device IPC publishing %s (any_direct_peer=%d, multi_process=%d)",
             device_ipc_needed_ ? "ENABLED" : "DISABLED (pure-relay topology)",
             any_direct_peer ? 1 : 0, config.multi_process ? 1 : 0);
  }

  rank_input_buffers_.assign(config.nranks, nullptr);
  rank_output_buffers_.assign(config.nranks, nullptr);
  rank_internal_buffers_.assign(config.nranks, nullptr);
  rank_internal_sizes_.assign(config.nranks, 0);
  for (int comm = 0; comm < PCCL_MAX_CONCURRENT_COMMS; ++comm) {
    for (int kind = 0; kind < 3; ++kind) {
      local_ipc_[comm][kind].assign(config.nranks, LocalDeviceBufferIpc{});
      remote_ipc_[comm][kind].assign(config.nranks, RemoteDeviceBufferIpc{});
    }
  }

  if (config.multi_process) {
    // Multi-process mode: shared memory for state and host buffers
    bool is_creator = (config.local_rank == 0);

    shared_state_ =
        std::make_unique<SharedStateManager>(config.nranks, config.unique_id, is_creator);

    host_pool_ = std::make_unique<HostMemoryPool>(
        config.nranks, config.host_buffer_size, &config.topology,
        config.unique_id, is_creator);

    // Barrier: wait for all processes to finish shared memory setup
    if (!shared_state_->barrier("ProcessState::initialize", config.local_rank, -1))
      return -1;

    LOG_INFO("ProcessState: Multi-process shared memory ready (is_creator=%d)", is_creator);
  } else {
    // Single-process mode (unchanged)
    shared_state_ = std::make_unique<SharedStateManager>(config.nranks);
    host_pool_ = std::make_unique<HostMemoryPool>(
        config.nranks, config.host_buffer_size, &config.topology);
  }

  // Create H2H engine — full worker count per process.
  // Workers without H2H tasks sleep on condition variables (zero CPU cost).
  // Active worker count is naturally bounded by IR task distribution.
  h2h_engine_ = std::make_unique<H2HEngine>(shared_state_.get(), &config.topology);
  h2h_engine_->initialize();

  // Bootstrap transport (if multi-node)
  if (!config.bootstrap_addr.empty()) {
    transport_.reset(bootstrapTransport(config.bootstrap_addr, 0, config.nranks));
  }

  initialized_ = true;
  LOG_INFO("ProcessState: Initialized (nranks=%d, multi_process=%d)", config.nranks, config.multi_process);
  return 0;
}

void ProcessState::finalize() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (!initialized_)
    return;

  if (h2h_engine_)
    h2h_engine_->shutdown();

  {
    std::lock_guard<std::mutex> buffers_lock(buffer_mutex_);
    for (int comm = 0; comm < PCCL_MAX_CONCURRENT_COMMS; ++comm) {
      for (int kind = 0; kind < 3; ++kind) {
        for (auto& remote : remote_ipc_[comm][kind]) {
          releaseRemoteDeviceBuffer(remote, true);
        }
      }
    }
    for (size_t r = 0; r < rank_internal_buffers_.size(); ++r) {
      if (rank_internal_buffers_[r]) {
        devFree(rank_internal_buffers_[r]);
        rank_internal_buffers_[r] = nullptr;
        rank_internal_sizes_[r] = 0;
      }
    }
    rank_input_buffers_.clear();
    rank_output_buffers_.clear();
    rank_internal_buffers_.clear();
    rank_internal_sizes_.clear();
    for (int comm = 0; comm < PCCL_MAX_CONCURRENT_COMMS; ++comm) {
      for (int kind = 0; kind < 3; ++kind) {
        local_ipc_[comm][kind].clear();
        remote_ipc_[comm][kind].clear();
      }
    }
  }

  transport_.reset();
  h2h_engine_.reset();
  host_pool_.reset();
  shared_state_.reset();
  config_ = nullptr;
  initialized_ = false;
  LOG_INFO("ProcessState: Finalized");
}

bool ProcessState::isMultiProcess() const {
  return config_ && config_->multi_process;
}

bool ProcessState::registerDeviceBuffers(int comm_id, int rank, int device_id, const void* input, void* output) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (rank < 0 || rank >= static_cast<int>(rank_input_buffers_.size())) {
    LOG_ERROR("ProcessState: invalid rank %d for device buffer registration (nranks=%zu)", rank,
              rank_input_buffers_.size());
    return false;
  }
  rank_input_buffers_[rank] = input;
  rank_output_buffers_[rank] = output;
  if (config_ && config_->multi_process && device_ipc_needed_) {
    if (input && !publishDeviceBufferLocked(comm_id, kIpcInput, rank, input, device_id))
      return false;
    if (output && !publishDeviceBufferLocked(comm_id, kIpcOutput, rank, output, device_id))
      return false;
  }
  return true;
}

const void* ProcessState::getInputBufferForRank(int rank) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (rank < 0 || rank >= static_cast<int>(rank_input_buffers_.size())) {
    LOG_ERROR("ProcessState: invalid rank %d for input buffer lookup (nranks=%zu)", rank,
              rank_input_buffers_.size());
    return nullptr;
  }
  return rank_input_buffers_[rank];
}

void* ProcessState::getOutputBufferForRank(int rank) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (rank < 0 || rank >= static_cast<int>(rank_output_buffers_.size())) {
    LOG_ERROR("ProcessState: invalid rank %d for output buffer lookup (nranks=%zu)", rank,
              rank_output_buffers_.size());
    return nullptr;
  }
  return rank_output_buffers_[rank];
}

int ProcessState::initializeInternalBuffer(int comm_id, int rank, int device_id) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (rank < 0 || rank >= static_cast<int>(rank_internal_buffers_.size())) {
    LOG_ERROR("ProcessState: invalid rank %d for internal buffer initialization (nranks=%zu)", rank,
              rank_internal_buffers_.size());
    return -1;
  }

  if (rank_internal_buffers_[rank] && rank_internal_sizes_[rank] >= kInternalBufferBytes) {
    if (config_ && config_->multi_process && device_ipc_needed_ &&
        !publishDeviceBufferLocked(comm_id, kIpcInternal, rank, rank_internal_buffers_[rank], device_id)) {
      return -1;
    }
    return 0;
  }

  if (rank_internal_buffers_[rank]) {
    LOG_ERROR("ProcessState: internal buffer for rank %d was already initialized with insufficient size=%zu",
              rank, rank_internal_sizes_[rank]);
    return -1;
  }

  devSetDevice(device_id);
  void* ptr = nullptr;
  if (devMalloc(&ptr, kInternalBufferBytes) != devSuccess || !ptr) {
    LOG_ERROR("ProcessState: Failed to preallocate internal device buffer for rank %d device=%d bytes=%zu",
              rank, device_id, kInternalBufferBytes);
    return -1;
  }

  rank_internal_buffers_[rank] = ptr;
  rank_internal_sizes_[rank] = kInternalBufferBytes;
  if (config_ && config_->multi_process && device_ipc_needed_ &&
      !publishDeviceBufferLocked(comm_id, kIpcInternal, rank, ptr, device_id)) {
    devFree(ptr);
    rank_internal_buffers_[rank] = nullptr;
    rank_internal_sizes_[rank] = 0;
    return -1;
  }

  LOG_INFO("ProcessState: Preallocated internal device buffer rank=%d device=%d bytes=%zu ptr=%p",
           rank, device_id, kInternalBufferBytes, ptr);
  return 0;
}

int ProcessState::ensureInternalBuffer(int rank, int device_id, size_t bytes) {
  (void)device_id;
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (rank < 0 || rank >= static_cast<int>(rank_internal_buffers_.size())) {
    LOG_ERROR("ProcessState: invalid rank %d for internal buffer check (nranks=%zu)", rank,
              rank_internal_buffers_.size());
    return -1;
  }
  if (bytes == 0) {
    LOG_ERROR("ProcessState: invalid zero-size internal buffer request for rank %d", rank);
    return -1;
  }

  if (rank_internal_sizes_[rank] >= bytes && rank_internal_buffers_[rank])
    return 0;

  LOG_ERROR("ProcessState: internal buffer budget exceeded rank=%d requested=%zu allocated=%zu budget=%zu",
            rank, bytes, rank_internal_sizes_[rank], kInternalBufferBytes);
  return -1;
}

void* ProcessState::getInternalBufferForRank(int rank) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (rank < 0 || rank >= static_cast<int>(rank_internal_buffers_.size())) {
    LOG_ERROR("ProcessState: invalid rank %d for internal buffer lookup (nranks=%zu)", rank,
              rank_internal_buffers_.size());
    return nullptr;
  }
  return rank_internal_buffers_[rank];
}

bool ProcessState::publishDeviceBufferLocked(int comm_id, int kind, int rank, const void* ptr, int device_id) {
  int comm = commIndex(comm_id);
  if (!shared_state_ || kind < 0 || kind >= 3 || rank < 0 ||
      rank >= static_cast<int>(local_ipc_[comm][kind].size()) || !ptr) {
    return false;
  }
  auto& local = local_ipc_[comm][kind][rank];
  if (local.published && local.ptr == ptr && local.device_id == device_id)
    return true;

#if defined(PCCL_DEVICE_CUDA)
  cudaPointerAttributes attr = {};
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    LOG_ERROR("ProcessState: cudaPointerGetAttributes failed kind=%d rank=%d ptr=%p: %s", kind, rank, ptr,
              cudaGetErrorString(err));
    cudaGetLastError();
    return false;
  }

  void* range_start = nullptr;
  size_t range_size = 0;
  CUresult cu_err = cuPointerGetAttribute(&range_start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                          reinterpret_cast<CUdeviceptr>(ptr));
  if (cu_err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(cu_err, &err_str);
    LOG_ERROR("ProcessState: cuPointerGetAttribute(RANGE_START) failed kind=%d rank=%d ptr=%p: %s", kind, rank, ptr,
              err_str ? err_str : "?");
    return false;
  }
  cu_err = cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                                 reinterpret_cast<CUdeviceptr>(ptr));
  if (cu_err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(cu_err, &err_str);
    LOG_ERROR("ProcessState: cuPointerGetAttribute(RANGE_SIZE) failed kind=%d rank=%d ptr=%p: %s", kind, rank, ptr,
              err_str ? err_str : "?");
    return false;
  }

  const void* base = range_start ? range_start : (attr.devicePointer ? attr.devicePointer : ptr);
  cudaIpcMemHandle_t handle;
  err = cudaIpcGetMemHandle(&handle, const_cast<void*>(base));
  if (err != cudaSuccess) {
    LOG_ERROR("ProcessState: cudaIpcGetMemHandle failed kind=%d rank=%d base=%p ptr=%p: %s", kind, rank, base, ptr,
              cudaGetErrorString(err));
    cudaGetLastError();
    return false;
  }

  uint64_t base_addr = reinterpret_cast<uint64_t>(base);
  uint64_t ptr_addr = reinterpret_cast<uint64_t>(ptr);
  uint64_t offset = ptr_addr >= base_addr ? ptr_addr - base_addr : 0;
  uint64_t size = std::max<uint64_t>(1, range_size);
  if (kind == kIpcInternal && rank < static_cast<int>(rank_internal_sizes_.size())) {
    size = std::max<uint64_t>(1, rank_internal_sizes_[rank]);
  }

  shared_state_->publishDeviceIpcBuffer(comm, kind, rank, base_addr, offset, size, &handle, sizeof(handle));
  local.ptr = ptr;
  local.base_addr = base_addr;
  local.offset = offset;
  local.size = size;
  local.published = true;
  local.device_id = device_id;
  LOG_INFO("ProcessState: Published CUDA IPC buffer comm=%d kind=%d rank=%d device=%d ptr=%p base=%p offset=%llu",
           comm, kind, rank, device_id, ptr, base, static_cast<unsigned long long>(offset));
  return true;
#else
  (void)device_id;
  LOG_ERROR("ProcessState: multi-process direct device IPC is only implemented for CUDA");
  return false;
#endif
}

bool ProcessState::isRemoteDeviceBufferReady(int comm_id, int local_rank, int kind, int rank) const {
  if (!config_ || !config_->multi_process)
    return false;
  if (rank == local_rank)
    return true;
  int comm = commIndex(comm_id);
  if (!shared_state_ || kind < 0 || kind >= 3 || rank < 0 ||
      rank >= static_cast<int>(remote_ipc_[comm][kind].size())) {
    return false;
  }
  DeviceIpcBufferRecord record;
  return shared_state_->readDeviceIpcBuffer(comm, kind, rank, &record);
}

void* ProcessState::getRemoteDeviceBuffer(int comm_id, int local_rank, int local_device_id, int kind, int rank) const {
  if (!config_ || !config_->multi_process)
    return nullptr;
  if (rank == local_rank) {
    switch (kind) {
      case kIpcInput:
        return const_cast<void*>(getInputBufferForRank(rank));
      case kIpcOutput:
        return getOutputBufferForRank(rank);
      case kIpcInternal:
        return getInternalBufferForRank(rank);
      default:
        return nullptr;
    }
  }
  int comm = commIndex(comm_id);
  if (!shared_state_ || kind < 0 || kind >= 3 || rank < 0 ||
      rank >= static_cast<int>(remote_ipc_[comm][kind].size())) {
    return nullptr;
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(remoteBufferTimeoutMs());
  DeviceIpcBufferRecord record;
  while (!shared_state_->readDeviceIpcBuffer(comm, kind, rank, &record)) {
    if (std::chrono::steady_clock::now() > deadline) {
      LOG_ERROR("ProcessState: timeout waiting for remote CUDA IPC buffer comm=%d kind=%d local_rank=%d remote_rank=%d",
                comm, kind, local_rank, rank);
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  auto& remote = remote_ipc_[comm][kind][rank];
  if (remote.user_ptr && remote.seq == record.seq && remote.base_addr == record.base_addr &&
      remote.offset == record.offset) {
    return remote.user_ptr;
  }
  if (remote.mapped_base && remote.base_addr == record.base_addr) {
    remote.user_ptr = static_cast<char*>(remote.mapped_base) + record.offset;
    remote.offset = record.offset;
    remote.size = record.size;
    remote.seq = record.seq;
    return remote.user_ptr;
  }

  releaseRemoteDeviceBuffer(remote);

#if defined(PCCL_DEVICE_CUDA)
  cudaIpcMemHandle_t handle;
  static_assert(sizeof(handle) <= DEVICE_IPC_HANDLE_BYTES, "cudaIpcMemHandle_t does not fit shared IPC record");
  std::memcpy(&handle, record.handle, sizeof(handle));
  void* mapped_base = nullptr;
  int previous_device = -1;
  cudaGetDevice(&previous_device);
  if (local_device_id >= 0 && previous_device != local_device_id) {
    cudaError_t set_err = cudaSetDevice(local_device_id);
    if (set_err != cudaSuccess) {
      LOG_ERROR("ProcessState: cudaSetDevice failed before cudaIpcOpenMemHandle comm=%d kind=%d local_rank=%d "
                "local_device=%d remote_rank=%d: %s",
                comm, kind, local_rank, local_device_id, rank, cudaGetErrorString(set_err));
      cudaGetLastError();
      return nullptr;
    }
  }
  cudaError_t err = cudaIpcOpenMemHandle(&mapped_base, handle, cudaIpcMemLazyEnablePeerAccess);
  if (previous_device >= 0 && local_device_id >= 0 && previous_device != local_device_id)
    cudaSetDevice(previous_device);
  if (err != cudaSuccess) {
    LOG_ERROR("ProcessState: cudaIpcOpenMemHandle failed comm=%d kind=%d local_rank=%d local_device=%d "
              "remote_rank=%d: %s",
              comm, kind, local_rank, local_device_id, rank, cudaGetErrorString(err));
    cudaGetLastError();
    return nullptr;
  }

  remote.mapped_base = mapped_base;
  remote.user_ptr = static_cast<char*>(mapped_base) + record.offset;
  remote.base_addr = record.base_addr;
  remote.offset = record.offset;
  remote.size = record.size;
  remote.seq = record.seq;
  remote.local_device_id = local_device_id;
  LOG_INFO("ProcessState: Opened remote CUDA IPC buffer comm=%d kind=%d local_rank=%d local_device=%d "
           "remote_rank=%d base=%p ptr=%p",
           comm, kind, local_rank, local_device_id, rank, mapped_base, remote.user_ptr);
  return remote.user_ptr;
#else
  LOG_ERROR("ProcessState: remote device buffer mapping is only implemented for CUDA");
  return nullptr;
#endif
}

void ProcessState::releaseRemoteDeviceBuffer(RemoteDeviceBufferIpc& remote, bool skip_cuda_close) {
#if defined(PCCL_DEVICE_CUDA)
  if (remote.mapped_base) {
    if (skip_cuda_close) {
      static int finalize_close = -1;
      if (finalize_close < 0) {
        const char* env = std::getenv("PCCL_CUDA_IPC_CLOSE_ON_FINALIZE");
        finalize_close = (env && std::atoi(env) != 0) ? 1 : 0;
      }
      if (!finalize_close) {
        remote = RemoteDeviceBufferIpc{};
        return;
      }
    }
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    if (remote.local_device_id >= 0)
      cudaSetDevice(remote.local_device_id);
    cudaError_t err = cudaIpcCloseMemHandle(remote.mapped_base);
    if (err != cudaSuccess) {
      LOG_ERROR("ProcessState: cudaIpcCloseMemHandle failed: %s", cudaGetErrorString(err));
      cudaGetLastError();
    }
    if (previous_device >= 0)
      cudaSetDevice(previous_device);
  }
#endif
  remote = RemoteDeviceBufferIpc{};
}

}  // namespace pccl
