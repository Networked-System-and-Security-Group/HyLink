#include "process_state.hpp"

#include <algorithm>
#include <mutex>

#include "../executor/h2h_engine.hpp"
#include "../memory/host_pool.hpp"
#include "../memory/shared_state.hpp"
#include "../runtime/ascend_runtime.hpp"
#include "../util/log.hpp"
#include "config.hpp"

namespace pccl {

ProcessState::ProcessState() : initialized_(false), config_(nullptr) {}

ProcessState::~ProcessState() {
  finalize();
}

ProcessState& ProcessState::instance() {
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

  comm_buffers_.clear();

  if (config.multi_process) {
    bool is_creator = (config.local_rank == 0);
    shared_state_ = std::make_unique<SharedStateManager>(config.nranks, config.unique_id, is_creator);
    if (!shared_state_ || !shared_state_->isValid()) {
      LOG_ERROR("ProcessState: SharedStateManager initialization failed");
      shared_state_.reset();
      config_ = nullptr;
      return -1;
    }
    host_pool_ = std::make_unique<HostMemoryPool>(config.nranks, config.host_buffer_size, &config.topology,
                                                  config.unique_id, is_creator);
    if (!host_pool_ || !host_pool_->isValid()) {
      LOG_ERROR("ProcessState: HostMemoryPool initialization failed");
      host_pool_.reset();
      shared_state_.reset();
      config_ = nullptr;
      return -1;
    }
    if (!shared_state_->barrier("ProcessState::initialize", config.local_rank, -1))
      return -1;
    LOG_INFO("ProcessState: Multi-process shared memory ready (is_creator=%d)", is_creator ? 1 : 0);
  } else {
    shared_state_ = std::make_unique<SharedStateManager>(config.nranks);
    host_pool_ = std::make_unique<HostMemoryPool>(config.nranks, config.host_buffer_size, &config.topology);
    if (!host_pool_ || !host_pool_->isValid()) {
      LOG_ERROR("ProcessState: HostMemoryPool initialization failed");
      host_pool_.reset();
      shared_state_.reset();
      config_ = nullptr;
      return -1;
    }
  }

  if (!config.bootstrap_addr.empty()) {
    LOG_ERROR("ProcessState: multi-node bootstrap is unsupported in this Ascend single-node build");
    h2h_engine_.reset();
    shared_state_.reset();
    host_pool_.reset();
    config_ = nullptr;
    return -1;
  }

  h2h_engine_ = std::make_unique<H2HEngine>(shared_state_.get(), &config.topology);
  if (!h2h_engine_ || h2h_engine_->initialize() != 0) {
    LOG_ERROR("ProcessState: H2HEngine initialization failed");
    h2h_engine_.reset();
    host_pool_.reset();
    shared_state_.reset();
    config_ = nullptr;
    return -1;
  }

  initialized_ = true;
  LOG_INFO("ProcessState: Initialized (nranks=%d, multi_process=%d)",
           config.nranks, config.multi_process ? 1 : 0);
  return 0;
}

void ProcessState::finalize() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (!initialized_)
    return;

  if (h2h_engine_)
    h2h_engine_->shutdown();
  h2h_engine_.reset();

  {
    std::lock_guard<std::mutex> buffers_lock(buffer_mutex_);
    for (auto& entry : comm_buffers_) {
      if (entry.second.internal) {
        devSetDevice(entry.first.device_id);
        devFree(entry.second.internal);
        entry.second.internal = nullptr;
        entry.second.internal_size = 0;
      }
    }
    comm_buffers_.clear();
  }

  host_pool_.reset();
  shared_state_.reset();
  config_ = nullptr;
  initialized_ = false;
  LOG_INFO("ProcessState: Finalized");
}

bool ProcessState::isMultiProcess() const {
  return config_ && config_->multi_process;
}

bool ProcessState::registerDeviceBuffers(uint64_t comm_hash, int comm_id, int rank, int device_id,
                                         const void* input, void* output) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const CommMemoryKey key{comm_hash, comm_id, rank, device_id};
  auto& buffers = comm_buffers_[key];
  buffers.input = input;
  buffers.output = output;
  return true;
}

const void* ProcessState::getInputBufferForComm(uint64_t comm_hash, int comm_id, int rank, int device_id) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const auto it = comm_buffers_.find(CommMemoryKey{comm_hash, comm_id, rank, device_id});
  return it == comm_buffers_.end() ? nullptr : it->second.input;
}

void* ProcessState::getOutputBufferForComm(uint64_t comm_hash, int comm_id, int rank, int device_id) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const auto it = comm_buffers_.find(CommMemoryKey{comm_hash, comm_id, rank, device_id});
  return it == comm_buffers_.end() ? nullptr : it->second.output;
}

int ProcessState::initializeInternalBuffer(uint64_t comm_hash, int comm_id, int rank, int device_id) {
  return ensureInternalBuffer(comm_hash, comm_id, rank, device_id, PCCL_INTERNAL_BUFFER_BYTES);
}

int ProcessState::ensureInternalBuffer(uint64_t comm_hash, int comm_id, int rank, int device_id, size_t bytes) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const CommMemoryKey key{comm_hash, comm_id, rank, device_id};
  auto& buffers = comm_buffers_[key];
  const size_t allocation = InternalBufferAllocationBytes(bytes);
  if (allocation == 0) {
    LOG_ERROR("ProcessState: internal buffer request exceeds budget key=(hash=%llu comm=%d rank=%d device=%d) "
              "requested=%zu budget=%zu",
              static_cast<unsigned long long>(comm_hash), comm_id, rank, device_id, bytes,
              PCCL_INTERNAL_BUFFER_BYTES);
    return -1;
  }
  if (buffers.internal && buffers.internal_size >= allocation)
    return 0;
  if (buffers.internal) {
    devSetDevice(device_id);
    void* replacement = nullptr;
    if (devMalloc(&replacement, allocation) != devSuccess || !replacement) {
      LOG_ERROR("ProcessState: failed to grow internal device buffer key=(hash=%llu comm=%d rank=%d device=%d) "
                "requested=%zu allocated=%zu",
                static_cast<unsigned long long>(comm_hash), comm_id, rank, device_id, allocation,
                buffers.internal_size);
      return -1;
    }
    devFree(buffers.internal);
    buffers.internal = replacement;
    buffers.internal_size = allocation;
    LOG_INFO("ProcessState: Reallocated internal device buffer key=(hash=%llu comm=%d rank=%d device=%d) "
             "requested=%zu allocated=%zu ptr=%p",
             static_cast<unsigned long long>(comm_hash), comm_id, rank, device_id, bytes, allocation, replacement);
    return 0;
  }
  devSetDevice(device_id);
  void* ptr = nullptr;
  if (devMalloc(&ptr, allocation) != devSuccess || !ptr) {
    LOG_ERROR("ProcessState: Failed to allocate internal device buffer key=(hash=%llu comm=%d rank=%d device=%d) "
              "bytes=%zu",
              static_cast<unsigned long long>(comm_hash), comm_id, rank, device_id, allocation);
    return -1;
  }
  buffers.internal = ptr;
  buffers.internal_size = allocation;
  LOG_INFO("ProcessState: Allocated internal device buffer key=(hash=%llu comm=%d rank=%d device=%d) "
           "requested=%zu allocated=%zu ptr=%p",
           static_cast<unsigned long long>(comm_hash), comm_id, rank, device_id, bytes, allocation, ptr);
  return 0;
}

size_t ProcessState::getInternalBufferSize(uint64_t comm_hash, int comm_id, int rank, int device_id) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const auto it = comm_buffers_.find(CommMemoryKey{comm_hash, comm_id, rank, device_id});
  return it == comm_buffers_.end() ? 0 : it->second.internal_size;
}

void* ProcessState::getInternalBufferForComm(uint64_t comm_hash, int comm_id, int rank, int device_id) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const auto it = comm_buffers_.find(CommMemoryKey{comm_hash, comm_id, rank, device_id});
  return it == comm_buffers_.end() ? nullptr : it->second.internal;
}

}  // namespace pccl
