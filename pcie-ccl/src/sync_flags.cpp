#include "include/sync_flags.hpp"

#include <cstring>

#include "include/log.hpp"

namespace pccl {

SyncFlagManager::SyncFlagManager()
    : initialized_(false), host_flag_one_(nullptr), host_flag_zeros_(nullptr), dev_h2d_completion_flags_(nullptr) {
  signal_stream_.handle = nullptr;
}

SyncFlagManager::~SyncFlagManager() {
  if (!initialized_) {
    return;
  }

  LOG_INFO("SyncFlagManager: Releasing resources");

  if (host_flag_one_) {
    devFreeHost(host_flag_one_);
    host_flag_one_ = nullptr;
  }

  if (host_flag_zeros_) {
    devFreeHost(host_flag_zeros_);
    host_flag_zeros_ = nullptr;
  }

  if (dev_h2d_completion_flags_) {
    devFree(dev_h2d_completion_flags_);
    dev_h2d_completion_flags_ = nullptr;
  }

  if (signal_stream_.handle) {
    devDestroyStream(signal_stream_);
    signal_stream_.handle = nullptr;
  }

  initialized_ = false;
  LOG_INFO("SyncFlagManager: Resources released");
}

int SyncFlagManager::initialize() {
  if (initialized_) {
    LOG_WARN("SyncFlagManager: Already initialized");
    return 0;
  }

  LOG_INFO("SyncFlagManager: Initializing (COMPLETION_FLAGS_SIZE=%zu)", COMPLETION_FLAGS_SIZE);

  devStatus status;

  // Allocate host flag source (value 1 for signaling)
  status = devMallocHost(reinterpret_cast<void**>(&host_flag_one_), sizeof(uint32_t));
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to allocate host flag one, status=%d", status);
    return -1;
  }
  *host_flag_one_ = 1;

  // Allocate host zero buffer (reused for every resetCompletionFlags call)
  status = devMallocHost(&host_flag_zeros_, COMPLETION_FLAGS_SIZE);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to allocate host flag zeros, status=%d", status);
    devFreeHost(host_flag_one_);
    host_flag_one_ = nullptr;
    return -1;
  }
  memset(host_flag_zeros_, 0, COMPLETION_FLAGS_SIZE);

  // Create dedicated signal stream (avoids deadlock with user stream)
  status = devCreateStream(&signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to create signal stream, status=%d", status);
    devFreeHost(host_flag_one_);
    devFreeHost(host_flag_zeros_);
    host_flag_one_ = nullptr;
    host_flag_zeros_ = nullptr;
    return -1;
  }

  // Allocate H2D completion flag buffer
  status = devMalloc(&dev_h2d_completion_flags_, COMPLETION_FLAGS_SIZE);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to allocate H2D completion flags, status=%d", status);
    devDestroyStream(signal_stream_);
    devFreeHost(host_flag_one_);
    devFreeHost(host_flag_zeros_);
    signal_stream_.handle = nullptr;
    host_flag_one_ = nullptr;
    host_flag_zeros_ = nullptr;
    return -1;
  }

  // Initialize completion flags to 0
  status =
      devMemcpyAsync(dev_h2d_completion_flags_, host_flag_zeros_, COMPLETION_FLAGS_SIZE, MemcpyDirection::H2D, signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to initialize H2D completion flags (async), status=%d", status);
    devFree(dev_h2d_completion_flags_);
    devDestroyStream(signal_stream_);
    devFreeHost(host_flag_one_);
    devFreeHost(host_flag_zeros_);
    dev_h2d_completion_flags_ = nullptr;
    signal_stream_.handle = nullptr;
    host_flag_one_ = nullptr;
    host_flag_zeros_ = nullptr;
    return -1;
  }

  status = devSynchronizeStream(signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to sync signal stream for completion flags init, status=%d", status);
    devFree(dev_h2d_completion_flags_);
    devDestroyStream(signal_stream_);
    devFreeHost(host_flag_one_);
    devFreeHost(host_flag_zeros_);
    dev_h2d_completion_flags_ = nullptr;
    signal_stream_.handle = nullptr;
    host_flag_one_ = nullptr;
    host_flag_zeros_ = nullptr;
    return -1;
  }

  initialized_ = true;
  LOG_INFO("SyncFlagManager: Initialization complete");
  return 0;
}

void* SyncFlagManager::getDeviceH2DCompletionFlagPtr(int chunk_idx) {
  if (!initialized_ || chunk_idx < 0 || chunk_idx >= MAX_CHUNKS) {
    LOG_ERROR("SyncFlagManager: Invalid getDeviceH2DCompletionFlagPtr call (init=%d, chunk=%d)", initialized_,
              chunk_idx);
    return nullptr;
  }
  return static_cast<char*>(dev_h2d_completion_flags_) + (chunk_idx * FLAG_STRIDE);
}

int SyncFlagManager::signalH2DComplete(int chunk_idx) {
  if (!initialized_ || chunk_idx < 0 || chunk_idx >= MAX_CHUNKS) {
    LOG_ERROR("SyncFlagManager: Invalid signalH2DComplete call (init=%d, chunk=%d)", initialized_, chunk_idx);
    return -1;
  }

  void* dst = static_cast<char*>(dev_h2d_completion_flags_) + (chunk_idx * FLAG_STRIDE);
  devStatus status = devMemcpyAsync(dst, host_flag_one_, sizeof(uint32_t), MemcpyDirection::H2D, signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to signal H2D complete for chunk %d (async), status=%d", chunk_idx, status);
    return -1;
  }
  status = devSynchronizeStream(signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to sync signal stream for H2D chunk %d, status=%d", chunk_idx, status);
    return -1;
  }

  LOG_DEBUG("SyncFlagManager: Signaled H2D complete for chunk %d", chunk_idx);
  return 0;
}

int SyncFlagManager::resetCompletionFlags() {
  if (!initialized_) {
    LOG_ERROR("SyncFlagManager: Cannot reset completion flags - not initialized");
    return -1;
  }

  devStatus status =
      devMemcpyAsync(dev_h2d_completion_flags_, host_flag_zeros_, COMPLETION_FLAGS_SIZE, MemcpyDirection::H2D, signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to reset H2D completion flags (async), status=%d", status);
    return -1;
  }

  status = devSynchronizeStream(signal_stream_);
  if (status != devSuccess) {
    LOG_ERROR("SyncFlagManager: Failed to sync signal stream for reset completion flags, status=%d", status);
    return -1;
  }

  LOG_DEBUG("SyncFlagManager: All completion flags reset to 0");
  return 0;
}

}  // namespace pccl
