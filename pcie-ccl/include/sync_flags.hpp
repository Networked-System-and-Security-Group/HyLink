#ifndef SYNC_FLAGS_HPP
#define SYNC_FLAGS_HPP

#include <cstdint>

#include "hal/device_rt.hpp"
#include "ir.hpp"

namespace pccl {

// SyncFlagManager: Manages device-side synchronization resources
//
// This class manages device memory resources:
// - H2D Completion Flags: Polled by WaitCompletionKernel for H2D sync
//
// Memory Layout:
// - H2D Completion Flags: uint32_t flags[MAX_CHUNKS] (aligned to 32 bytes)
class SyncFlagManager {
 public:
  SyncFlagManager();
  ~SyncFlagManager();

  // Delete copy/move constructors and assignment operators
  SyncFlagManager(const SyncFlagManager&) = delete;
  SyncFlagManager& operator=(const SyncFlagManager&) = delete;
  SyncFlagManager(SyncFlagManager&&) = delete;
  SyncFlagManager& operator=(SyncFlagManager&&) = delete;

  // Initialize resources (allocate device memory)
  // Returns 0 on success, non-zero on failure
  int initialize();

  // --- H2D Completion Flag Management ---

  // Get device pointer to H2D completion flag for a specific chunk
  // Polled by WaitCompletionKernel on user_stream
  void* getDeviceH2DCompletionFlagPtr(int chunk_idx);

  // Signal H2D complete for a chunk (CPU writes flag=1 via sync H2D)
  // Called by monitor thread when H2D transfer completes
  int signalH2DComplete(int chunk_idx);

  // Reset all H2D completion flags to 0 (call before starting new task)
  int resetCompletionFlags();

 private:
  bool initialized_;

  uint32_t* host_flag_one_;    // Host constant value 1 (for signaling)
  void* host_flag_zeros_;       // Host zero buffer (COMPLETION_FLAGS_SIZE) for bulk reset

  // Device memory for H2D completion flags
  void* dev_h2d_completion_flags_;  // uint32_t[MAX_CHUNKS] on device, polled by WaitCompletionKernel

  // Dedicated stream for signaling operations (avoids deadlock with user stream)
  devStream signal_stream_;

  // Each flag occupies 32 bytes (8 x uint32_t) for alignment
  static constexpr size_t FLAG_STRIDE = 32;
  static constexpr size_t COMPLETION_FLAGS_SIZE = MAX_CHUNKS * FLAG_STRIDE;
};

}  // namespace pccl

#endif  // SYNC_FLAGS_HPP
