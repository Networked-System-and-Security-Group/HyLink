#ifndef TRANSFER_TASKS_HPP
#define TRANSFER_TASKS_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "hal/device_rt.hpp"
#include "ir.hpp"

namespace pccl {

//==============================================================================
// D2H Task Structures
//==============================================================================

// D2HSubChunk: A single sub-chunk transfer within a D2H task
struct D2HSubChunk {
  void* dev_src;   // Device source address
  void* host_dst;  // Host destination address
  size_t size;     // Transfer size in bytes
  int slice_idx;   // Slice index for progress tracking (0-based)
};

// D2HTask: A complete D2H transfer task with sub-chunk pipelining
struct D2HTask {
  int src_chunk_idx;     // Source chunk index in device buffer
  int dst_chunk_idx;     // Destination chunk index in host buffer
  int effect_chunk_idx;  // Chunk index for version update (from OutputEffect)
  int numa_node;         // NUMA node for the destination

  void* dev_base;     // Device buffer base address
  void* host_base;    // Host buffer base address
  size_t chunk_size;  // Total chunk size in bytes

  std::vector<D2HSubChunk> sub_chunks;  // Pre-computed sub-chunk transfers
};

//==============================================================================
// H2D Task Structures
//==============================================================================

// H2DSubChunk: A single sub-chunk transfer within an H2D task
struct H2DSubChunk {
  void* host_src;  // Host source address
  void* dev_dst;   // Device destination address
  size_t size;     // Transfer size in bytes
  int slice_idx;   // Slice index for progress tracking (0-based)
};

// H2DTask: A complete H2D transfer task with dependency tracking
struct H2DTask {
  int src_chunk_idx;              // Source chunk index in host buffer
  int dst_chunk_idx;              // Destination chunk index in device buffer
  int completion_flag_chunk_idx;  // Chunk index for completion flag signaling

  void* host_base;    // Host buffer base address
  void* dev_base;     // Device buffer base address
  size_t chunk_size;  // Total chunk size in bytes

  std::vector<H2DSubChunk> sub_chunks;  // Pre-computed sub-chunk transfers
  std::vector<DataDependency> deps;     // Version dependencies to wait for

  // Barrier-only flag: when true, this task has no actual H2D work,
  // but participates in the global completion barrier (for ranks without H2D)
  bool is_barrier_only = false;
};

//==============================================================================
// EventRingBuffer: Thread-safe producer-consumer circular buffer for events
//==============================================================================

// EventRingBuffer: Thread-safe circular buffer for tracking async operations.
// Producer thread: acquire() -> recordEvent() -> publish()
// Consumer thread: tryConsume() -> queryEvent() -> release()
class EventRingBuffer {
 public:
  explicit EventRingBuffer(size_t capacity = 8);
  ~EventRingBuffer();

  // Delete copy/move
  EventRingBuffer(const EventRingBuffer&) = delete;
  EventRingBuffer& operator=(const EventRingBuffer&) = delete;
  EventRingBuffer(EventRingBuffer&&) = delete;
  EventRingBuffer& operator=(EventRingBuffer&&) = delete;

  // Initialize (creates events). Returns 0 on success.
  int initialize();

  // Cleanup (destroys events)
  void cleanup();

  bool isInitialized() const { return initialized_; }
  size_t capacity() const { return capacity_; }

  //--- Producer interface (D2H submit thread) ---

  // Acquire an event slot for recording. Blocks if ring is full.
  // Returns the event to record to.
  devEvent acquire();

  // Mark the last acquired slot as published (record complete).
  // Must be called after recordEvent().
  void publish();

  //--- Consumer interface (Progress update thread) ---

  // Check if there are published but not yet released events.
  bool hasPublished() const;

  // Get the oldest published event (for querying).
  // Only valid if hasPublished() returns true.
  devEvent peekOldest() const;

  // Release the oldest slot after event is confirmed complete.
  // Wakes up producer if it was blocked.
  void release();

  // Get count of published but unreleased events.
  size_t publishedCount() const;

  //--- Legacy interface (for H2D which doesn't need producer-consumer) ---

  // Wait for all outstanding events to complete (blocking poll)
  void waitAll();

 private:
  size_t capacity_;
  size_t head_;       // Next slot to acquire (producer writes)
  size_t published_;  // Number of published slots (producer writes after record)
  size_t tail_;       // Number of released slots (consumer writes)
  bool initialized_;
  devEvent* events_;

  mutable std::mutex mutex_;
  std::condition_variable not_full_cv_;  // Producer waits when full
};

}  // namespace pccl

#endif  // TRANSFER_TASKS_HPP
