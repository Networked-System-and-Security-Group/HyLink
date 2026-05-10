#ifndef PCCL_EVENT_RING_HPP
#define PCCL_EVENT_RING_HPP

#include <unistd.h>

#include <cstddef>

#include "../runtime/ascend_runtime.hpp"
#include "../util/log.hpp"

namespace pccl {

// Lock-free event ring for SINGLE-THREADED use (D2H executor).
// All produce/consume happens on the same thread — no mutex needed.
class EventRingBuffer {
 public:
  static constexpr size_t DEFAULT_CAPACITY = 64;

  explicit EventRingBuffer(size_t capacity = DEFAULT_CAPACITY)
      : capacity_(capacity), head_(0), tail_(0), initialized_(false), events_(nullptr) {}

  ~EventRingBuffer() { cleanup(); }

  EventRingBuffer(const EventRingBuffer&) = delete;
  EventRingBuffer& operator=(const EventRingBuffer&) = delete;

  int initialize() {
    if (initialized_)
      return 0;
    events_ = new devEvent[capacity_];
    for (size_t i = 0; i < capacity_; ++i) {
      devStatus status = devCreateEvent(&events_[i]);
      if (status != devSuccess) {
        LOG_ERROR("EventRingBuffer: Failed to create event %zu", i);
        for (size_t j = 0; j < i; ++j)
          devDestroyEvent(events_[j]);
        delete[] events_;
        events_ = nullptr;
        return -1;
      }
    }
    head_ = tail_ = 0;
    initialized_ = true;
    return 0;
  }

  void cleanup() {
    if (!initialized_)
      return;
    for (size_t i = 0; i < capacity_; ++i)
      devDestroyEvent(events_[i]);
    delete[] events_;
    events_ = nullptr;
    initialized_ = false;
  }

  // How many events are in-flight
  size_t inflight() const { return head_ - tail_; }

  bool isFull() const { return inflight() >= capacity_; }

  // Get next event slot and advance head (caller must ensure !isFull())
  devEvent nextSlot() { return events_[head_++ % capacity_]; }

  // Index of the slot that was just consumed by nextSlot() (head-1)
  // Use (head_ + capacity_ - 1) to avoid unsigned underflow when head_ == 0.
  size_t lastHeadIndex() const { return (head_ + capacity_ - 1) % capacity_; }

  // Index of the oldest in-flight slot (will be released next)
  size_t tailIndex() const { return tail_ % capacity_; }

  // Peek at the oldest in-flight event (caller must ensure inflight() > 0)
  devEvent peekOldest() const { return events_[tail_ % capacity_]; }

  // Peek at the newest in-flight event (caller must ensure inflight() > 0)
  devEvent peekNewest() const { return events_[(head_ + capacity_ - 1) % capacity_]; }

  // Release the oldest slot
  void releaseOldest() { ++tail_; }

  // Reset counters (keep events allocated)
  void reset() { head_ = tail_ = 0; }

 private:
  size_t capacity_;
  size_t head_;
  size_t tail_;
  bool initialized_;
  devEvent* events_;
};

}  // namespace pccl

#endif /* PCCL_EVENT_RING_HPP */
