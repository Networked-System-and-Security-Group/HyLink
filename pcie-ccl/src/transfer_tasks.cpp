#include "include/transfer_tasks.hpp"

#include "include/log.hpp"

namespace pccl {

EventRingBuffer::EventRingBuffer(size_t capacity)
    : capacity_(capacity), head_(0), published_(0), tail_(0), initialized_(false), events_(nullptr) {}

EventRingBuffer::~EventRingBuffer() {
  cleanup();
}

int EventRingBuffer::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    return 0;  // Already initialized
  }

  events_ = new devEvent[capacity_];

  for (size_t i = 0; i < capacity_; ++i) {
    devStatus status = devCreateEvent(&events_[i]);
    if (status != 0) {
      LOG_ERROR("EventRingBuffer: Failed to create event %zu, status=%d", i, status);
      // Cleanup already created events
      for (size_t j = 0; j < i; ++j) {
        devDestroyEvent(events_[j]);
      }
      delete[] events_;
      events_ = nullptr;
      return -1;
    }
  }

  head_ = 0;
  published_ = 0;
  tail_ = 0;
  initialized_ = true;
  LOG_DEBUG("EventRingBuffer: Initialized with capacity %zu", capacity_);
  return 0;
}

void EventRingBuffer::cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    return;
  }

  for (size_t i = 0; i < capacity_; ++i) {
    devDestroyEvent(events_[i]);
  }

  delete[] events_;
  events_ = nullptr;
  initialized_ = false;
  head_ = 0;
  published_ = 0;
  tail_ = 0;
  LOG_DEBUG("EventRingBuffer: Cleaned up");
}

//--- Producer interface ---

devEvent EventRingBuffer::acquire() {
  std::unique_lock<std::mutex> lock(mutex_);

  // Wait until there's space in the ring (head - tail < capacity)
  not_full_cv_.wait(lock, [this] { return (head_ - tail_) < capacity_; });

  size_t slot = head_ % capacity_;
  ++head_;
  return events_[slot];
}

void EventRingBuffer::publish() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++published_;
}

//--- Consumer interface ---

bool EventRingBuffer::hasPublished() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return published_ > tail_;
}

devEvent EventRingBuffer::peekOldest() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return events_[tail_ % capacity_];
}

void EventRingBuffer::release() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++tail_;
  not_full_cv_.notify_one();  // Wake up producer if it was waiting
}

size_t EventRingBuffer::publishedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return published_ - tail_;
}

//--- Legacy interface ---

void EventRingBuffer::waitAll() {
  // Wait for all published events to complete
  while (true) {
    devEvent ev;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (tail_ >= published_) {
        break;  // All published events have been consumed
      }
      ev = events_[tail_ % capacity_];
    }

    // Poll until this event completes (outside the lock)
    devEventStatus status;
    while (true) {
      devStatus result = devQueryEvent(ev, &status);
      if (result != 0) {
        LOG_ERROR("EventRingBuffer::waitAll: devQueryEvent failed, status=%d", result);
        break;
      }
      if (status == devEventStatusComplete) {
        break;
      }
    }

    // Release this slot
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++tail_;
      not_full_cv_.notify_one();
    }
  }
}

}  // namespace pccl
