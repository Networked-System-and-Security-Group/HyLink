#ifndef AMPCCL_CORE_TIMER_QUERY_POOL_H_
#define AMPCCL_CORE_TIMER_QUERY_POOL_H_

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ampccl {

// Runs device timer queries off the stream-callback thread (runtime APIs are
// unsafe there) without paying a thread create/join per collective. Grows on
// demand so concurrent ranks never queue behind each other's spinning query.
class TimerQueryPool {
public:
    static TimerQueryPool& GetInstance() {
        static TimerQueryPool instance;
        return instance;
    }

    void RunAndWait(std::function<void()> task) {
        auto slot = std::make_shared<Slot>();
        slot->task = std::move(task);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(slot);
            if (idle_ < queue_.size()) {
                workers_.emplace_back(&TimerQueryPool::WorkerLoop, this);
            }
        }
        cv_.notify_one();

        std::unique_lock<std::mutex> lock(slot->mutex);
        slot->cv.wait(lock, [&] { return slot->done; });
    }

private:
    struct Slot {
        std::function<void()> task;
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
    };

    TimerQueryPool() = default;

    ~TimerQueryPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    TimerQueryPool(const TimerQueryPool&) = delete;
    TimerQueryPool& operator=(const TimerQueryPool&) = delete;

    void WorkerLoop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            ++idle_;
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            --idle_;
            if (queue_.empty()) {
                return;
            }
            std::shared_ptr<Slot> slot = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();

            slot->task();
            {
                std::lock_guard<std::mutex> slot_lock(slot->mutex);
                slot->done = true;
            }
            slot->cv.notify_one();

            lock.lock();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Slot>> queue_;
    std::vector<std::thread> workers_;
    size_t idle_ = 0;
    bool stop_ = false;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_TIMER_QUERY_POOL_H_
