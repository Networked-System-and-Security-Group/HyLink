#ifndef AMPCCL_TELEMETRY_TIMER_H_
#define AMPCCL_TELEMETRY_TIMER_H_

#include <chrono>
#include <thread>
#include "common/log.h"

#define AMPCCL_TIMER_IS_CUDA defined(AMPCCL_USE_CUDA_TIMER)
#define AMPCCL_TIMER_IS_ACL  defined(AMPCCL_USE_ACL_TIMER)

#if AMPCCL_TIMER_IS_CUDA
#include <cuda_runtime.h>
#elif AMPCCL_TIMER_IS_ACL
#include <acl/acl_rt.h>
#ifndef ACL_EVENT_TIME_LINE
#define ACL_EVENT_TIME_LINE 0x00000008u
#endif
#endif

namespace ampccl {

#if AMPCCL_TIMER_IS_CUDA
static constexpr int kTimerBackend = 1;
#elif AMPCCL_TIMER_IS_ACL
static constexpr int kTimerBackend = 2;
#else
static constexpr int kTimerBackend = 0;
#endif

class Timer {
public:
    Timer()
        : use_device_events_(false),
          start_event_created_(false),
          end_event_created_(false),
          start_event_recorded_(false),
          end_event_recorded_(false) {
#if AMPCCL_TIMER_IS_CUDA
        cudaError_t start_ret = cudaEventCreate(&start_event_);
        cudaError_t end_ret = cudaEventCreate(&end_event_);
        start_event_created_ = (start_ret == cudaSuccess);
        end_event_created_ = (end_ret == cudaSuccess);
        use_device_events_ = start_event_created_ && end_event_created_;
        if (!use_device_events_) {
            AMPCCL_LOG(WARN, "Timer: CUDA event create failed start_ret=%d end_ret=%d",
                       static_cast<int>(start_ret), static_cast<int>(end_ret));
        }
#elif AMPCCL_TIMER_IS_ACL
        aclError start_ret = aclrtCreateEventWithFlag(&start_event_, ACL_EVENT_TIME_LINE);
        aclError end_ret = aclrtCreateEventWithFlag(&end_event_, ACL_EVENT_TIME_LINE);
        start_event_created_ = (start_ret == ACL_SUCCESS);
        end_event_created_ = (end_ret == ACL_SUCCESS);
        use_device_events_ = start_event_created_ && end_event_created_;
        if (!use_device_events_) {
            AMPCCL_LOG(WARN, "Timer: ACL event create failed start_ret=%d end_ret=%d",
                       static_cast<int>(start_ret), static_cast<int>(end_ret));
        }
#endif
    }

    ~Timer() {
#if AMPCCL_TIMER_IS_CUDA
        if (start_event_created_) {
            cudaEventDestroy(start_event_);
        }
        if (end_event_created_) {
            cudaEventDestroy(end_event_);
        }
#elif AMPCCL_TIMER_IS_ACL
        if (start_event_created_) {
            aclrtDestroyEvent(start_event_);
        }
        if (end_event_created_) {
            aclrtDestroyEvent(end_event_);
        }
#endif
    }

    void Start(void* stream) {
        start_time_ = std::chrono::high_resolution_clock::now();
        end_time_ = start_time_;
        start_event_recorded_ = false;
        end_event_recorded_ = false;
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_ && stream) {
            cudaError_t ret = cudaEventRecord(start_event_, static_cast<cudaStream_t>(stream));
            start_event_recorded_ = (ret == cudaSuccess);
            if (!start_event_recorded_) {
                AMPCCL_LOG(WARN, "Timer: CUDA start event record failed ret=%d stream=%p",
                           static_cast<int>(ret), stream);
            }
        } else if (use_device_events_) {
            AMPCCL_LOG(WARN, "Timer: CUDA start event not recorded because stream is null");
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_ && stream) {
            aclError ret = aclrtRecordEvent(start_event_, static_cast<aclrtStream>(stream));
            start_event_recorded_ = (ret == ACL_SUCCESS);
            if (!start_event_recorded_) {
                AMPCCL_LOG(WARN, "Timer: ACL start event record failed ret=%d stream=%p",
                           static_cast<int>(ret), stream);
            }
        } else if (use_device_events_) {
            AMPCCL_LOG(WARN, "Timer: ACL start event not recorded because stream is null");
        }
#endif
    }

    void Stop(void* stream) {
        end_time_ = std::chrono::high_resolution_clock::now();
        end_event_recorded_ = false;
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_ && stream) {
            cudaError_t ret = cudaEventRecord(end_event_, static_cast<cudaStream_t>(stream));
            end_event_recorded_ = (ret == cudaSuccess);
            if (!end_event_recorded_) {
                AMPCCL_LOG(WARN, "Timer: CUDA end event record failed ret=%d stream=%p",
                           static_cast<int>(ret), stream);
            }
        } else if (use_device_events_) {
            AMPCCL_LOG(WARN, "Timer: CUDA end event not recorded because stream is null");
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_ && stream) {
            aclError ret = aclrtRecordEvent(end_event_, static_cast<aclrtStream>(stream));
            end_event_recorded_ = (ret == ACL_SUCCESS);
            if (!end_event_recorded_) {
                AMPCCL_LOG(WARN, "Timer: ACL end event record failed ret=%d stream=%p",
                           static_cast<int>(ret), stream);
            }
        } else if (use_device_events_) {
            AMPCCL_LOG(WARN, "Timer: ACL end event not recorded because stream is null");
        }
#endif
    }

    void Synchronize() {
#if AMPCCL_TIMER_IS_CUDA
        if (CanUseRecordedEvents()) {
            cudaError_t ret = cudaEventSynchronize(end_event_);
            if (ret != cudaSuccess) {
                AMPCCL_LOG(WARN, "Timer: CUDA end event synchronize failed ret=%d", static_cast<int>(ret));
            }
        }
#elif AMPCCL_TIMER_IS_ACL
        if (CanUseRecordedEvents()) {
            aclError ret = aclrtSynchronizeEvent(end_event_);
            if (ret != ACL_SUCCESS) {
                AMPCCL_LOG(WARN, "Timer: ACL end event synchronize failed ret=%d", static_cast<int>(ret));
            }
        }
#endif
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    bool QueryComplete() const {
#if AMPCCL_TIMER_IS_CUDA
        if (CanUseRecordedEvents()) {
            cudaError_t st = cudaEventQuery(end_event_);
            return st == cudaSuccess;
        }
#elif AMPCCL_TIMER_IS_ACL
        if (CanUseRecordedEvents()) {
            aclrtEventRecordedStatus status;
            aclError st = aclrtQueryEventStatus(end_event_, &status);
            return st == ACL_SUCCESS && status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
        }
#endif
        return true;
    }

    bool WaitUntilCompleteByQuery(uint64_t max_spins = 0) {
        // Pure yield()-spin holds this thread's core at ~100% for the entire
        // wait, competing with pcieccl's H2H/executor threads and every other
        // rank's callback thread for the same cores. Most collectives finish
        // within a handful of spins (device work already in flight when the
        // callback fires), so a short pure-spin window keeps the fast case
        // latency-free; falling back to short sleeps afterward stops an
        // unusually slow completion from monopolizing a core while this
        // thread waits.
        constexpr uint64_t kPureSpinIterations = 1000;
        uint64_t spins = 0;
        while (!QueryComplete()) {
            // Count unconditionally so the sleep fallback below also applies
            // to unbounded waits (max_spins == 0, the default at call sites).
            ++spins;
            if (max_spins != 0 && spins >= max_spins) {
                return false;
            }
            if (spins < kPureSpinIterations) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }
        end_time_ = std::chrono::high_resolution_clock::now();
        return true;
    }

    void MarkCompletedOnHost() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    double ElapsedSeconds() const {
        double seconds = 0.0;
        return ElapsedSecondsChecked(&seconds) ? seconds : 0.0;
    }

    bool ElapsedSecondsChecked(double* seconds, bool* used_device_timer = nullptr) const {
        if (!seconds) {
            return false;
        }
        if (used_device_timer) {
            *used_device_timer = false;
        }
#if AMPCCL_TIMER_IS_CUDA
        if (CanUseRecordedEvents()) {
            float ms = 0.0f;
            cudaError_t st = cudaEventElapsedTime(&ms, start_event_, end_event_);
            if (st == cudaSuccess && ms > 0.0f) {
                *seconds = ms / 1000.0;
                if (used_device_timer) {
                    *used_device_timer = true;
                }
                return true;
            }
            AMPCCL_LOG(WARN, "Timer: CUDA elapsed time invalid ret=%d ms=%.6f; fallback to CPU timer",
                       static_cast<int>(st), static_cast<double>(ms));
        }
#elif AMPCCL_TIMER_IS_ACL
        if (CanUseRecordedEvents()) {
            float ms = 0.0f;
            aclError st = aclrtEventElapsedTime(&ms, start_event_, end_event_);
            if (st == ACL_SUCCESS && ms > 0.0f) {
                *seconds = ms / 1000.0;
                if (used_device_timer) {
                    *used_device_timer = true;
                }
                return true;
            }
            AMPCCL_LOG(WARN, "Timer: ACL elapsed time invalid ret=%d ms=%.6f; fallback to CPU timer",
                       static_cast<int>(st), static_cast<double>(ms));
        }
#endif
        *seconds = ElapsedCpuSeconds();
        return *seconds > 0.0;
    }

    double ElapsedMilliseconds() const { return ElapsedSeconds() * 1000.0; }

    double ElapsedCpuSeconds() const {
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }

    double ElapsedCpuMilliseconds() const { return ElapsedCpuSeconds() * 1000.0; }

    void WaitEventOnStream(void* stream, int rank) const {
#if AMPCCL_TIMER_IS_CUDA
        if (CanUseRecordedEvents() && stream) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), end_event_, 0);
        }
#elif AMPCCL_TIMER_IS_ACL
        if (CanUseRecordedEvents() && stream) {
            aclrtStreamWaitEvent(static_cast<aclrtStream>(stream), end_event_);
        }
#else
        (void)stream;
        (void)rank;
#endif
    }

    void* GetEndEvent() const {
        return CanUseRecordedEvents() ? reinterpret_cast<void*>(end_event_) : nullptr;
    }

private:
    bool CanUseRecordedEvents() const {
        return use_device_events_ && start_event_recorded_ && end_event_recorded_;
    }

    bool use_device_events_;
    bool start_event_created_;
    bool end_event_created_;
    bool start_event_recorded_;
    bool end_event_recorded_;
#if AMPCCL_TIMER_IS_CUDA
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
#elif AMPCCL_TIMER_IS_ACL
    aclrtEvent start_event_;
    aclrtEvent end_event_;
#endif
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

}  // namespace ampccl

#endif  // AMPCCL_TELEMETRY_TIMER_H_
