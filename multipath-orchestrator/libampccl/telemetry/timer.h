#ifndef AMPCCL_TELEMETRY_TIMER_H_
#define AMPCCL_TELEMETRY_TIMER_H_

#include <chrono>

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
    Timer() : use_device_events_(false) {
#if AMPCCL_TIMER_IS_CUDA
        if (cudaEventCreate(&start_event_) == cudaSuccess &&
            cudaEventCreate(&end_event_) == cudaSuccess) {
            use_device_events_ = true;
        }
#elif AMPCCL_TIMER_IS_ACL
        if (aclrtCreateEventWithFlag(&start_event_, ACL_EVENT_TIME_LINE) == ACL_SUCCESS &&
            aclrtCreateEventWithFlag(&end_event_, ACL_EVENT_TIME_LINE) == ACL_SUCCESS) {
            use_device_events_ = true;
        }
#endif
    }

    ~Timer() {
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_) {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(end_event_);
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_) {
            aclrtDestroyEvent(start_event_);
            aclrtDestroyEvent(end_event_);
        }
#endif
    }

    void Start(void* stream) {
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_ && stream) {
            cudaEventRecord(start_event_, static_cast<cudaStream_t>(stream));
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_ && stream) {
            aclrtRecordEvent(start_event_, static_cast<aclrtStream>(stream));
        }
#endif
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void Stop(void* stream) {
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_ && stream) {
            cudaEventRecord(end_event_, static_cast<cudaStream_t>(stream));
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_ && stream) {
            aclrtRecordEvent(end_event_, static_cast<aclrtStream>(stream));
        }
#endif
        
    }

    void Synchronize() {
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_) cudaEventSynchronize(end_event_);
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_) aclrtSynchronizeEvent(end_event_);
#endif
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    double ElapsedSeconds() const {
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_event_, end_event_);
            return ms / 1000.0;
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_) {
            float ms = 0.0f;
            if (aclrtEventElapsedTime(&ms, start_event_, end_event_) == ACL_SUCCESS) {
                return ms / 1000.0;
            }
            return 0.0;
        }
#endif
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time_ - start_time_);
        return duration.count() / 1000000.0;
    }

    double ElapsedMilliseconds() const { return ElapsedSeconds() * 1000.0; }

    double ElapsedCpuSeconds() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time_ - start_time_);
        return duration.count() / 1000000.0;
    }

    double ElapsedCpuMilliseconds() const { return ElapsedCpuSeconds() * 1000.0; }

    void WaitEventOnStream(void* stream) const {
#if AMPCCL_TIMER_IS_CUDA
        if (use_device_events_ && stream) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), end_event_, 0);
        }
#elif AMPCCL_TIMER_IS_ACL
        if (use_device_events_ && stream) {
            aclrtStreamWaitEvent(static_cast<aclrtStream>(stream), end_event_);
        }
#else
        (void)stream;
#endif
    }

private:
    bool use_device_events_;
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
