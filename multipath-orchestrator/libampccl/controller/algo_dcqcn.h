#ifndef AMPCCL_CONTROLLER_ALGO_DCQCN_H_
#define AMPCCL_CONTROLLER_ALGO_DCQCN_H_

#include "algo_base.h"
#include "telemetry/stats.h"
#include "cache/param_cache.h"
#include <deque>
#include <algorithm>

namespace ampccl {

// DCQCN-style algorithm (Data Center Quantized Congestion Notification)
// Uses feedback from bandwidth measurements to adjust alpha
class DCQCNAlgo : public AdaptiveAlgo {
public:
    DCQCNAlgo()
        : alpha_(0.5),
          target_ratio_(1.0),  // Target ratio of PCIe to fast bandwidth
          kp_(0.1),            // Proportional gain
          ki_(0.01),           // Integral gain
          kd_(0.001),          // Derivative gain
          integral_error_(0.0),
          last_error_(0.0),
          window_size_(10) {}

    double Suggest(const ParamValue& current) override {
        alpha_ = current.alpha;
        return alpha_;
    }

    double GetAlpha() const override { return alpha_; }

    void Update(const ExecStat& stat) override {
        if (!stat.fast_success || !stat.pcie_success) {
            // If either backend failed, decrease alpha (less PCIe)
            alpha_ *= 0.8;
            if (alpha_ < 0.1) alpha_ = 0.1;
            return;
        }

        double fast_bw = stat.GetFastBandwidth();
        double pcie_bw = stat.GetPCIeBandwidth();

        if (fast_bw <= 0.0 || pcie_bw <= 0.0) {
            return;  // Invalid measurements
        }

        // Calculate current ratio
        double current_ratio = pcie_bw / fast_bw;

        // Calculate error (difference from target)
        double error = target_ratio_ - current_ratio;

        // PID controller
        integral_error_ += error;
        // Clamp integral to prevent windup
        if (integral_error_ > 1.0) integral_error_ = 1.0;
        if (integral_error_ < -1.0) integral_error_ = -1.0;

        double derivative_error = error - last_error_;
        last_error_ = error;

        // PID output
        double pid_output = kp_ * error + ki_ * integral_error_ + kd_ * derivative_error;

        // Update alpha
        alpha_ += pid_output;

        // Clamp alpha to valid range
        if (alpha_ < 0.1) alpha_ = 0.1;
        if (alpha_ > 0.9) alpha_ = 0.9;
    }

    void Reset() override {
        alpha_ = 0.5;
        integral_error_ = 0.0;
        last_error_ = 0.0;
    }

private:
    double alpha_;
    double target_ratio_;
    double kp_, ki_, kd_;
    double integral_error_;
    double last_error_;
    size_t window_size_;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_DCQCN_H_
