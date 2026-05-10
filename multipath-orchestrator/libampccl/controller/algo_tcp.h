#ifndef AMPCCL_CONTROLLER_ALGO_TCP_H_
#define AMPCCL_CONTROLLER_ALGO_TCP_H_

#include <cmath>
#include "algo_base.h"
#include "telemetry/stats.h"
#include "cache/param_cache.h"
#include "common/log.h"

namespace ampccl {

// TCP-style hybrid link tuning algorithm.
// Design: (1) Optimal alpha when pcie_time = fast_time; (2) Safety margin for hardware variance;
// (3) PCIe straggler: multiplicative decrease when pcie > fast; (4) Convergence toward target;
// (5) Track best_alpha when combined bandwidth improves.
class TCPAlgo : public AdaptiveAlgo {
public:
    explicit TCPAlgo(int nranks = 1) {
        first_flag = 0;
        alpha_ = 0.37;

        best_alpha_ = 0.12976;
        best_bandwidth_ = 0.0;

        min_alpha_ = 0.001;
        max_alpha_ = 0.39;//static best
        
        
        increase_factor_ = 0.001;  // Additive increase step when no best yet
        

        if (nranks <= 1) {
            decrease_factor_ = 0.5;
            return;
        }
        double n = static_cast<double>(nranks);
        double scale = std::sqrt(n);
        decrease_factor_ = 0.5 * std::sqrt(2.0 / n);
        if (decrease_factor_ < 0.2) decrease_factor_ = 0.2;
        if (decrease_factor_ > 1.0) decrease_factor_ = 1.0;
    }

    double Suggest(const ParamValue& current) override {
        alpha_ = current.alpha;
        best_alpha_ = current.best_alpha;
        best_bandwidth_ = current.best_bandwidth;
        if (alpha_ < min_alpha_) alpha_ = min_alpha_;
        if (alpha_ > max_alpha_) alpha_ = max_alpha_;
        return alpha_;
    }

    double GetAlpha() const override { return alpha_; }

    double GetBestAlpha() const override { return best_alpha_; }
    double GetBestBandwidth() const override { return best_bandwidth_; }

    void Update(const ExecStat& stat) override {
        // AMPCCL_LOG(OFF, "before,flag:%d, best_alpha:%.5f, best_bw:%.4fGB/s, current_alpha:%.5f", first_flag, best_alpha_, best_bandwidth_, alpha_);
        if (!stat.fast_success || !stat.pcie_success) {
            alpha_ *= decrease_factor_;
            if (alpha_ < min_alpha_) alpha_ = min_alpha_;
            return;
        }

        double fast_time = stat.fast_time;
        double pcie_time = stat.pcie_time;

        first_flag++;
        if(first_flag <= 2){  // Skip first invalid update and first unstable result
            return;
        }





        // if(pcie_time > 0 && fast_time > 0){
        //     double rate = fast_time / pcie_time;
        //     if(0.95 < rate && rate < 1.05){
        //         // Near balance, keep stable
        //         return;
        //     }else if( rate <=0.95){
        //         alpha_ *= 0.5;
        //     }else if( rate >= 1.05){
        //         alpha_ *= rate * 0.97;
        //     }
        // }

        alpha_ += increase_factor_;

        if (alpha_ < min_alpha_) alpha_ = min_alpha_;
        if (alpha_ > max_alpha_) alpha_ = max_alpha_;
        // AMPCCL_LOG(OFF, "after,flag:%d, best_alpha:%.5f, best_bw:%.4fGB/s, current_alpha:%.5f", first_flag, best_alpha_, best_bandwidth_, alpha_);
    }

    void Reset() override {
        alpha_ = 0.05;
        best_alpha_ = 0.0;
        best_bandwidth_ = 0.0;
    }

private:
    double alpha_;
    double best_alpha_;
    double best_bandwidth_;
    double min_alpha_;
    double max_alpha_;
   
    double increase_factor_;  // Additive increase when no best
    double decrease_factor_;  // Multiplicative decrease on PCIe straggler
    int first_flag;  // Skip first unstable iter
    
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_TCP_H_
