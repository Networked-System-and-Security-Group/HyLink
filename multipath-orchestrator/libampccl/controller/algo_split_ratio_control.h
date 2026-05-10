#ifndef AMPCCL_CONTROLLER_ALGO_SPLIT_RATIO_CONTROL_H_
#define AMPCCL_CONTROLLER_ALGO_SPLIT_RATIO_CONTROL_H_

#include <algorithm>
#include <cstdlib>
#include "algo_base.h"
#include "telemetry/stats.h"
#include "cache/param_cache.h"
#include "common/config.h"

namespace ampccl {

// 文档 adaptive_split_ratio_control.md：x 为 PCIe 流量占比，与 Planner 中 alpha（pcie_bytes/total）一致。
class SplitRatioControlAlgo : public AdaptiveAlgo {
public:
    SplitRatioControlAlgo()
        : alpha_(GetDefaultInitialAlpha()),
          bp_hat_(0.0),
          best_alpha_(0.0),
          best_bandwidth_(0.0),
          update_count_(0) {}

    double Suggest(const ParamValue& current) override {
        if (current.alpha > 0.0) {
            alpha_ = current.alpha;
        }
        best_alpha_ = current.best_alpha;
        best_bandwidth_ = current.best_bandwidth;
        const double min_a = GetMinAlpha();
        alpha_ = std::max(min_a, std::min(max_alpha_, alpha_));
        return alpha_;
    }

    double GetAlpha() const override { return alpha_; }

    double GetBestAlpha() const override { return best_alpha_; }
    double GetBestBandwidth() const override { return best_bandwidth_; }

    void Update(const ExecStat& stat) override {
        const double min_a = GetMinAlpha();
        ++update_count_;

        // 与反比例一致：任一路失败则减小 PCIe 占比，避免错误放大
        if (!stat.fast_success || !stat.pcie_success) {
            alpha_ *= 0.5;
            if (alpha_ < min_a) {
                alpha_ = min_a;
            }
            return;
        }

        // 本轮未走 PCIe：无有效 PCIe 计时，不更新 alpha（避免被「纯 fast」样本拉偏）
        if (stat.pcie_bytes == 0) {
            return;
        }

        const double Bp = stat.GetPCIeBandwidth();
        const double Bf = stat.GetFastBandwidth();
        if (Bp <= 0.0 || Bf <= 0.0) {
            return;
        }

        const double smooth = Config::GetSplitRatioSmoothing();
        const double eta = Config::GetSplitRatioEta();
        const size_t warmup_steps = GetWarmupSteps();
        constexpr size_t kMinBandwidthSampleBytes = 4 * 1024 * 1024;   // 4MB
        constexpr size_t kMinPenaltyBytes = 16 * 1024 * 1024;           // 16MB

        // 仅在分流载荷达到一定大小时更新真实 PCIe 带宽 EMA，避免极小包的时延底线污染带宽估计
        if (stat.pcie_bytes >= kMinBandwidthSampleBytes || bp_hat_ <= 0.0) {
            if (bp_hat_ <= 0.0) {
                bp_hat_ = Bp;
            } else {
                bp_hat_ = smooth * Bp + (1.0 - smooth) * bp_hat_;
            }
        }

        const double bf_hat = Bf;
        const double x_star = bp_hat_ / (bp_hat_ + bf_hat);

        const double T_pcie = alpha_ / bp_hat_;
        const double T_fast = (1.0 - alpha_) / bf_hat;

        // Warmup 保护期内不执行激进惩罚，允许系统稳定探索
        if (update_count_ <= warmup_steps) {
            alpha_ += eta * (x_star - alpha_);
        } else if (T_pcie > T_fast * 1.05) {
            // 仅对具有代表性的大包应用瓶颈回退，小包避免进入下限死锁
            if (stat.pcie_bytes >= kMinPenaltyBytes) {
                alpha_ *= 0.9;
            } else {
                alpha_ += eta * (x_star - alpha_);
            }
        } else {
            alpha_ += eta * (x_star - alpha_);
        }

        alpha_ = std::max(min_a, std::min(max_alpha_, alpha_));
    }

    void Reset() override {
        alpha_ = GetDefaultInitialAlpha();
        bp_hat_ = 0.0;
        best_alpha_ = 0.0;
        best_bandwidth_ = 0.0;
        update_count_ = 0;
    }

private:
    static double GetDefaultInitialAlpha() {
        const char* env = std::getenv("AMPCCL_SPLIT_RATIO_INITIAL_ALPHA");
        if (env && env[0] != '\0') return std::atof(env);
        const char* static_a = std::getenv("AMPCCL_STATIC_ALPHA");
        if (static_a && static_a[0] != '\0') return std::atof(static_a);
        return 0.20;
    }

    static double GetMinAlpha() {
        const char* env = std::getenv("AMPCCL_SPLIT_RATIO_MIN_ALPHA");
        if (env && env[0] != '\0') return std::atof(env);
        return 0.05;
    }

    static size_t GetWarmupSteps() {
        const char* env = std::getenv("AMPCCL_SPLIT_RATIO_WARMUP_STEPS");
        if (env && env[0] != '\0') return static_cast<size_t>(std::strtoul(env, nullptr, 10));
        return 5;
    }

    double alpha_;
    double bp_hat_;
    double best_alpha_;
    double best_bandwidth_;
    size_t update_count_;

    static constexpr double max_alpha_ = 0.999;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_SPLIT_RATIO_CONTROL_H_
