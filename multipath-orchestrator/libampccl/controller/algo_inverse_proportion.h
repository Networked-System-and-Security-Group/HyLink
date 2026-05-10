#ifndef AMPCCL_CONTROLLER_ALGO_INVERSE_PROPORTION_H_
#define AMPCCL_CONTROLLER_ALGO_INVERSE_PROPORTION_H_

#include <cmath>
#include <cstdlib>
#include "algo_base.h"
#include "telemetry/stats.h"
#include "cache/param_cache.h"
#include "common/log.h"

namespace ampccl {

// 反比例算法：按 fast / PCIe 耗时比例调节 alpha（原「TCP」命名易误解为网络 TCP）。
class InverseProportionAlgo : public AdaptiveAlgo {
public:
    explicit InverseProportionAlgo(int nranks = 1) {
        first_flag = 0;
        alpha_ = InitialAlpha(0.01);

        best_alpha_ = 0.12976;
        best_bandwidth_ = 0.0;

        min_alpha_ = 0.001;
        max_alpha_ = 0.39;

        increase_factor_ = 0.001;
        hold_rate_low_ = ReadPositiveRate("AMPCCL_TCP_HOLD_RATE_LOW", 0.95);
        hold_rate_high_ = ReadPositiveRate("AMPCCL_TCP_HOLD_RATE_HIGH", 1.1);
        if (!std::isfinite(hold_rate_low_) || !std::isfinite(hold_rate_high_) ||
            hold_rate_low_ >= hold_rate_high_) {
            hold_rate_low_ = 0.95;
            hold_rate_high_ = 1.1;
        }

        if (nranks <= 1) {
            decrease_factor_ = 0.5;
            return;
        }
        double n = static_cast<double>(nranks);
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
        if (!stat.fast_success || !stat.pcie_success) {
            alpha_ *= decrease_factor_;
            if (alpha_ < min_alpha_) alpha_ = min_alpha_;
            return;
        }

        // 本轮未走 PCIe 路径：无有效 pcie 计时，避免用 0 时间误调 alpha
        if (stat.pcie_bytes == 0) {
            return;
        }

        double fast_time = stat.fast_time;
        double pcie_time = stat.pcie_time;

        first_flag++;
        if (first_flag <= 2) {
            return;
        }

        if (pcie_time > 0 && fast_time > 0) {
            double rate = fast_time / pcie_time;
            if (hold_rate_low_ < rate && rate < hold_rate_high_) {
                return;
            } else if (rate <= hold_rate_low_) {
                alpha_ *= 0.5;
            } else if (rate >= hold_rate_high_) {
                alpha_ *= rate * 0.96;
            }
        }

        if (alpha_ < min_alpha_) alpha_ = min_alpha_;
        if (alpha_ > max_alpha_) alpha_ = max_alpha_;
    }

    void Reset() override {
        alpha_ = InitialAlpha(0.05);
        best_alpha_ = 0.0;
        best_bandwidth_ = 0.0;
    }

private:
    static double ReadPositiveRate(const char* name, double fallback) {
        const char* value = std::getenv(name);
        if (!value || !value[0]) return fallback;
        char* end = nullptr;
        double rate = std::strtod(value, &end);
        // Invalid explicit values invalidate the entire interval, rather than
        // combining a fallback bound with the other user-specified bound.
        if (end == value || *end != '\0' || !std::isfinite(rate) || rate <= 0)
            return NAN;
        return rate;
    }

    static double InitialAlpha(double fallback) {
        const char* value = std::getenv("AMPCCL_TCP_INITIAL_ALPHA");
        if (!value || !value[0]) return fallback;
        char* end = nullptr;
        double alpha = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(alpha) || alpha < 0.001 || alpha > 0.39)
            return fallback;
        return alpha;
    }

    double alpha_;
    double best_alpha_;
    double best_bandwidth_;
    double min_alpha_;
    double max_alpha_;

    double increase_factor_;
    double decrease_factor_;
    double hold_rate_low_;
    double hold_rate_high_;
    int first_flag;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_INVERSE_PROPORTION_H_
