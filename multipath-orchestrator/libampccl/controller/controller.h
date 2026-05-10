#ifndef AMPCCL_CONTROLLER_CONTROLLER_H_
#define AMPCCL_CONTROLLER_CONTROLLER_H_

#include "algo_base.h"
#include "algo_factory.h"
#include "cache/param_cache.h"
#include "telemetry/stats.h"
#include "common/op_key.h"
#include "common/config.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ampccl {

// Adaptive controller that manages algorithm and parameter cache
class AdaptiveController {
public:
    explicit AdaptiveController(int nranks) : nranks_(nranks) {}

    double SuggestAlpha(const OpKey& op_key, const ParamCache& cache) {
        std::lock_guard<std::mutex> lock(mutex_);
        AdaptiveAlgo& algo = GetOrCreateAlgoLocked(op_key);
        if (!cache.Contains(op_key)) {
            return algo.GetAlpha();
        }
        ParamValue current = cache.Lookup(op_key);
        return algo.Suggest(current);
    }

    // Update controller state based on execution statistics
    void Update(const OpKey& op_key, const ExecStat& stat, ParamCache& cache) {
        std::lock_guard<std::mutex> lock(mutex_);
        AdaptiveAlgo& algo = GetOrCreateAlgoLocked(op_key);
        ParamValue prev = cache.Lookup(op_key);
        algo.Update(stat);
        double new_alpha = algo.GetAlpha();

        double fast_bw = stat.GetFastBandwidth();
        double pcie_bw = stat.GetPCIeBandwidth();

        bool use_pcie = prev.use_pcie;
        if (Config::IsPCIeEnabled()) {
            if (stat.pcie_bytes > 0) {
                // 本轮实际跑了 PCIe：由成功与否与带宽决定
                use_pcie = stat.pcie_success && pcie_bw > 0.0;
            }
            // stat.pcie_bytes == 0：Planner 跳过或纯 Fast，不将 use_pcie 强行置 false
        } else {
            use_pcie = false;
        }

        ParamValue updated(new_alpha, use_pcie, fast_bw, pcie_bw,
                          algo.GetBestAlpha(), algo.GetBestBandwidth());

        cache.Update(op_key, updated);
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        algorithms_.clear();
    }

private:
    AdaptiveAlgo& GetOrCreateAlgoLocked(const OpKey& op_key) {
        auto it = algorithms_.find(op_key);
        if (it == algorithms_.end()) {
            auto inserted = algorithms_.emplace(op_key, AlgoFactory::Create(nranks_));
            return *inserted.first->second;
        }
        return *it->second;
    }

    int nranks_;
    std::unordered_map<OpKey, std::unique_ptr<AdaptiveAlgo>> algorithms_;
    mutable std::mutex mutex_;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_CONTROLLER_H_
