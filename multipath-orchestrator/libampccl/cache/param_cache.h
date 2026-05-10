#ifndef AMPCCL_CACHE_PARAM_CACHE_H_
#define AMPCCL_CACHE_PARAM_CACHE_H_

#include "common/op_key.h"
#include <unordered_map>
#include <mutex>
#include <vector>
#include <utility>

namespace ampccl {

struct ParamValue {
    double alpha;
    bool use_pcie;        // whether to use PCIe backend
    double fast_bw;       // estimated fast backend bandwidth (GB/s)
    double pcie_bw;       // estimated PCIe backend bandwidth (GB/s)
    double best_alpha;
    double best_bandwidth;

    ParamValue()
        : alpha(0.5), use_pcie(true), fast_bw(0.0), pcie_bw(0.0),
          best_alpha(0.0), best_bandwidth(0.0) {}

    ParamValue(double a, bool use, double fbw, double pbw)
        : alpha(a), use_pcie(use), fast_bw(fbw), pcie_bw(pbw),
          best_alpha(0.0), best_bandwidth(0.0) {}

    ParamValue(double a, bool use, double fbw, double pbw, double best_a, double best_bw)
        : alpha(a), use_pcie(use), fast_bw(fbw), pcie_bw(pbw),
          best_alpha(best_a), best_bandwidth(best_bw) {}
};

class ParamCache {
public:
    bool Contains(const OpKey& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.find(key) != table_.end();
    }

    // Lookup parameters for an operation
    // Returns default if not found
    ParamValue Lookup(const OpKey& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = table_.find(key);
        if (it != table_.end()) {
            return it->second;
        }
        // Return default: 50% split, PCIe enabled
        return ParamValue(0.5, true, 0.0, 0.0);
    }

    // Update parameters for an operation
    void Update(const OpKey& key, const ParamValue& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        table_[key] = value;
    }

    // Clear all cached parameters
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        table_.clear();
    }

    // Get cache size
    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.size();
    }

    // Snapshot for shared-memory sync: copy all entries to vector.
    void GetAll(std::vector<std::pair<OpKey, ParamValue>>* out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        out->clear();
        out->reserve(table_.size());
        for (const auto& p : table_) {
            out->emplace_back(p.first, p.second);
        }
    }

    // Load from snapshot (e.g., read from shared memory). Merges into table_.
    void SetFrom(const std::vector<std::pair<OpKey, ParamValue>>& in) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& p : in) {
            table_[p.first] = p.second;
        }
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<OpKey, ParamValue> table_;
};

}  // namespace ampccl

#endif  // AMPCCL_CACHE_PARAM_CACHE_H_
