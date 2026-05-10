#ifndef AMPCCL_CONTROLLER_ALGO_BASE_H_
#define AMPCCL_CONTROLLER_ALGO_BASE_H_

#include "cache/param_cache.h"
#include "telemetry/stats.h"

namespace ampccl {

// Base class for adaptive algorithms
class AdaptiveAlgo {
public:
    virtual ~AdaptiveAlgo() = default;

    // Suggest alpha based on current parameters
    // Returns alpha in [0.0, 1.0]
    virtual double Suggest(const ParamValue& current) = 0;

    virtual double GetAlpha() const = 0;
    virtual double GetBestAlpha() const { return 0.0; }
    virtual double GetBestBandwidth() const { return 0.0; }

    // Update algorithm state based on execution statistics
    virtual void Update(const ExecStat& stat) = 0;

    // Reset algorithm state
    virtual void Reset() = 0;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_BASE_H_
