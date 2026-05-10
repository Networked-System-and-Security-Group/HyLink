#ifndef AMPCCL_CONTROLLER_ALGO_FACTORY_H_
#define AMPCCL_CONTROLLER_ALGO_FACTORY_H_

#include "core/domain.h"
#include "algo_base.h"
#include "algo_tcp.h"
#include "algo_dcqcn.h"
#include "common/config.h"
#include "core/domain.h"
#include <memory>

namespace ampccl {

// Forward declaration - will be defined in algo_static.h or here
class StaticAlgo : public AdaptiveAlgo {
public:
    StaticAlgo() : alpha_(0.12976) {}

    double Suggest(const ParamValue& current) override {
        (void)current;
        return alpha_;  // Fixed PCIe ratio
    }

    double GetAlpha() const override { return alpha_; }

    void Update(const ExecStat& stat) override {
        (void)stat;
        //
        // alpha_ = alpha_ + 0.0001;
        // Static algorithm doesn't adapt
    }

    void Reset() override {
        alpha_ =0.12976;
    }

private:
    double alpha_;
};

class AlgoFactory {
public:
    static std::unique_ptr<AdaptiveAlgo> Create(int nranks = 1) {
        AdaptiveAlgorithm algo_type = Config::GetAlgorithm();

        switch (algo_type) {
            case AdaptiveAlgorithm::TCP:
                return std::make_unique<TCPAlgo>(nranks);

            case AdaptiveAlgorithm::DCQCN:
                return std::make_unique<DCQCNAlgo>();

            case AdaptiveAlgorithm::STATIC:
                return std::make_unique<StaticAlgo>();

            default:
                return std::make_unique<TCPAlgo>(nranks);  // Default to TCP
        }
    }
};



}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_FACTORY_H_
