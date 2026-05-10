#ifndef AMPCCL_CONTROLLER_ALGO_FACTORY_H_
#define AMPCCL_CONTROLLER_ALGO_FACTORY_H_

#include "algo_base.h"
#include "algo_inverse_proportion.h"
#include "algo_split_ratio_control.h"
#include "common/config.h"
#include <memory>

namespace ampccl {

// Forward declaration - will be defined in algo_static.h or here
class StaticAlgo : public AdaptiveAlgo {
public:
    StaticAlgo() : alpha_(Config::GetStaticAlpha()) {}

    double Suggest(const ParamValue& current) override {
        (void)current;
        return alpha_;
    }

    double GetAlpha() const override { return alpha_; }

    void Update(const ExecStat& stat) override {
        (void)stat;
    }

    void Reset() override {
        alpha_ = Config::GetStaticAlpha();
    }

private:
    double alpha_;
};

class AlgoFactory {
public:
    static std::unique_ptr<AdaptiveAlgo> Create(int nranks = 1) {
        AdaptiveAlgorithm algo_type = Config::GetAlgorithm();

        switch (algo_type) {
            case AdaptiveAlgorithm::INVERSE_PROPORTION:
                return std::make_unique<InverseProportionAlgo>(nranks);

            case AdaptiveAlgorithm::SPLIT_RATIO_CONTROL:
                return std::make_unique<SplitRatioControlAlgo>();

            case AdaptiveAlgorithm::STATIC:
                return std::make_unique<StaticAlgo>();

            default:
                return std::make_unique<InverseProportionAlgo>(nranks);
        }
    }
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_FACTORY_H_
