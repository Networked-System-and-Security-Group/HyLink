#include "controller/controller.h"
#include <cassert>
#include <cmath>
#include <cstdlib>

int main() {
    setenv("AMPCCL_ALGO", "tcp", 1);
    setenv("AMPCCL_TCP_INITIAL_ALPHA", "0.075", 1);
    ampccl::AdaptiveController controller(2);
    ampccl::ParamCache cache;
    ampccl::OpKey key{ampccl::CollectiveType::AllGather, 536870912, 536870912, 1, 2, 123};
    assert(std::abs(controller.SuggestAlpha(key, cache) - 0.075) < 1e-12);
    controller.Reset();
    assert(std::abs(controller.SuggestAlpha(key, cache) - 0.075) < 1e-12);
    ampccl::InverseProportionAlgo algo(2);
    algo.Reset();
    assert(std::abs(algo.GetAlpha() - 0.075) < 1e-12);
    unsetenv("AMPCCL_TCP_INITIAL_ALPHA");
    ampccl::InverseProportionAlgo defaults(2);
    assert(std::abs(defaults.GetAlpha() - 0.01) < 1e-12);
    defaults.Reset();
    assert(std::abs(defaults.GetAlpha() - 0.05) < 1e-12);
    for (const char* invalid : {"nan", "bad", "0.075junk", "0", "0.9"}) {
        setenv("AMPCCL_TCP_INITIAL_ALPHA", invalid, 1);
        ampccl::InverseProportionAlgo fallback(2);
        assert(std::abs(fallback.GetAlpha() - 0.01) < 1e-12);
    }
}
