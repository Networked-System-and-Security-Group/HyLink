#include "telemetry/time_aggregation.h"

#include <cmath>
#include <cstdio>

namespace {

bool AlmostEqual(double actual, double expected) {
    return std::fabs(actual - expected) < 1e-12;
}

}  // namespace

int main() {
    if (!AlmostEqual(ampccl::MeanRankTime(6.0, 3), 2.0) ||
        !AlmostEqual(ampccl::MeanRankTime(0.0, 0), 0.0) ||
        !AlmostEqual(ampccl::MeanRankTime(5.0, -1), 0.0)) {
        std::fprintf(stderr, "time aggregation test failed\n");
        return 1;
    }
    std::printf("time aggregation test passed\n");
    return 0;
}
