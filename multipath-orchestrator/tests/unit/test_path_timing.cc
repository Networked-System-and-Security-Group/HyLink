#include "telemetry/path_timing.h"

#include <cmath>
#include <cstdio>

namespace {

bool AlmostEqual(double actual, double expected, double tolerance = 1e-9) {
    return std::fabs(actual - expected) <= tolerance;
}

bool CheckOverlappedIntervals() {
    ampccl::PathEventTimestamps events;
    events.fast_start = 1000;
    events.fast_end = 5000;
    events.pcie_start = 3000;
    events.pcie_end = 7000;

    ampccl::PathTimingResult result;
    if (!ampccl::ComputePathTiming(events, 1.0, 1.0, &result)) {
        return false;
    }
    return AlmostEqual(result.start_delta_ms, 0.5) &&
           AlmostEqual(result.end_delta_ms, 0.5) &&
           AlmostEqual(result.overlap_ms, 0.5) &&
           AlmostEqual(result.union_ms, 1.5) &&
           AlmostEqual(result.nonoverlap_ms, 1.0) &&
           AlmostEqual(result.critical_path_overhead_ms, 0.5) &&
           AlmostEqual(result.pcie_overlap_ratio, 0.5) &&
           AlmostEqual(result.fast_overlap_ratio, 0.5) &&
           AlmostEqual(result.parallel_efficiency, 2.0 / 3.0);
}

bool CheckReverseLaunchOrder() {
    ampccl::PathEventTimestamps events;
    events.fast_start = 3000;
    events.fast_end = 7000;
    events.pcie_start = 1000;
    events.pcie_end = 5000;

    ampccl::PathTimingResult result;
    if (!ampccl::ComputePathTiming(events, 1.0, 1.0, &result)) {
        return false;
    }
    return AlmostEqual(result.start_delta_ms, -0.5) &&
           AlmostEqual(result.end_delta_ms, -0.5) &&
           AlmostEqual(result.overlap_ms, 0.5) &&
           AlmostEqual(result.union_ms, 1.5);
}

bool CheckInvalidIntervals() {
    ampccl::PathEventTimestamps events;
    events.fast_start = 1000;
    events.fast_end = 1000;
    events.pcie_start = 2000;
    events.pcie_end = 3000;
    ampccl::PathTimingResult result;
    return !ampccl::ComputePathTiming(events, 0.0, 1.0, &result);
}

}  // namespace

int main() {
    if (!CheckOverlappedIntervals() || !CheckReverseLaunchOrder() || !CheckInvalidIntervals()) {
        std::fprintf(stderr, "path timing test failed\n");
        return 1;
    }
    std::printf("path timing test passed\n");
    return 0;
}
