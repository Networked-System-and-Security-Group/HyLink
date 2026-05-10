#ifndef AMPCCL_TELEMETRY_PATH_TIMING_H_
#define AMPCCL_TELEMETRY_PATH_TIMING_H_

#include <algorithm>
#include <cstdint>

namespace ampccl {

// Device timestamps recorded by the four events surrounding one mixed
// collective. All four timestamps must come from the same device.
struct PathEventTimestamps {
    uint64_t fast_start = 0;
    uint64_t fast_end = 0;
    uint64_t pcie_start = 0;
    uint64_t pcie_end = 0;
};

struct PathTimingResult {
    double start_delta_ms = 0.0;       // pcie_start - fast_start
    double end_delta_ms = 0.0;         // pcie_end - fast_end
    double overlap_ms = 0.0;
    double union_ms = 0.0;
    double nonoverlap_ms = 0.0;
    double critical_path_overhead_ms = 0.0;  // union - max(path duration)
    double pcie_overlap_ratio = 0.0;         // overlap / PCIe duration
    double fast_overlap_ratio = 0.0;         // overlap / fast duration
    double parallel_efficiency = 0.0;  // max(path duration) / union duration
};

// Convert the timestamp intervals to relative milliseconds. The device event
// elapsed time supplies the scale from timestamp ticks to milliseconds, so no
// device clock frequency or absolute NPU clock is required by the caller.
inline bool ComputePathTiming(const PathEventTimestamps& events,
                              double fast_ms,
                              double pcie_ms,
                              PathTimingResult* result) {
    if (!result || fast_ms <= 0.0 || pcie_ms <= 0.0 ||
        events.fast_end <= events.fast_start || events.pcie_end <= events.pcie_start) {
        return false;
    }

    const uint64_t fast_ticks = events.fast_end - events.fast_start;
    const uint64_t pcie_ticks = events.pcie_end - events.pcie_start;
    const double ticks_per_ms = fast_ticks > 0
                                    ? static_cast<double>(fast_ticks) / fast_ms
                                    : static_cast<double>(pcie_ticks) / pcie_ms;
    if (!(ticks_per_ms > 0.0)) {
        return false;
    }

    const uint64_t interval_start = std::min(events.fast_start, events.pcie_start);
    const uint64_t interval_end = std::max(events.fast_end, events.pcie_end);
    const uint64_t overlap_start = std::max(events.fast_start, events.pcie_start);
    const uint64_t overlap_end = std::min(events.fast_end, events.pcie_end);
    const uint64_t overlap_ticks = overlap_end > overlap_start ? overlap_end - overlap_start : 0;
    const uint64_t union_ticks = interval_end - interval_start;

    const long double start_delta_ticks =
        static_cast<long double>(events.pcie_start) - static_cast<long double>(events.fast_start);
    const long double end_delta_ticks =
        static_cast<long double>(events.pcie_end) - static_cast<long double>(events.fast_end);

    result->start_delta_ms = static_cast<double>(start_delta_ticks / ticks_per_ms);
    result->end_delta_ms = static_cast<double>(end_delta_ticks / ticks_per_ms);
    result->overlap_ms = static_cast<double>(overlap_ticks) / ticks_per_ms;
    result->union_ms = static_cast<double>(union_ticks) / ticks_per_ms;
    result->nonoverlap_ms = std::max(0.0, result->union_ms - result->overlap_ms);
    result->critical_path_overhead_ms = std::max(
        0.0, result->union_ms - std::max(fast_ms, pcie_ms));
    result->pcie_overlap_ratio = std::min(1.0, std::max(0.0, result->overlap_ms / pcie_ms));
    result->fast_overlap_ratio = std::min(1.0, std::max(0.0, result->overlap_ms / fast_ms));
    result->parallel_efficiency = result->union_ms > 0.0
                                      ? std::min(1.0, std::max(0.0,
                                            std::max(fast_ms, pcie_ms) / result->union_ms))
                                      : 0.0;
    return true;
}

}  // namespace ampccl

#endif  // AMPCCL_TELEMETRY_PATH_TIMING_H_
