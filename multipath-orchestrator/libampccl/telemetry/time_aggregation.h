#ifndef AMPCCL_TELEMETRY_TIME_AGGREGATION_H_
#define AMPCCL_TELEMETRY_TIME_AGGREGATION_H_

namespace ampccl {

inline double MeanRankTime(double sum_seconds, int nranks) {
    if (nranks <= 0) {
        return 0.0;
    }
    return sum_seconds / static_cast<double>(nranks);
}

}  // namespace ampccl

#endif  // AMPCCL_TELEMETRY_TIME_AGGREGATION_H_
