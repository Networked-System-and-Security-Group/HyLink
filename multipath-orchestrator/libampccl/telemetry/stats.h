#ifndef AMPCCL_TELEMETRY_STATS_H_
#define AMPCCL_TELEMETRY_STATS_H_

#include <cstddef>

namespace ampccl {

struct ExecStat {
    double fast_time;      // Time taken by fast backend (seconds)
    double pcie_time;       // Time taken by PCIe backend (seconds)
    size_t fast_bytes;      // Bytes sent via fast backend
    size_t pcie_bytes;      // Bytes sent via PCIe backend
    bool fast_success;      // Whether fast backend succeeded
    bool pcie_success;      // Whether PCIe backend succeeded

    ExecStat()
        : fast_time(0.0), pcie_time(0.0),
          fast_bytes(0), pcie_bytes(0),
          fast_success(true), pcie_success(true) {}

    double GetFastBandwidth() const {
        if (fast_time > 0.0 && fast_bytes > 0) {
            return fast_bytes / fast_time / (1024.0 * 1024.0 * 1024.0);  // GB/s
        }
        return 0.0;
    }

    double GetPCIeBandwidth() const {
        if (pcie_time > 0.0 && pcie_bytes > 0) {
            return pcie_bytes / pcie_time / (1024.0 * 1024.0 * 1024.0);  // GB/s
        }
        return 0.0;
    }

    double GetTotalTime() const {
        return fast_time > pcie_time ? fast_time : pcie_time;  // max of both
    }

    double GetCombinedBandwidth() const {
        double total_time = GetTotalTime();
        if (total_time > 0.0 && (fast_bytes + pcie_bytes) > 0) {
            return (fast_bytes + pcie_bytes) / total_time / (1024.0 * 1024.0 * 1024.0);  // GB/s
        }
        return 0.0;
    }
};

}  // namespace ampccl

#endif  // AMPCCL_TELEMETRY_STATS_H_
