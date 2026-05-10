#ifndef AMPCCL_CORE_PLANNER_H_
#define AMPCCL_CORE_PLANNER_H_

#include "common/config.h"
#include <cstddef>

namespace ampccl {

struct Plan {
    size_t fast_bytes;   // Bytes to send via fast backend
    size_t pcie_bytes;   // Bytes to send via PCIe backend
    bool use_pcie;       // Whether to use PCIe backend

    Plan() : fast_bytes(0), pcie_bytes(0), use_pcie(false) {}
};

class Planner {
public:
    static Plan CreatePlan(size_t total_bytes, double alpha, bool use_pcie_hint) {
        Plan plan;

        // Check if PCIe should be disabled
        size_t min_msg_size = Config::GetMinMsgSize();
        size_t min_chunk_size = Config::GetMinChunkSize();
        // AMPCCL_LOG(INFO,"CreatePlan, total_bytes=%zu, min_msg_size=%zu, min_chunk_size=%zu, use_pcie_hint=%d",total_bytes, min_msg_size, min_chunk_size, use_pcie_hint);

        if (!Config::IsPCIeEnabled() ||
            total_bytes < min_msg_size ||
            !use_pcie_hint) {
            plan.fast_bytes = total_bytes;
            plan.pcie_bytes = 0;
            plan.use_pcie = false;
            return plan;
        }

        // Clamp alpha to valid range (alpha = PCIe ratio)
        // if(0 < alpha && alpha < 0.005) alpha = 0;

        if (alpha < 0.0) alpha = 0.0;
        if (alpha > 1.0) alpha = 1.0;

        plan.pcie_bytes = static_cast<size_t>(total_bytes * alpha);
        plan.fast_bytes = total_bytes - plan.pcie_bytes;


        // Enforce minimum chunk sizes
        if (plan.fast_bytes > 0 && plan.fast_bytes < min_chunk_size) {
            plan.pcie_bytes += plan.fast_bytes;
            plan.fast_bytes = 0;
        }
        if (plan.pcie_bytes > 0 && plan.pcie_bytes < min_chunk_size) {
            plan.fast_bytes += plan.pcie_bytes;
            plan.pcie_bytes = 0;
        }

        // If either chunk is too small, use only one backend
        if (plan.pcie_bytes < min_chunk_size) {
            plan.fast_bytes = total_bytes;
            plan.pcie_bytes = 0;
            plan.use_pcie = false;
        } else if (plan.fast_bytes < min_chunk_size) {
            plan.fast_bytes = 0;
            plan.pcie_bytes = total_bytes;
            plan.use_pcie = true;
        } else {
            plan.use_pcie = true;
        }

          // Align fast_bytes and pcie_bytes for best performance (NCCL: 16B, HCCL: 64B/128B)
         constexpr size_t kAlign16 = 127;
         plan.fast_bytes = (plan.fast_bytes + kAlign16) & ~kAlign16;
         plan.pcie_bytes = (plan.pcie_bytes + kAlign16) & ~kAlign16;

        // Ensure total doesn't exceed original
        if (plan.fast_bytes + plan.pcie_bytes > total_bytes) {
            plan.pcie_bytes = total_bytes - plan.fast_bytes;
        }

        return plan;
    }
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_PLANNER_H_
