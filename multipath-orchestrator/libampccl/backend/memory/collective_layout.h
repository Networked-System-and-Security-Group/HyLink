#ifndef AMPCCL_BACKEND_MEMORY_COLLECTIVE_LAYOUT_H_
#define AMPCCL_BACKEND_MEMORY_COLLECTIVE_LAYOUT_H_

#include "common/op_key.h"      // for CollectiveType
#include "core/planner.h"       // for Plan
#include <cstddef>

namespace ampccl {

class CommDomain;

struct SplitCollectiveBuffers {
    const void* fast_send;
    void* fast_recv;

    const void* pcie_send;
    void* pcie_recv;

    void* scratch;

    size_t total_bytes;
    Plan plan;
};

SplitCollectiveBuffers PrepareSplitBuffers(
    CollectiveType op,
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t elem_count,
    int datatype,
    const Plan& plan);

void MergeSplitBuffers(
    const SplitCollectiveBuffers& bufs,
    CollectiveType op,
    CommDomain* domain,
    void* recvbuff,
    size_t elem_count,
    int datatype);

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_MEMORY_COLLECTIVE_LAYOUT_H_

