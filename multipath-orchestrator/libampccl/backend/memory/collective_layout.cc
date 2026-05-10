#include "backend/memory/collective_layout.h"
#include "core/domain.h"

namespace ampccl {

SplitCollectiveBuffers PrepareSplitBuffers(
    CollectiveType op,
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t elem_count,
    int datatype,
    const Plan& plan) {
    (void)op;
    (void)domain;
    (void)datatype;

    SplitCollectiveBuffers bufs;
    bufs.total_bytes = elem_count;
    bufs.plan = plan;
    bufs.scratch = nullptr;

    const char* send_base = static_cast<const char*>(sendbuff);
    char* recv_base = static_cast<char*>(recvbuff);

    bufs.fast_send = send_base;
    bufs.fast_recv = recv_base;

    if (plan.use_pcie && plan.pcie_bytes > 0) {
        bufs.pcie_send = send_base + plan.fast_bytes;
        bufs.pcie_recv = recv_base + plan.fast_bytes;
    } else {
        bufs.pcie_send = nullptr;
        bufs.pcie_recv = nullptr;
    }

    return bufs;
}

void MergeSplitBuffers(
    const SplitCollectiveBuffers& bufs,
    CollectiveType op,
    CommDomain* domain,
    void* recvbuff,
    size_t elem_count,
    int datatype) {
    (void)bufs;
    (void)op;
    (void)domain;
    (void)recvbuff;
    (void)elem_count;
    (void)datatype;
}

}  // namespace ampccl

