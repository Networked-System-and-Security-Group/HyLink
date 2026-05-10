
#include "original_collective_api.h"
#include <cstddef>

namespace ampccl {

using HcclAllReduceFn = int (*)(const void*, void*, unsigned long, int, int, void*, void*);
using HcclAllGatherFn = int (*)(const void*, void*, unsigned long, int, void*, void*);
using HcclReduceScatterFn = int (*)(const void*, void*, unsigned long, int, int, void*, void*);
using HcclBroadcastFn = int (*)(const void*, void*, unsigned long, int, unsigned int, void*, void*);

using NcclAllReduceFn = int (*)(const void*, void*, size_t, int, int, void*, void*);
using NcclAllGatherFn = int (*)(const void*, void*, size_t, int, void*, void*);
using NcclReduceScatterFn = int (*)(const void*, void*, size_t, int, int, void*, void*);
using NcclBroadcastFn = int (*)(const void*, void*, size_t, int, int, void*, void*);

static HcclAllReduceFn g_hccl_allreduce = nullptr;
static HcclAllGatherFn g_hccl_allgather = nullptr;
static HcclReduceScatterFn g_hccl_reducescatter = nullptr;
static HcclBroadcastFn g_hccl_broadcast = nullptr;

static NcclAllReduceFn g_nccl_allreduce = nullptr;
static NcclAllGatherFn g_nccl_allgather = nullptr;
static NcclReduceScatterFn g_nccl_reducescatter = nullptr;
static NcclBroadcastFn g_nccl_broadcast = nullptr;

void OriginalCollectiveAPI::RegisterHccl(
    void* allreduce, void* allgather, void* reducescatter, void* broadcast) {
    g_hccl_allreduce = reinterpret_cast<HcclAllReduceFn>(allreduce);
    g_hccl_allgather = reinterpret_cast<HcclAllGatherFn>(allgather);
    g_hccl_reducescatter = reinterpret_cast<HcclReduceScatterFn>(reducescatter);
    g_hccl_broadcast = reinterpret_cast<HcclBroadcastFn>(broadcast);
}

void OriginalCollectiveAPI::RegisterNccl(
    void* allreduce, void* allgather, void* reducescatter, void* broadcast) {
    g_nccl_allreduce = reinterpret_cast<NcclAllReduceFn>(allreduce);
    g_nccl_allgather = reinterpret_cast<NcclAllGatherFn>(allgather);
    g_nccl_reducescatter = reinterpret_cast<NcclReduceScatterFn>(reducescatter);
    g_nccl_broadcast = reinterpret_cast<NcclBroadcastFn>(broadcast);
}

static BackendResult IntResultToBackend(int r) {
    return (r == 0) ? BackendResult::Success : BackendResult::UnhandledError;
}

BackendResult OriginalCollectiveAPI::AllReduce(
    const void* sendbuff, void* recvbuff, size_t count,
    int datatype, int op, void* comm, void* stream) {
    if (g_hccl_allreduce) {
        int r = g_hccl_allreduce(sendbuff, recvbuff, static_cast<unsigned long>(count),
                                 datatype, op, comm, stream);
        return IntResultToBackend(r);
    }
    if (g_nccl_allreduce) {
        int r = g_nccl_allreduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        return IntResultToBackend(r);
    }
    return BackendResult::UnhandledError;
}

BackendResult OriginalCollectiveAPI::AllGather(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    int datatype, void* comm, void* stream) {
    if (g_hccl_allgather) {
        int r = g_hccl_allgather(sendbuff, recvbuff, static_cast<unsigned long>(sendcount),
                                 datatype, comm, stream);
        return IntResultToBackend(r);
    }
    if (g_nccl_allgather) {
        int r = g_nccl_allgather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        return IntResultToBackend(r);
    }
    return BackendResult::UnhandledError;
}

BackendResult OriginalCollectiveAPI::ReduceScatter(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    int datatype, int op, void* comm, void* stream) {
    if (g_hccl_reducescatter) {
        int r = g_hccl_reducescatter(sendbuff, recvbuff, static_cast<unsigned long>(recvcount),
                                     datatype, op, comm, stream);
        return IntResultToBackend(r);
    }
    if (g_nccl_reducescatter) {
        int r = g_nccl_reducescatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
        return IntResultToBackend(r);
    }
    return BackendResult::UnhandledError;
}

BackendResult OriginalCollectiveAPI::Broadcast(
    const void* sendbuff, void* recvbuff, size_t count,
    int datatype, int root, void* comm, void* stream) {
    if (g_hccl_broadcast) {
        int r = g_hccl_broadcast(sendbuff, recvbuff, static_cast<unsigned long>(count),
                                 datatype, static_cast<unsigned int>(root), comm, stream);
        return IntResultToBackend(r);
    }
    if (g_nccl_broadcast) {
        int r = g_nccl_broadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
        return IntResultToBackend(r);
    }
    return BackendResult::UnhandledError;
}

}  // namespace ampccl
