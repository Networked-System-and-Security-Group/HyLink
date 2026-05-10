
#include "original_collective_api.h"
#include "nccl_p2p_v_collectives.h"
#include "common/config.h"
#include "common/log.h"
#include <cstddef>
#include <cstdint>

namespace ampccl {

using HcclAllReduceFn = int (*)(const void*, void*, unsigned long, int, int, void*, void*);
using HcclAllGatherFn = int (*)(const void*, void*, unsigned long, int, void*, void*);
// Matches hccl.h HcclReduceScatter (void* send, void* recv, recvCount, ...).
using HcclReduceScatterFn = int (*)(void*, void*, unsigned long, int, int, void*, void*);
// Matches hccl.h HcclAlltoAll (8-arg canonical: send/recv counts per peer; sendType/recvType).
using HcclAlltoAllFn = int (*)(const void*, uint64_t, int, const void*, uint64_t, int, void*, void*);
// Matches legacy HcclAllToAll (6-arg symmetric: single count and dtype for both send and recv).
using HcclAlltoAll_6ArgFn = int (*)(const void*, void*, uint64_t, int, void*, void*);

using HcclAllGatherVFn = int (*)(void*, uint64_t, void*, const void*, const void*, int, void*, void*);
using HcclReduceScatterVFn =
    int (*)(void*, const void*, const void*, void*, uint64_t, int, int, void*, void*);
using HcclAlltoAllVFn = int (*)(const void*, const void*, const void*, int, const void*, const void*,
                               const void*, int, void*, void*);

using NcclAllReduceFn = int (*)(const void*, void*, size_t, int, int, void*, void*);
using NcclAllGatherFn = int (*)(const void*, void*, size_t, int, void*, void*);
using NcclReduceScatterFn = int (*)(const void*, void*, size_t, int, int, void*, void*);
using NcclBroadcastFn = int (*)(const void*, void*, size_t, int, int, void*, void*);
using NcclAllToAllFn = int (*)(const void*, void*, size_t, int, void*, void*);

// Generic stream synchronize function pointer type
using StreamSynchronizeFn = int (*)(void* stream);
using EventSynchronizeFn = int (*)(void* event);

static StreamSynchronizeFn g_stream_synchronize = nullptr;
static EventSynchronizeFn g_event_synchronize = nullptr;

static HcclAllReduceFn g_hccl_allreduce = nullptr;
static HcclAllGatherFn g_hccl_allgather = nullptr;
static HcclReduceScatterFn g_hccl_reducescatter = nullptr;
static HcclAlltoAllFn g_hccl_alltoall = nullptr;
static HcclAlltoAll_6ArgFn g_hccl_alltoall_6arg = nullptr;
static HcclAllGatherVFn g_hccl_allgather_v = nullptr;
// ncclAllGatherv(sendbuff, sendcount, recvbuff, recvcounts[], displs[], dtype, comm, stream)
typedef int (*NcclAllGatherVFn)(const void*, size_t, void*, const size_t*, const size_t*, int, void*, void*);
static NcclAllGatherVFn g_nccl_allgather_v = nullptr;
static HcclReduceScatterVFn g_hccl_reducescatter_v = nullptr;
static HcclAlltoAllVFn g_hccl_alltoall_v = nullptr;

static NcclAllReduceFn g_nccl_allreduce = nullptr;
static NcclAllGatherFn g_nccl_allgather = nullptr;
static NcclReduceScatterFn g_nccl_reducescatter = nullptr;
static NcclBroadcastFn g_nccl_broadcast = nullptr;
static NcclAllToAllFn g_nccl_alltoall = nullptr;

void OriginalCollectiveAPI::RegisterSynchronize(void* sync_func) {
    g_stream_synchronize = reinterpret_cast<StreamSynchronizeFn>(sync_func);
}

void OriginalCollectiveAPI::RegisterSynchronizeEvent(void* sync_event_func) {
    g_event_synchronize = reinterpret_cast<EventSynchronizeFn>(sync_event_func);
}

void OriginalCollectiveAPI::RegisterHccl(void* allreduce, void* allgather, void* reducescatter,
                                         void* alltoall_8arg, void* alltoall_6arg,
                                         void* allgather_v, void* reducescatter_v,
                                         void* alltoall_v) {
    g_hccl_allreduce = reinterpret_cast<HcclAllReduceFn>(allreduce);
    g_hccl_allgather = reinterpret_cast<HcclAllGatherFn>(allgather);
    g_hccl_reducescatter = reinterpret_cast<HcclReduceScatterFn>(reducescatter);
    g_hccl_alltoall = reinterpret_cast<HcclAlltoAllFn>(alltoall_8arg);
    g_hccl_alltoall_6arg = reinterpret_cast<HcclAlltoAll_6ArgFn>(alltoall_6arg);
    g_hccl_allgather_v = reinterpret_cast<HcclAllGatherVFn>(allgather_v);
    g_hccl_reducescatter_v = reinterpret_cast<HcclReduceScatterVFn>(reducescatter_v);
    g_hccl_alltoall_v = reinterpret_cast<HcclAlltoAllVFn>(alltoall_v);
}

void OriginalCollectiveAPI::RegisterNccl(void* allreduce, void* allgather, void* reducescatter,
                                         void* broadcast, void* alltoall) {
    g_nccl_allreduce = reinterpret_cast<NcclAllReduceFn>(allreduce);
    g_nccl_allgather = reinterpret_cast<NcclAllGatherFn>(allgather);
    g_nccl_reducescatter = reinterpret_cast<NcclReduceScatterFn>(reducescatter);
    g_nccl_broadcast = reinterpret_cast<NcclBroadcastFn>(broadcast);
    g_nccl_alltoall = reinterpret_cast<NcclAllToAllFn>(alltoall);
}

void OriginalCollectiveAPI::RegisterNcclP2P(void* send, void* recv, void* group_start,
                                            void* group_end, void* comm_count,
                                            void* comm_user_rank, void* reduce) {
    NcclP2PVCollectives::Register(
        reinterpret_cast<NcclP2PVCollectives::NcclSendFn>(send),
        reinterpret_cast<NcclP2PVCollectives::NcclRecvFn>(recv),
        reinterpret_cast<NcclP2PVCollectives::NcclGroupStartFn>(group_start),
        reinterpret_cast<NcclP2PVCollectives::NcclGroupEndFn>(group_end),
        reinterpret_cast<NcclP2PVCollectives::NcclCommCountFn>(comm_count),
        reinterpret_cast<NcclP2PVCollectives::NcclCommUserRankFn>(comm_user_rank),
        reinterpret_cast<NcclP2PVCollectives::NcclReduceFn>(reduce));
}

bool OriginalCollectiveAPI::HasHcclAllGatherV() {
    return g_hccl_allgather_v != nullptr;
}
bool OriginalCollectiveAPI::HasHcclReduceScatterV() {
    return g_hccl_reducescatter_v != nullptr;
}
bool OriginalCollectiveAPI::HasHcclAlltoAllV() {
    return g_hccl_alltoall_v != nullptr;
}

bool OriginalCollectiveAPI::HasNcclP2P() {
    return NcclP2PVCollectives::Available();
}

void OriginalCollectiveAPI::RegisterNcclNativeAllGatherV(void* allgatherv) {
    g_nccl_allgather_v = reinterpret_cast<NcclAllGatherVFn>(allgatherv);
}

bool OriginalCollectiveAPI::HasNcclNativeAllGatherV() {
    return g_nccl_allgather_v != nullptr;
}

bool OriginalCollectiveAPI::HasFastAllGatherV() {
#if defined(AMPCCL_BUILD_HCCL)
    return HasHcclAllGatherV();
#elif defined(AMPCCL_BUILD_NCCL)
    return Config::IsNcclVEnabled() && (HasNcclNativeAllGatherV() || HasNcclP2P());
#else
    return false;
#endif
}

bool OriginalCollectiveAPI::HasFastReduceScatterV() {
#if defined(AMPCCL_BUILD_HCCL)
    return HasHcclReduceScatterV();
#elif defined(AMPCCL_BUILD_NCCL)
    return Config::IsNcclVEnabled() && NcclP2PVCollectives::ReduceScatterVAvailable();
#else
    return false;
#endif
}

bool OriginalCollectiveAPI::HasFastAllToAllV() {
#if defined(AMPCCL_BUILD_HCCL)
    return HasHcclAlltoAllV();
#elif defined(AMPCCL_BUILD_NCCL)
    return Config::IsNcclVEnabled() && HasNcclP2P();
#else
    return false;
#endif
}

static BackendResult IntResultToBackend(int r) {
    if (r == 0) return BackendResult(BackendResult::Success, 0);
    AMPCCL_LOG(ERROR, "native collective call failed: code=%d", r);
    return BackendResult(BackendResult::UnhandledError, r);
}
BackendResult OriginalCollectiveAPI::Synchronize(void* stream) {
    if (!g_stream_synchronize) {
        return BackendResult::UnhandledError;
    }
    int r = g_stream_synchronize(stream);
    return IntResultToBackend(r);
}

BackendResult OriginalCollectiveAPI::SynchronizeEvent(void* event) {
    if (!g_event_synchronize || !event) {
        return BackendResult::UnhandledError;
    }
    int r = g_event_synchronize(event);
    return IntResultToBackend(r);
}




BackendResult OriginalCollectiveAPI::FastAllGatherV(const void* sendbuff, size_t sendcount,
                                                    void* recvbuff, const size_t* recvcounts,
                                                    const size_t* rdispls, int datatype,
                                                    void* comm, void* stream) {
#if defined(AMPCCL_BUILD_HCCL)
    if (!g_hccl_allgather_v) {
        return BackendResult::UnhandledError;
    }
    return HcclAllGatherV(const_cast<void*>(sendbuff), static_cast<uint64_t>(sendcount), recvbuff,
                          recvcounts, rdispls, datatype, comm, stream);
#elif defined(AMPCCL_BUILD_NCCL)
    if (g_nccl_allgather_v) {
        int r = g_nccl_allgather_v(sendbuff, sendcount, recvbuff, recvcounts, rdispls, datatype,
                                   comm, stream);
        return IntResultToBackend(r);
    }
    return NcclAllGatherV(sendbuff, sendcount, recvbuff, recvcounts, rdispls, datatype, comm,
                          stream);
#else
    (void)sendbuff;
    (void)sendcount;
    (void)recvbuff;
    (void)recvcounts;
    (void)rdispls;
    (void)datatype;
    (void)comm;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult OriginalCollectiveAPI::FastReduceScatterV(const void* sendbuff,
                                                        const size_t* sendcounts,
                                                        const size_t* sdispls, void* recvbuff,
                                                        size_t recvcount, int datatype, int op,
                                                        void* comm, void* stream) {
#if defined(AMPCCL_BUILD_HCCL)
    if (!g_hccl_reducescatter_v) {
        return BackendResult::UnhandledError;
    }
    return HcclReduceScatterV(const_cast<void*>(sendbuff), sendcounts, sdispls, recvbuff,
                              static_cast<uint64_t>(recvcount), datatype, op, comm, stream);
#elif defined(AMPCCL_BUILD_NCCL)
    return NcclReduceScatterV(sendbuff, sendcounts, sdispls, recvbuff, recvcount, datatype, op,
                              comm, stream);
#else
    (void)sendbuff;
    (void)sendcounts;
    (void)sdispls;
    (void)recvbuff;
    (void)recvcount;
    (void)datatype;
    (void)op;
    (void)comm;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult OriginalCollectiveAPI::FastAllToAllV(const void* sendbuff,
                                                   const size_t* sendcounts,
                                                   const size_t* sdispls, void* recvbuff,
                                                   const size_t* recvcounts,
                                                   const size_t* rdispls, int datatype,
                                                   void* comm, void* stream) {
#if defined(AMPCCL_BUILD_HCCL)
    if (!g_hccl_alltoall_v) {
        return BackendResult::UnhandledError;
    }
    return HcclAlltoAllV(sendbuff, sendcounts, sdispls, datatype, recvbuff, recvcounts, rdispls,
                         datatype, comm, stream);
#elif defined(AMPCCL_BUILD_NCCL)
    return NcclAllToAllV(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
                         comm, stream);
#else
    (void)sendbuff;
    (void)sendcounts;
    (void)sdispls;
    (void)recvbuff;
    (void)recvcounts;
    (void)rdispls;
    (void)datatype;
    (void)comm;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult OriginalCollectiveAPI::HcclAllGatherV(void* sendbuff, uint64_t send_count_elems,
                                                    void* recvbuff, const void* recv_counts_dev,
                                                    const void* recv_displs_dev, int datatype,
                                                    void* comm, void* stream) {
    if (!g_hccl_allgather_v) {
        return BackendResult::UnhandledError;
    }
    int r = g_hccl_allgather_v(sendbuff, send_count_elems, recvbuff, recv_counts_dev, recv_displs_dev,
                               datatype, comm, stream);
    return IntResultToBackend(r);
}

BackendResult OriginalCollectiveAPI::HcclReduceScatterV(void* sendbuff, const void* send_counts_dev,
                                                      const void* send_displs_dev, void* recvbuff,
                                                      uint64_t recv_count_elems, int datatype, int op,
                                                      void* comm, void* stream) {
    if (!g_hccl_reducescatter_v) {
        return BackendResult::UnhandledError;
    }
    int r = g_hccl_reducescatter_v(sendbuff, send_counts_dev, send_displs_dev, recvbuff,
                                   recv_count_elems, datatype, op, comm, stream);
    return IntResultToBackend(r);
}

BackendResult OriginalCollectiveAPI::HcclAlltoAllV(const void* sendbuff, const void* send_counts_dev,
                                                   const void* sdispls_dev, int send_datatype,
                                                   void* recvbuff, const void* recv_counts_dev,
                                                   const void* rdispls_dev, int recv_datatype,
                                                   void* comm, void* stream) {
    if (!g_hccl_alltoall_v) {
        return BackendResult::UnhandledError;
    }
    int r = g_hccl_alltoall_v(sendbuff, send_counts_dev, sdispls_dev, send_datatype, recvbuff,
                              recv_counts_dev, rdispls_dev, recv_datatype, comm, stream);
    return IntResultToBackend(r);
}

BackendResult OriginalCollectiveAPI::NcclAllGatherV(const void* sendbuff, size_t sendcount,
                                                    void* recvbuff, const size_t* recvcounts,
                                                    const size_t* rdispls, int datatype,
                                                    void* comm, void* stream) {
    return NcclP2PVCollectives::AllGatherV(sendbuff, sendcount, recvbuff, recvcounts, rdispls,
                                          datatype, comm, stream);
}

BackendResult OriginalCollectiveAPI::NcclAllToAllV(const void* sendbuff, const size_t* sendcounts,
                                                   const size_t* sdispls, void* recvbuff,
                                                   const size_t* recvcounts, const size_t* rdispls,
                                                   int datatype, void* comm, void* stream) {
    return NcclP2PVCollectives::AllToAllV(sendbuff, sendcounts, sdispls, recvbuff, recvcounts,
                                         rdispls, datatype, comm, stream);
}

BackendResult OriginalCollectiveAPI::NcclReduceScatterV(const void* sendbuff,
                                                        const size_t* sendcounts,
                                                        const size_t* sdispls, void* recvbuff,
                                                        size_t recvcount, int datatype, int op,
                                                        void* comm, void* stream) {
    return NcclP2PVCollectives::ReduceScatterV(sendbuff, sendcounts, sdispls, recvbuff,
                                              recvcount, datatype, op, comm, stream);
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
        int r = g_hccl_reducescatter(const_cast<void*>(sendbuff), recvbuff,
                                     static_cast<unsigned long>(recvcount), datatype, op, comm, stream);
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
    if (g_nccl_broadcast) {
        int r = g_nccl_broadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
        return IntResultToBackend(r);
    }
    return BackendResult::UnhandledError;
}

BackendResult OriginalCollectiveAPI::AllToAll(
    const void* sendbuff, void* recvbuff, size_t count,
    int datatype, void* comm, void* stream) {
    const uint64_t c = static_cast<uint64_t>(count);
    // Prefer 6-arg legacy, then 8-arg canonical, then NCCL.
    if (g_hccl_alltoall_6arg) {
        int r = g_hccl_alltoall_6arg(sendbuff, recvbuff, c, datatype, comm, stream);
        return IntResultToBackend(r);
    }
    if (g_hccl_alltoall) {
        int r = g_hccl_alltoall(sendbuff, c, datatype, recvbuff, c, datatype, comm, stream);
        return IntResultToBackend(r);
    }
    if (g_nccl_alltoall) {
        int r = g_nccl_alltoall(sendbuff, recvbuff, count, datatype, comm, stream);
        return IntResultToBackend(r);
    }
    return BackendResult::UnhandledError;
}

}  // namespace ampccl
