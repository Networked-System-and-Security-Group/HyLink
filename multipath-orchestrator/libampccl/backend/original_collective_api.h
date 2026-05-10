#ifndef AMPCCL_BACKEND_ORIGINAL_COLLECTIVE_API_H_
#define AMPCCL_BACKEND_ORIGINAL_COLLECTIVE_API_H_

#include "backend_base.h"
#include <cstddef>
#include <cstdint>

namespace ampccl {

class OriginalCollectiveAPI {
public:
    // Register synchronize API (hook intercepts but does not re-export)
    static void RegisterSynchronize(void* sync_func);
    static void RegisterSynchronizeEvent(void* sync_event_func);

    // Last three: Hccl*V via dlsym for VirtualCollective hybrid split only (not LD_PRELOAD-exported).
    // alltoall_8arg: canonical 8-arg HcclAlltoAll; alltoall_6arg: legacy 6-arg HcclAllToAll (may be null).
    // AllToAll dispatch tries 6-arg first, then 8-arg, then NCCL.
    static void RegisterHccl(
        void* allreduce,
        void* allgather,
        void* reducescatter,
        void* alltoall_8arg,
        void* alltoall_6arg,
        void* allgather_v,
        void* reducescatter_v,
        void* alltoall_v);

    static void RegisterNccl(
        void* allreduce,
        void* allgather,
        void* reducescatter,
        void* broadcast,
        void* alltoall);

    static void RegisterNcclP2P(
        void* send,
        void* recv,
        void* group_start,
        void* group_end,
        void* comm_count,
        void* comm_user_rank,
        void* reduce);

    // Native ncclAllGatherv (NCCL >= 2.28). Preferred over the send/recv
    // emulation when present -- avoids the extra D2D copy in end-to-end runs.
    static void RegisterNcclNativeAllGatherV(void* allgatherv);
    static bool HasNcclNativeAllGatherV();

    static bool HasHcclAllGatherV();
    static bool HasHcclReduceScatterV();
    static bool HasHcclAlltoAllV();
    static bool HasNcclP2P();
    static bool HasFastAllGatherV();
    static bool HasFastReduceScatterV();
    static bool HasFastAllToAllV();

    static BackendResult Synchronize(void* stream);
    static BackendResult SynchronizeEvent(void* event);

    static BackendResult AllReduce(
        const void* sendbuff, void* recvbuff, size_t count,
        int datatype, int op, void* comm, void* stream);

    static BackendResult AllGather(
        const void* sendbuff, void* recvbuff, size_t sendcount,
        int datatype, void* comm, void* stream);

    static BackendResult ReduceScatter(
        const void* sendbuff, void* recvbuff, size_t recvcount,
        int datatype, int op, void* comm, void* stream);

    static BackendResult Broadcast(
        const void* sendbuff, void* recvbuff, size_t count,
        int datatype, int root, void* comm, void* stream);

    static BackendResult AllToAll(
        const void* sendbuff, void* recvbuff, size_t count,
        int datatype, void* comm, void* stream);

    static BackendResult FastAllGatherV(
        const void* sendbuff,
        size_t sendcount,
        void* recvbuff,
        const size_t* recvcounts,
        const size_t* rdispls,
        int datatype,
        void* comm,
        void* stream);

    static BackendResult FastReduceScatterV(
        const void* sendbuff,
        const size_t* sendcounts,
        const size_t* sdispls,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* comm,
        void* stream);

    static BackendResult FastAllToAllV(
        const void* sendbuff,
        const size_t* sendcounts,
        const size_t* sdispls,
        void* recvbuff,
        const size_t* recvcounts,
        const size_t* rdispls,
        int datatype,
        void* comm,
        void* stream);

    // HCCL V collectives (dlsym); NCCL build leaves pointers null.
    static BackendResult HcclAllGatherV(
        void* sendbuff,
        uint64_t send_count_elems,
        void* recvbuff,
        const void* recv_counts_dev,
        const void* recv_displs_dev,
        int datatype,
        void* comm,
        void* stream);

    static BackendResult HcclReduceScatterV(
        void* sendbuff,
        const void* send_counts_dev,
        const void* send_displs_dev,
        void* recvbuff,
        uint64_t recv_count_elems,
        int datatype,
        int op,
        void* comm,
        void* stream);

    static BackendResult HcclAlltoAllV(
        const void* sendbuff,
        const void* send_counts_dev,
        const void* sdispls_dev,
        int send_datatype,
        void* recvbuff,
        const void* recv_counts_dev,
        const void* rdispls_dev,
        int recv_datatype,
        void* comm,
        void* stream);

    static BackendResult NcclAllGatherV(
        const void* sendbuff,
        size_t sendcount,
        void* recvbuff,
        const size_t* recvcounts,
        const size_t* rdispls,
        int datatype,
        void* comm,
        void* stream);

    static BackendResult NcclAllToAllV(
        const void* sendbuff,
        const size_t* sendcounts,
        const size_t* sdispls,
        void* recvbuff,
        const size_t* recvcounts,
        const size_t* rdispls,
        int datatype,
        void* comm,
        void* stream);

    static BackendResult NcclReduceScatterV(
        const void* sendbuff,
        const size_t* sendcounts,
        const size_t* sdispls,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* comm,
        void* stream);
};

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_ORIGINAL_COLLECTIVE_API_H_
