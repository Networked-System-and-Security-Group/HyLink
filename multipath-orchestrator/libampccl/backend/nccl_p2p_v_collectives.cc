#include "nccl_p2p_v_collectives.h"

#include "backend/dtype/native_dtype.h"
#include "backend/memory/device_mem.h"
#include "common/log.h"

namespace ampccl {

namespace {

NcclP2PVCollectives::NcclSendFn g_send = nullptr;
NcclP2PVCollectives::NcclRecvFn g_recv = nullptr;
NcclP2PVCollectives::NcclGroupStartFn g_group_start = nullptr;
NcclP2PVCollectives::NcclGroupEndFn g_group_end = nullptr;
NcclP2PVCollectives::NcclCommCountFn g_comm_count = nullptr;
NcclP2PVCollectives::NcclCommUserRankFn g_comm_user_rank = nullptr;
NcclP2PVCollectives::NcclReduceFn g_reduce = nullptr;

BackendResult NcclResultToBackend(int r) {
    if (r == 0) return BackendResult(BackendResult::Success, 0);
    AMPCCL_LOG(ERROR, "NCCL P2P call failed: code=%d", r);
    return BackendResult(BackendResult::UnhandledError, r);
}

char* ElemPtr(void* base, size_t elem_offset, size_t elem_size) {
    return static_cast<char*>(base) + elem_offset * elem_size;
}

const char* ElemPtr(const void* base, size_t elem_offset, size_t elem_size) {
    return static_cast<const char*>(base) + elem_offset * elem_size;
}

BackendResult GetCommRankCount(void* comm, int* rank, int* nranks) {
    if (!comm || !rank || !nranks || !g_comm_count || !g_comm_user_rank) {
        return BackendResult::InvalidArgument;
    }
    int n = 0;
    int r = 0;
    if (g_comm_count(comm, &n) != 0 || g_comm_user_rank(comm, &r) != 0 || n <= 0 || r < 0 ||
        r >= n) {
        return BackendResult::UnhandledError;
    }
    *rank = r;
    *nranks = n;
    return BackendResult::Success;
}

bool SameRange(const void* src, const void* dst) {
    return src == dst;
}

}  // namespace

void NcclP2PVCollectives::Register(NcclSendFn send, NcclRecvFn recv,
                                   NcclGroupStartFn group_start, NcclGroupEndFn group_end,
                                   NcclCommCountFn comm_count,
                                   NcclCommUserRankFn comm_user_rank, NcclReduceFn reduce) {
    g_send = send;
    g_recv = recv;
    g_group_start = group_start;
    g_group_end = group_end;
    g_comm_count = comm_count;
    g_comm_user_rank = comm_user_rank;
    g_reduce = reduce;
}

bool NcclP2PVCollectives::Available() {
    return g_send && g_recv && g_group_start && g_group_end && g_comm_count && g_comm_user_rank;
}

bool NcclP2PVCollectives::ReduceScatterVAvailable() {
    return Available() && g_reduce;
}

BackendResult NcclP2PVCollectives::AllGatherV(const void* sendbuff, size_t sendcount,
                                              void* recvbuff, const size_t* recvcounts,
                                              const size_t* rdispls, int datatype, void* comm,
                                              void* stream) {
    if (!Available()) {
        return BackendResult::UnhandledError;
    }
    if (!sendbuff || !recvbuff || !recvcounts || !rdispls || !comm || !stream) {
        return BackendResult::InvalidArgument;
    }
    const size_t elem_size = NativeDataTypeElemSize(datatype);
    if (elem_size == 0) {
        return BackendResult::InvalidArgument;
    }
    int rank = 0;
    int nranks = 0;
    BackendResult rc = GetCommRankCount(comm, &rank, &nranks);
    if (rc != BackendResult::Success) {
        return rc;
    }
    if (recvcounts[static_cast<size_t>(rank)] != sendcount) {
        AMPCCL_LOG(WARN, "Nccl AllGatherV fallback: recvcounts[rank] != sendcount");
        return BackendResult::InvalidArgument;
    }

    void* local_dst = ElemPtr(recvbuff, rdispls[static_cast<size_t>(rank)], elem_size);
    const size_t local_bytes = sendcount * elem_size;
    if (local_bytes > 0 && !SameRange(sendbuff, local_dst)) {
        DeviceMemcpyD2DAsync(local_dst, sendbuff, local_bytes, stream);
    }

    int r = g_group_start();
    if (r != 0) {
        return NcclResultToBackend(r);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        if (sendcount > 0) {
            r = g_send(sendbuff, sendcount, datatype, peer, comm, stream);
            if (r != 0) {
                (void)g_group_end();
                return NcclResultToBackend(r);
            }
        }
        const size_t recv_count = recvcounts[static_cast<size_t>(peer)];
        if (recv_count > 0) {
            void* recv_ptr = ElemPtr(recvbuff, rdispls[static_cast<size_t>(peer)], elem_size);
            r = g_recv(recv_ptr, recv_count, datatype, peer, comm, stream);
            if (r != 0) {
                (void)g_group_end();
                return NcclResultToBackend(r);
            }
        }
    }
    return NcclResultToBackend(g_group_end());
}

BackendResult NcclP2PVCollectives::AllToAllV(const void* sendbuff, const size_t* sendcounts,
                                             const size_t* sdispls, void* recvbuff,
                                             const size_t* recvcounts, const size_t* rdispls,
                                             int datatype, void* comm, void* stream) {
    if (!Available()) {
        return BackendResult::UnhandledError;
    }
    if (!sendbuff || !recvbuff || !sendcounts || !sdispls || !recvcounts || !rdispls || !comm ||
        !stream) {
        return BackendResult::InvalidArgument;
    }
    const size_t elem_size = NativeDataTypeElemSize(datatype);
    if (elem_size == 0) {
        return BackendResult::InvalidArgument;
    }
    int rank = 0;
    int nranks = 0;
    BackendResult rc = GetCommRankCount(comm, &rank, &nranks);
    if (rc != BackendResult::Success) {
        return rc;
    }

    const size_t self = static_cast<size_t>(rank);
    const void* local_src = ElemPtr(sendbuff, sdispls[self], elem_size);
    void* local_dst = ElemPtr(recvbuff, rdispls[self], elem_size);
    const size_t local_bytes = sendcounts[self] * elem_size;
    if (sendcounts[self] != recvcounts[self]) {
        AMPCCL_LOG(WARN, "Nccl AllToAllV fallback: self sendcount != self recvcount");
        return BackendResult::InvalidArgument;
    }
    if (local_bytes > 0 && !SameRange(local_src, local_dst)) {
        DeviceMemcpyD2DAsync(local_dst, local_src, local_bytes, stream);
    }

    int r = g_group_start();
    if (r != 0) {
        return NcclResultToBackend(r);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        const size_t p = static_cast<size_t>(peer);
        if (sendcounts[p] > 0) {
            const void* send_ptr = ElemPtr(sendbuff, sdispls[p], elem_size);
            r = g_send(send_ptr, sendcounts[p], datatype, peer, comm, stream);
            if (r != 0) {
                (void)g_group_end();
                return NcclResultToBackend(r);
            }
        }
        if (recvcounts[p] > 0) {
            void* recv_ptr = ElemPtr(recvbuff, rdispls[p], elem_size);
            r = g_recv(recv_ptr, recvcounts[p], datatype, peer, comm, stream);
            if (r != 0) {
                (void)g_group_end();
                return NcclResultToBackend(r);
            }
        }
    }
    return NcclResultToBackend(g_group_end());
}

BackendResult NcclP2PVCollectives::ReduceScatterV(const void* sendbuff,
                                                  const size_t* sendcounts,
                                                  const size_t* sdispls, void* recvbuff,
                                                  size_t recvcount, int datatype, int op,
                                                  void* comm, void* stream) {
    if (!ReduceScatterVAvailable()) {
        return BackendResult::UnhandledError;
    }
    if (!sendbuff || !sendcounts || !sdispls || !recvbuff || !comm || !stream) {
        return BackendResult::InvalidArgument;
    }
    const size_t elem_size = NativeDataTypeElemSize(datatype);
    if (elem_size == 0) {
        return BackendResult::InvalidArgument;
    }
    int rank = 0;
    int nranks = 0;
    BackendResult rc = GetCommRankCount(comm, &rank, &nranks);
    if (rc != BackendResult::Success) {
        return rc;
    }
    if (sendcounts[static_cast<size_t>(rank)] != recvcount) {
        AMPCCL_LOG(WARN, "Nccl ReduceScatterV fallback: sendcounts[rank] != recvcount");
        return BackendResult::InvalidArgument;
    }

    for (int root = 0; root < nranks; ++root) {
        const size_t r = static_cast<size_t>(root);
        const size_t count = sendcounts[r];
        const void* src = ElemPtr(sendbuff, sdispls[r], elem_size);
        void* dst = (root == rank) ? recvbuff : const_cast<void*>(src);
        int st = g_reduce(src, dst, count, datatype, op, root, comm, stream);
        if (st != 0) {
            return NcclResultToBackend(st);
        }
    }
    return BackendResult::Success;
}

}  // namespace ampccl
