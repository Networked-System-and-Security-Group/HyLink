#ifndef AMPCCL_BACKEND_NCCL_P2P_V_COLLECTIVES_H_
#define AMPCCL_BACKEND_NCCL_P2P_V_COLLECTIVES_H_

#include "backend/backend_base.h"
#include <cstddef>

namespace ampccl {

class NcclP2PVCollectives {
public:
    using NcclSendFn = int (*)(const void*, size_t, int, int, void*, void*);
    using NcclRecvFn = int (*)(void*, size_t, int, int, void*, void*);
    using NcclGroupStartFn = int (*)();
    using NcclGroupEndFn = int (*)();
    using NcclCommCountFn = int (*)(void*, int*);
    using NcclCommUserRankFn = int (*)(void*, int*);
    using NcclReduceFn = int (*)(const void*, void*, size_t, int, int, int, void*, void*);

    static void Register(NcclSendFn send, NcclRecvFn recv, NcclGroupStartFn group_start,
                         NcclGroupEndFn group_end, NcclCommCountFn comm_count,
                         NcclCommUserRankFn comm_user_rank, NcclReduceFn reduce);

    static bool Available();
    static bool ReduceScatterVAvailable();

    static BackendResult AllGatherV(const void* sendbuff, size_t sendcount, void* recvbuff,
                                    const size_t* recvcounts, const size_t* rdispls,
                                    int datatype, void* comm, void* stream);

    static BackendResult AllToAllV(const void* sendbuff, const size_t* sendcounts,
                                   const size_t* sdispls, void* recvbuff,
                                   const size_t* recvcounts, const size_t* rdispls,
                                   int datatype, void* comm, void* stream);

    static BackendResult ReduceScatterV(const void* sendbuff, const size_t* sendcounts,
                                        const size_t* sdispls, void* recvbuff,
                                        size_t recvcount, int datatype, int op,
                                        void* comm, void* stream);
};

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_NCCL_P2P_V_COLLECTIVES_H_
