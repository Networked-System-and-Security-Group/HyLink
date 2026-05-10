#ifndef AMPCCL_BACKEND_ORIGINAL_COLLECTIVE_API_H_
#define AMPCCL_BACKEND_ORIGINAL_COLLECTIVE_API_H_

#include "backend_base.h"
#include <cstddef>

namespace ampccl {

class OriginalCollectiveAPI {
public:
    static void RegisterHccl(
        void* allreduce,
        void* allgather,
        void* reducescatter,
        void* broadcast);

    static void RegisterNccl(
        void* allreduce,
        void* allgather,
        void* reducescatter,
        void* broadcast);

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
};

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_ORIGINAL_COLLECTIVE_API_H_
