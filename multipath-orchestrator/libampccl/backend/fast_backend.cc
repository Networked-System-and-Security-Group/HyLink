#include "fast_backend.h"
#include "original_collective_api.h"

namespace ampccl {


BackendResult BackendBase<FastBackend>::AllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int op,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::AllReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream);
}

BackendResult BackendBase<FastBackend>::AllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::AllGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

BackendResult BackendBase<FastBackend>::ReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::ReduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

BackendResult BackendBase<FastBackend>::Broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int root,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::Broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream);
}

BackendResult BackendBase<FastBackend>::AllToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::AllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

BackendResult BackendBase<FastBackend>::AllGatherV(
    const void* sendbuff,
    size_t sendcount,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* rdispls,
    int datatype,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::NcclAllGatherV(sendbuff, sendcount, recvbuff, recvcounts, rdispls,
                                                datatype, comm, stream);
}

BackendResult BackendBase<FastBackend>::AllToAllV(
    const void* sendbuff,
    const size_t* sendcounts,
    const size_t* sdispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* rdispls,
    int datatype,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::NcclAllToAllV(sendbuff, sendcounts, sdispls, recvbuff, recvcounts,
                                               rdispls, datatype, comm, stream);
}

BackendResult BackendBase<FastBackend>::ReduceScatterV(
    const void* sendbuff,
    const size_t* sendcounts,
    const size_t* sdispls,
    void* recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    void* comm,
    void* stream) {
    return OriginalCollectiveAPI::NcclReduceScatterV(sendbuff, sendcounts, sdispls, recvbuff,
                                                    recvcount, datatype, op, comm, stream);
}

}  // namespace ampccl
