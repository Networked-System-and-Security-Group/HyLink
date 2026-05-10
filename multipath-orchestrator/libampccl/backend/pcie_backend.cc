#include "pcie_backend.h"
#include "backend/dtype/pccl_dtype.h"
#include "core/domain.h"

#ifdef AMPCCL_ENABLE_PCIE
#include <pccl/pccl.h>
#endif

namespace ampccl {

#ifdef AMPCCL_ENABLE_PCIE

static pcclComm_t GetComm(CommDomain* domain) {
    return static_cast<pcclComm_t>(domain->pcie_comm());
}

static pcclStream_t GetStream(void* stream) {
    return static_cast<pcclStream_t>(stream);
}

static bool GetPcclDataType(int datatype, pcclDataType_t* mapped) {
    return MapHcclDataTypeToPccl(datatype, mapped) &&
           HcclPcclDataTypeSizeMatches(datatype, *mapped);
}

static bool GetPcclReduceOp(int op, pcclReduceOp_t* mapped) {
    return MapHcclReduceOpToPccl(op, mapped);
}

#endif

bool BackendBase<PCIeBackend>::SupportsAllReduce(int datatype, int op) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    pcclReduceOp_t pccl_op = pcclInvalidReduceOp;
    return GetPcclDataType(datatype, &pccl_datatype) && GetPcclReduceOp(op, &pccl_op);
#else
    (void)datatype;
    (void)op;
    return false;
#endif
}

bool BackendBase<PCIeBackend>::SupportsAllGather(int datatype) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    return GetPcclDataType(datatype, &pccl_datatype);
#else
    (void)datatype;
    return false;
#endif
}

bool BackendBase<PCIeBackend>::SupportsReduceScatter(int datatype, int op) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    pcclReduceOp_t pccl_op = pcclInvalidReduceOp;
    return GetPcclDataType(datatype, &pccl_datatype) && GetPcclReduceOp(op, &pccl_op);
#else
    (void)datatype;
    (void)op;
    return false;
#endif
}

bool BackendBase<PCIeBackend>::SupportsAllToAll(int datatype) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    return GetPcclDataType(datatype, &pccl_datatype);
#else
    (void)datatype;
    return false;
#endif
}

BackendResult BackendBase<PCIeBackend>::AllReduce(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int op,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsAllReduce(datatype, op)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    pcclReduceOp_t pccl_op = pcclInvalidReduceOp;
    if (!GetPcclDataType(datatype, &pccl_datatype) || !GetPcclReduceOp(op, &pccl_op)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclAllReduceTyped(sendbuff, recvbuff, count, pccl_datatype, pccl_op,
                                         comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                             : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::AllGather(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsAllGather(datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    if (!GetPcclDataType(datatype, &pccl_datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclAllGatherTyped(sendbuff, recvbuff, sendcount, pccl_datatype,
                                         comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                              : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)sendcount;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::AllGatherStrided(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    size_t stride_elems,
    int datatype,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsAllGather(datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    if (!GetPcclDataType(datatype, &pccl_datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclAllGatherStridedTyped(sendbuff, recvbuff, sendcount, stride_elems,
                                               pccl_datatype, comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                              : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)sendcount;
    (void)stride_elems;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::ReduceScatter(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsReduceScatter(datatype, op)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    pcclReduceOp_t pccl_op = pcclInvalidReduceOp;
    if (!GetPcclDataType(datatype, &pccl_datatype) || !GetPcclReduceOp(op, &pccl_op)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclReduceScatterTyped(sendbuff, recvbuff, recvcount, pccl_datatype,
                                             pccl_op, comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                              : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)recvcount;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::ReduceScatterStrided(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t total_elems,
    size_t stride_elems,
    int datatype,
    int op,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsReduceScatter(datatype, op)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    pcclReduceOp_t pccl_op = pcclInvalidReduceOp;
    if (!GetPcclDataType(datatype, &pccl_datatype) || !GetPcclReduceOp(op, &pccl_op)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclReduceScatterStridedTyped(sendbuff, recvbuff, total_elems, stride_elems,
                                                   pccl_datatype, pccl_op, comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                              : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)total_elems;
    (void)stride_elems;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::AllToAll(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsAllToAll(datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    if (!GetPcclDataType(datatype, &pccl_datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclAllToAllTyped(sendbuff, recvbuff, count, pccl_datatype,
                                        comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                              : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::AllToAllStrided(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t total_elems,
    size_t stride_elems,
    int datatype,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    pcclComm_t comm = GetComm(domain);
    if (!comm) {
        return BackendResult::UnhandledError;
    }
    if (!SupportsAllToAll(datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclDataType_t pccl_datatype = pcclInvalidDataType;
    if (!GetPcclDataType(datatype, &pccl_datatype)) {
        return BackendResult::InvalidArgument;
    }
    pcclResult_t r = pcclAllToAllStridedTyped(sendbuff, recvbuff, total_elems, stride_elems,
                                              pccl_datatype, comm, GetStream(stream));
    return (r == pcclSuccess) ? BackendResult(BackendResult::Success, 0)
                              : BackendResult(BackendResult::UnhandledError, static_cast<int>(r));
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)total_elems;
    (void)stride_elems;
    (void)stream;
    return BackendResult::UnhandledError;
#endif
}

BackendResult BackendBase<PCIeBackend>::Broadcast(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int root,
    void* stream) {
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)root;
    (void)stream;
    return BackendResult::UnhandledError;
}

}  // namespace ampccl
