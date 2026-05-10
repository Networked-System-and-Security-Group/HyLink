#ifndef PCCL_H
#define PCCL_H

#include "types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Public typed values are intentionally independent of HCCL/NCCL enum values. */
typedef enum {
    pcclInvalidDataType = -1,
    pcclInt8 = 0,
    pcclUint8 = 1,
    pcclInt32 = 2,
    pcclUint32 = 3,
    pcclInt64 = 4,
    pcclUint64 = 5,
    pcclFloat16 = 6,
    pcclFloat32 = 7,
    pcclFloat64 = 8,
    pcclBfloat16 = 9,
} pcclDataType_t;

typedef enum {
    pcclInvalidReduceOp = -1,
    pcclReduceSum = 0,
    pcclReduceMax = 1,
    pcclReduceMin = 2,
} pcclReduceOp_t;

/* Generate a unique ID for multi-process coordination.
 * Rank 0 calls this and broadcasts the result to all ranks. */
pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId);

/* Process-level finalization */
pcclResult_t pcclFinalize(void);

/* Per-NPU communicator management (auto-initializes process state on first call).
 * For single-process mode, pass a zero-initialized pcclUniqueId. */
pcclResult_t pcclCommInitRank(pcclComm_t* comm, int nranks, pcclUniqueId uniqueId, int rank);
pcclResult_t pcclCommDestroy(pcclComm_t comm);

/* Communicator queries */
pcclResult_t pcclCommRank(pcclComm_t comm, int* rank);
pcclResult_t pcclCommSize(pcclComm_t comm, int* size);

/* Collective operations */
pcclResult_t pcclAllReduce(
    const void* sendbuff, void* recvbuff,
    size_t count,
    pcclComm_t comm,
    pcclStream_t stream);

pcclResult_t pcclAllGather(
    const void* sendbuff, void* recvbuff,
    size_t sendcount,
    pcclComm_t comm,
    pcclStream_t stream);

pcclResult_t pcclAllGatherTyped(
    const void* sendbuff, void* recvbuff,
    size_t sendcount, pcclDataType_t datatype,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclReduceScatter(
    const void* sendbuff, void* recvbuff,
    size_t recvcount,
    pcclComm_t comm,
    pcclStream_t stream);

pcclResult_t pcclReduceScatterTyped(
    const void* sendbuff, void* recvbuff,
    size_t recvcount, pcclDataType_t datatype, pcclReduceOp_t op,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllToAll(
    const void* sendbuff, void* recvbuff,
    size_t count,
    pcclComm_t comm,
    pcclStream_t stream);

pcclResult_t pcclAllToAllTyped(
    const void* sendbuff, void* recvbuff,
    size_t count, pcclDataType_t datatype,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllReduceTyped(
    const void* sendbuff, void* recvbuff,
    size_t count, pcclDataType_t datatype, pcclReduceOp_t op,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclBroadcastRoot(
    const void* sendbuff, void* recvbuff,
    size_t count, int root,
    pcclComm_t comm,
    pcclStream_t stream);

/* Strided variants for segmented buffers with a per-segment leading offset.
 *
 * The `stride` argument is the number of leading elements reserved in every
 * segment. PCCL operates on the remaining `(segment_count - stride)` elements
 * while preserving the reserved prefix.
 */
pcclResult_t pcclAllGatherStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllGatherStridedTyped(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride, pcclDataType_t datatype,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclReduceScatterStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclReduceScatterStridedTyped(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride, pcclDataType_t datatype, pcclReduceOp_t op,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllToAllStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllToAllStridedTyped(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride, pcclDataType_t datatype,
    pcclComm_t comm, pcclStream_t stream);

/* Internal typed submission entry used by the HCCL compatibility layer. */
#ifdef __cplusplus
}

namespace pccl {
struct CompileOptions;
pcclResult_t pcclSubmitTypedCollective(
    pcclComm_t comm, const char* collective,
    const void* sendbuff, void* recvbuff,
    size_t seg_count, size_t reserved_offset,
    int send_segs, int recv_segs,
    pcclStream_t stream,
    const CompileOptions* options);
}

extern "C" {
#endif

/* Error string */
const char* pcclGetErrorString(pcclResult_t result);

#ifdef __cplusplus
}
#endif

#endif /* PCCL_H */
