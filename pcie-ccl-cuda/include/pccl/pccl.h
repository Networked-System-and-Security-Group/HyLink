#ifndef PCCL_H
#define PCCL_H

#include "types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generate a unique ID for multi-process coordination.
 * Rank 0 calls this and broadcasts the result to all ranks. */
pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId);

/* Process-level finalization */
pcclResult_t pcclFinalize(void);

/* Per-GPU communicator management (auto-initializes process state on first call).
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

pcclResult_t pcclReduceScatter(
    const void* sendbuff, void* recvbuff,
    size_t recvcount,
    pcclComm_t comm,
    pcclStream_t stream);

pcclResult_t pcclAllToAll(
    const void* sendbuff, void* recvbuff,
    size_t count,
    pcclComm_t comm,
    pcclStream_t stream);

/*
 * Strided variants for MPCCL multi-path splitting.
 *
 * When MPCCL splits a collective into a fast path (e.g. NVLink) and a PCIe
 * path, the fast path handles the first `stride` elements of each per-rank
 * segment, and PCCL handles the remainder.  For buffers that are laid out as
 * consecutive per-rank segments (e.g. AllGather recvbuf, ReduceScatter
 * sendbuf), PCCL must skip over the fast-path portion in *every* segment,
 * producing a strided (non-contiguous) access pattern on the device buffer.
 *
 * Parameters:
 *   count  — Meaning varies by collective:
 *              AllGatherStrided:      per-segment element count (= sendcount).
 *              ReduceScatterStrided:  total element count (= recvcount * nranks).
 *                                    Internally divided by nranks → seg_count.
 *              AllToAllStrided:       total element count (= per-rank count * nranks).
 *                                    Internally divided by nranks → seg_count.
 *            In all cases, the per-segment size includes both fast-path and
 *            PCIe portions: seg_count = count (or count/nranks) elements.
 *   stride — Number of leading elements per segment handled by the fast path.
 *            PCCL operates on the (seg_count - stride) elements after this
 *            region in each segment.  0 means no split (equivalent to
 *            non-strided).
 *
 * Address formula (per chunk):
 *   segment_size   = seg_count * elem_size
 *   stride_offset  = stride * elem_size
 *   pcie_per_seg   = segment_size - stride_offset
 *   chunks_per_seg = device_chunks / num_segments
 *   chunk_size     = pcie_per_seg / chunks_per_seg
 *
 *   addr = base + seg * segment_size + stride_offset + chunk_in_seg * chunk_size
 *          where seg = chunk_idx / chunks_per_seg
 *                chunk_in_seg = chunk_idx % chunks_per_seg
 *
 * Example (AllGather, 4 ranks, count=64M, stride=48M):
 *
 *   recvbuf layout (4 segments of 256 MB each):
 *     seg0              seg1              seg2              seg3
 *     [fast 192M|pcie 64M] [fast 192M|pcie 64M] [fast 192M|pcie 64M] [fast 192M|pcie 64M]
 *                ^chunk0                ^chunk1                ^chunk2                ^chunk3
 *
 *   PCCL writes chunk k at: recvbuf + k * 256MB + 192MB
 */
pcclResult_t pcclAllGatherStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclReduceScatterStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllToAllStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

/* Internal typed submission entry used by the NCCL compatibility layer. */
#ifdef __cplusplus
}

namespace pccl {
struct CompileOptions;
pcclResult_t pcclSubmitTypedCollective(
    pcclComm_t comm, const char* collective,
    const void* sendbuff, void* recvbuff,
    size_t seg_count, size_t stride,
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
