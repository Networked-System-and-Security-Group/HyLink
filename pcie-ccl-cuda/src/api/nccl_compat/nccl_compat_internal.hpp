#ifndef PCCL_NCCL_COMPAT_INTERNAL_HPP
#define PCCL_NCCL_COMPAT_INTERNAL_HPP

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../../ir/compiler.hpp"
#include "pccl/pccl.h"

#define NCCL_UNIQUE_ID_BYTES 128

typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

// Opaque comm handle -- binary-compatible with pcclComm_t (both are struct pointers).
typedef pcclComm_t ncclComm_t;

typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 } ncclRedOp_t;

typedef enum {
  ncclInt8 = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclFloat32 = 7,
  ncclFloat64 = 8,
  ncclBfloat16 = 9
} ncclDataType_t;

typedef enum { ncclScalarDevice = 0, ncclScalarHostImmediate = 1 } ncclScalarResidence_t;
typedef struct ncclSimInfo ncclSimInfo_t;

// ncclConfig_t layout for NCCL 2.17.x.
typedef struct {
  size_t size;
  unsigned int magic;
  unsigned int version;
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char* netName;
} ncclConfig_t;

enum class NcclCompatOpType {
  AllReduce,
  AllGather,
  ReduceScatter,
  Broadcast,
  Reduce,
};

struct NcclCompatPendingOp {
  NcclCompatOpType type;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
  int device_id;
};

struct NcclCompatCommState {
  uint64_t id_hash = 0;
  bool multi_process = false;
  uint64_t split_seq = 0;
  uint64_t broadcast_seq = 0;
};

ncclResult_t from_pccl(pcclResult_t r);
long parsePositiveEnvLong(const char* name, long fallback = 0);
uint64_t hashUniqueId(const ncclUniqueId& commId);
void rememberLocallyGeneratedUniqueId(const ncclUniqueId& commId);
ncclResult_t detectCompatMultiProcessByRankPids(const ncclUniqueId& commId, int nranks, int rank,
                                                bool* multi_process);

void rememberCommState(ncclComm_t comm, uint64_t id_hash, bool multi_process);
void forgetCommState(ncclComm_t comm);
bool getCommState(ncclComm_t comm, NcclCompatCommState* state);
uint64_t nextBroadcastSeq(ncclComm_t comm);
uint64_t nextSplitSeq(ncclComm_t comm);

const char* opTypeName(NcclCompatOpType type);
size_t datatypeSize(ncclDataType_t datatype);
bool datatypeToScalar(ncclDataType_t datatype, pccl::ScalarDataType* out);
bool redOpToPccl(ncclRedOp_t op, pccl::ReduceOpKind* out);
ncclResult_t makeCompileOptions(ncclDataType_t datatype, ncclRedOp_t op, pccl::CompileOptions* options,
                                bool reduction);
ncclResult_t validateDatatype(ncclDataType_t datatype);

bool waitForShmSize(int fd, size_t expected_size, int timeout_ms);
void reapDeferredDeviceBuffers(bool block);
ncclResult_t deferDeviceFree(void* ptr, cudaStream_t stream, int device);
ncclResult_t runSingleRankCollective(const NcclCompatPendingOp& op);

#endif /* PCCL_NCCL_COMPAT_INTERNAL_HPP */
