#ifndef HCCL_TYPES_H_
#define HCCL_TYPES_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HCCL_SUCCESS = 0,
  HCCL_E_PARA = 1,
  HCCL_E_PTR = 2,
  HCCL_E_MEMORY = 3,
  HCCL_E_INTERNAL = 4,
  HCCL_E_NOT_SUPPORT = 5,
  HCCL_E_NOT_FOUND = 6,
  HCCL_E_UNAVAIL = 7,
  HCCL_E_SYSCALL = 8,
  HCCL_E_TIMEOUT = 9,
  HCCL_E_RUNTIME = 15,
  HCCL_E_RESERVED
} HcclResult;

typedef void* HcclComm;
typedef void* HcclConn;
typedef void* CommSymWindow;

typedef enum {
  HCCL_WIN_DEFAULT = 0,
  HCCL_WIN_COLL_SYMMETRIC = 1
} symmetricMemoryFlag;

typedef enum {
  HCCL_REDUCE_SUM = 0,
  HCCL_REDUCE_PROD = 1,
  HCCL_REDUCE_MAX = 2,
  HCCL_REDUCE_MIN = 3,
  HCCL_REDUCE_RESERVED = 255
} HcclReduceOp;

typedef enum {
  HCCL_DATA_TYPE_INT8 = 0,
  HCCL_DATA_TYPE_INT16 = 1,
  HCCL_DATA_TYPE_INT32 = 2,
  HCCL_DATA_TYPE_FP16 = 3,
  HCCL_DATA_TYPE_FP32 = 4,
  HCCL_DATA_TYPE_INT64 = 5,
  HCCL_DATA_TYPE_UINT64 = 6,
  HCCL_DATA_TYPE_UINT8 = 7,
  HCCL_DATA_TYPE_UINT16 = 8,
  HCCL_DATA_TYPE_UINT32 = 9,
  HCCL_DATA_TYPE_FP64 = 10,
  HCCL_DATA_TYPE_BFP16 = 11,
  HCCL_DATA_TYPE_INT128 = 12,
  HCCL_DATA_TYPE_RESERVED = 255
} HcclDataType;

typedef enum {
  HCCL_DETERMINISTIC = 0,
  HCCL_CONFIG_RESERVED
} HcclConfig;

union HcclConfigValue {
  int32_t value;
};

enum {
  HCCL_ROOT_INFO_BYTES = 4108,
  HCCL_COMM_CONFIG_INFO_BYTES = 24,
  COMM_NAME_MAX_LENGTH = 128,
  BUFFER_NAME_MAX_LENGTH = 128,
  UDI_MAX_LENGTH = 128,
  HCCL_COMM_ALGO_MAX_LENGTH = 1600,
  HCCL_COMM_RETRY_ENABLE_MAX_LENGTH = 50,
  HCCL_COMM_RETRY_PARAMS_MAX_LENGTH = 128
};

typedef struct HcclRootInfoDef {
  char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

typedef struct HcclCommConfigDef {
  char reserved[HCCL_COMM_CONFIG_INFO_BYTES];
  uint32_t hcclBufferSize;
  uint32_t hcclDeterministic;
  char hcclCommName[COMM_NAME_MAX_LENGTH];
  char hcclUdi[UDI_MAX_LENGTH];
  uint32_t hcclOpExpansionMode;
  uint32_t hcclRdmaTrafficClass;
  uint32_t hcclRdmaServiceLevel;
  uint32_t hcclWorldRankID;
  uint64_t hcclJobID;
  uint8_t aclGraphZeroCopyEnable;
  int32_t hcclExecTimeOut;
  char hcclAlgo[HCCL_COMM_ALGO_MAX_LENGTH];
  char hcclRetryEnable[HCCL_COMM_RETRY_ENABLE_MAX_LENGTH];
  char hcclRetryParams[HCCL_COMM_RETRY_PARAMS_MAX_LENGTH];
  char hcclBufferName[BUFFER_NAME_MAX_LENGTH];
  uint32_t hcclQos;
  uint64_t hcclSymWinMaxMemSizePerRank;
} HcclCommConfig;

typedef enum {
  HCCL_COMM_CONFIG_BUFFER_SIZE = 0,
  HCCL_COMM_CONFIG_DETERMINISTIC = 1,
  HCCL_COMM_CONFIG_COMM_NAME = 2,
  HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3,
  HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV = 4,
  HCCL_COMM_CONFIG_WORLD_RANKID = 5,
  HCCL_COMM_CONFIG_JOBID = 6,
  HCCL_COMM_CONFIG_ACLGRAPH_ZEROCOPY_ENABLE = 7,
  HCCL_COMM_CONFIG_EXEC_TIMEOUT = 8,
  HCCL_COMM_CONFIG_ALGO = 9,
  HCCL_COMM_CONFIG_RETRY = 10,
  HCCL_COMM_CONFIG_BUFFER_NAME = 11,
  HCCL_COMM_CONFIG_RESERVED
} HcclCommConfigCapability;

typedef enum {
  HCCL_SEND = 0,
  HCCL_RECV = 1,
  HCCL_SEND_RECV_RESERVED
} HcclSendRecvType;

typedef struct HcclSendRecvItemDef {
  HcclSendRecvType sendRecvType;
  void* buf;
  uint64_t count;
  HcclDataType dataType;
  uint32_t remoteRank;
} HcclSendRecvItem;

#ifdef __cplusplus
}
#endif

#endif /* HCCL_TYPES_H_ */
