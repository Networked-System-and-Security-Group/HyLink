#ifndef PCCL_TYPES_H
#define PCCL_TYPES_H

#include <acl/acl_rt.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    pcclSuccess = 0,
    pcclInvalidArgument = 1,
    pcclSystemError = 2,
    pcclInternalError = 3,
    pcclDeviceError = 4,
    pcclNetworkError = 5,
    pcclTimeout = 6
} pcclResult_t;

typedef struct pcclComm* pcclComm_t;
typedef aclrtStream pcclStream_t;

#define PCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[PCCL_UNIQUE_ID_BYTES]; } pcclUniqueId;

#ifdef __cplusplus
}
#endif

#endif /* PCCL_TYPES_H */
