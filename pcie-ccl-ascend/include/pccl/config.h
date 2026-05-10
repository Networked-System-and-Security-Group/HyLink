#ifndef PCCL_CONFIG_H
#define PCCL_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nranks;                /* Total rank count */
    int device_count;          /* Local device/NPU count */
    const int* device_ids;     /* Device/NPU ID list (NULL = auto) */
    const char* bootstrap_addr; /* Reserved; multi-node is unsupported in this Ascend build */
    size_t host_buffer_size;   /* Per-NUMA host buffer size (0 = auto) */
    const char* npu_numa_map;  /* NPU-to-NUMA mapping string (NULL = auto-detect) */
    const char* ir_path;       /* IR XML search path (NULL = default) */
    const int* gpu_ids;        /* Legacy alias for device_ids */
    const char* gpu_numa_map;  /* Legacy alias for npu_numa_map */
} pcclConfig_t;

#define PCCL_CONFIG_DEFAULT { 0, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL }

#ifdef __cplusplus
}
#endif

#endif /* PCCL_CONFIG_H */
