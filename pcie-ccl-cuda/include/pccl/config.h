#ifndef PCCL_CONFIG_H
#define PCCL_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nranks;                /* Total rank count */
    int device_count;          /* Local GPU/NPU count */
    const int* gpu_ids;        /* Device ID list (NULL = auto) */
    const char* bootstrap_addr; /* "host:port" for multi-node (NULL = single-node) */
    size_t host_buffer_size;   /* Per-NUMA host buffer size (0 = auto) */
    const char* gpu_numa_map;  /* GPU-to-NUMA mapping string (NULL = auto-detect) */
    const char* ir_path;       /* IR XML search path (NULL = default) */
} pcclConfig_t;

#define PCCL_CONFIG_DEFAULT { 0, 0, NULL, NULL, 0, NULL, NULL }

#ifdef __cplusplus
}
#endif

#endif /* PCCL_CONFIG_H */
