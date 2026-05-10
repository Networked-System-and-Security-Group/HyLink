/**
 * PCIe D2H interference - Ascend/ACL version (counterpart of d2h_test.cpp CUDA)
 *
 * Multi-thread, each bound to one NPU, continuous D2H copy to create PCIe contention.
 * Can run alongside HCCL collectives (e.g. AllGather) to observe dynamic interference.
 *
 * Depends: Ascend CANN (ACL), no HCCL.
 * Build: ./build_hccl.sh
 * Run: ./run_hccl.sh  or  ./d2h_test_hccl
 */

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include "acl/acl.h"

#define NUM_DEVICES 4
#define DATA_SIZE (1024 * 1024 * 256)  // 256MB per device

#define ACLCHECK(cmd)                                                                          \
    do {                                                                                       \
        aclError ret = (cmd);                                                                  \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("ACL error %s:%d, ret=%d\n", __FILE__, __LINE__, (int)ret);                 \
            return;                                                                            \
        }                                                                                      \
    } while (0)

struct ThreadArgs {
    int device_id;
};

static void gpu_worker(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    int dev = args->device_id;

    aclError err = aclrtSetDevice(dev);
    if (err != ACL_SUCCESS) {
        printf("NPU %d: aclrtSetDevice failed, ret=%d\n", dev, (int)err);
        return;
    }

    void* h_data = nullptr;
    void* d_data = nullptr;

    err = aclrtMallocHost(&h_data, DATA_SIZE * sizeof(float));
    if (err != ACL_SUCCESS) {
        printf("NPU %d: aclrtMallocHost failed\n", dev);
        return;
    }

    err = aclrtMalloc(&d_data, DATA_SIZE * sizeof(float), ACL_MEM_MALLOC_HUGE_ONLY);
    if (err != ACL_SUCCESS) {
        printf("NPU %d: aclrtMalloc failed\n", dev);
        aclrtFreeHost(h_data);
        return;
    }

    size_t copy_bytes = DATA_SIZE * sizeof(float);

    while (1) {
        err = aclrtMemcpy(h_data, copy_bytes, d_data, copy_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (err != ACL_SUCCESS) {
            printf("NPU %d: aclrtMemcpy D2H failed\n", dev);
        }
    }

    aclrtFree(d_data);
    aclrtFreeHost(h_data);
}

int main() {
    aclError err = aclInit(nullptr);
    if (err != ACL_SUCCESS) {
        printf("aclInit failed, ret=%d\n", (int)err);
        return -1;
    }

    uint32_t deviceCount = 0;
    err = aclrtGetDeviceCount(&deviceCount);
    if (err != ACL_SUCCESS) {
        printf("aclrtGetDeviceCount failed\n");
        aclFinalize();
        return -1;
    }

    if (deviceCount < NUM_DEVICES) {
        printf("Error: only %u NPU(s) available, need %d\n", deviceCount, NUM_DEVICES);
        aclFinalize();
        return -1;
    }

    std::vector<std::thread> threads(NUM_DEVICES);
    std::vector<ThreadArgs> args(NUM_DEVICES);

    for (int i = 0; i < NUM_DEVICES; i++) {
        args[i].device_id = i;
        threads[i] = std::thread(gpu_worker, &args[i]);
    }

    for (int i = 0; i < NUM_DEVICES; i++) {
        threads[i].join();
    }

    aclFinalize();
    return 0;
}
