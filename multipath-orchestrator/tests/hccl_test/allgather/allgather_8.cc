#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t sendCount = 1U;
    uint64_t recvCount = ctx->devCount;
    size_t sendSize = sendCount * sizeof(float);
    size_t recvSize = recvCount * sizeof(float);

    // Set device for current thread
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // Allocate device memory for collective
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // Allocate host memory for input, initialize with device ID
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    float *tmpHostBuff = static_cast<float *>(hostBuf);
    for (uint64_t i = 0; i < sendCount; ++i) {
        tmpHostBuff[i] = static_cast<float>(device);
    }
    // Copy host input to device
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // Free host memory
    ACLCHECK(aclrtFreeHost(hostBuf));

    // Initialize collective communicator
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // Create stream
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // Execute AllGather: concatenate all ranks' sendBuf by rank_id, send result to all ranks' recvBuf
    HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    // Block until collective completes
    ACLCHECK(aclrtSynchronizeStream(stream));

    // Copy collective result from device to host and print
    std::this_thread::sleep_for(std::chrono::seconds(ctx->device));
    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, recvSize));
    ACLCHECK(aclrtMemcpy(resultBuff, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float *tmpResBuff = static_cast<float *>(resultBuff);
    std::cout << "rankId: " << ctx->device << ", output: [";
    for (uint32_t i = 0; i < recvCount; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // Cleanup
    HCCLCHECK(HcclCommDestroy(hcclComm));  // Destroy communicator
    ACLCHECK(aclrtFree(sendBuf));          // Free device memory
    ACLCHECK(aclrtFree(recvBuf));          // Free device memory
    ACLCHECK(aclrtDestroyStream(stream));  // Destroy stream
    return 0;
}

int main()
{
    // Initialize device resources
    ACLCHECK(aclInit(NULL));
    // Query device count
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // Generate root info, all threads share same RootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // Launch threads for collective
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    // Cleanup
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // Free host memory
    ACLCHECK(aclFinalize());               // Device finalize
    return 0;
}
