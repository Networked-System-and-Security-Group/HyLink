/**
 * AllToAll configurable parameter test (HCCL)
 *
 * HcclAlltoAll: each rank sends `count` float elements to every peer; buffers are
 * layout sendbuff[dst * count + i], recvbuff[src * count + i].
 *
 * Build (in tests/hccl_test/alltoall):
 *   ./build.sh alltoall_param.cc hccl_alltoall_param
 *
 * Run:
 *   ./run.sh ./hccl_alltoall_param
 *   ./run.sh ./hccl_alltoall_param 2 536870912 5
 *   ./run.sh ./hccl_alltoall_param --devices=2 --size=536870912 --iters=3
 *
 * Args:
 *   size bytes for each rank's send and recv buffer.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cmath>
#include <atomic>
#include <pthread.h>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                         \
    do {                                                                                       \
        if (ret != HCCL_SUCCESS) {                                                             \
            printf("hccl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

struct ThreadContext {
    HcclRootInfo* rootInfo;
    uint32_t device;
    uint32_t devCount;
    uint64_t peerCount;  // elements sent to each rank (and recv from each)
    uint32_t numIters;
};

// Value rank `src` puts in the block destined for rank `dst`, element i
static float SendBlockValue(uint32_t src, uint32_t dst, uint32_t iter, uint64_t i, uint32_t devCount) {
    return static_cast<float>(iter * devCount + src + (i % 10000));
}

static const uint32_t WARMUP_ITERS = 6;

static pthread_barrier_t g_barrier;

static int Sample(void* arg) {
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    const uint32_t device = ctx->device;
    const uint32_t devCount = ctx->devCount;
    const uint64_t peerCount = ctx->peerCount;
    const uint32_t numIters = ctx->numIters;

    if (peerCount == 0 || devCount == 0) {
        printf("rank %u: invalid peerCount or devCount\n", device);
        return -1;
    }

    const uint64_t totalCount = devCount * peerCount;
    const size_t bufBytes = totalCount * sizeof(float);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));
    ACLCHECK(aclrtMalloc(&sendBuf, bufBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, bufBytes, ACL_MEM_MALLOC_HUGE_ONLY));

    void* hostSend = nullptr;
    void* hostRecv = nullptr;
    ACLCHECK(aclrtMallocHost(&hostSend, bufBytes));
    ACLCHECK(aclrtMallocHost(&hostRecv, bufBytes));

    HcclComm hcclComm;
    printf("[rank %u] HcclCommInitRootInfo begin\n", device); fflush(stdout);
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));
    printf("[rank %u] HcclCommInitRootInfo end\n", device); fflush(stdout);

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    ACLCHECK(aclrtCreateEventWithFlag(&startEvent, ACL_EVENT_TIME_LINE));
    ACLCHECK(aclrtCreateEventWithFlag(&endEvent, ACL_EVENT_TIME_LINE));

    int failed = 0;
    const double bufSizeGB = bufBytes / (1024.0 * 1024.0 * 1024.0);
    double sumElapsedMs = 0.0;

    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint32_t dst = 0; dst < devCount; ++dst) {
            for (uint64_t i = 0; i < peerCount; ++i) {
                pSend[static_cast<size_t>(dst) * peerCount + i] =
                    SendBlockValue(device, dst, w, i, devCount);
            }
        }
        ACLCHECK(aclrtMemcpy(sendBuf, bufBytes, hostSend, bufBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        HCCLCHECK(HcclAlltoAll(sendBuf, peerCount, HCCL_DATA_TYPE_FP32, recvBuf, peerCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
    }
    if (device == 0) {
        printf("[rank0] AllToAll warmup done, %u rounds\n", WARMUP_ITERS);
    }

    for (uint32_t iter = 0; iter < numIters; ++iter) {
        if (device == 0) {
            printf("[rank0] begin formal %u iters\n", iter);
        }
        float* pSend = static_cast<float*>(hostSend);
        for (uint32_t dst = 0; dst < devCount; ++dst) {
            for (uint64_t i = 0; i < peerCount; ++i) {
                pSend[static_cast<size_t>(dst) * peerCount + i] =
                    SendBlockValue(device, dst, iter, i, devCount);
            }
        }
        ACLCHECK(aclrtMemcpy(sendBuf, bufBytes, hostSend, bufBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        pthread_barrier_wait(&g_barrier);
        ACLCHECK(aclrtRecordEvent(startEvent, stream));
        HCCLCHECK(HcclAlltoAll(sendBuf, peerCount, HCCL_DATA_TYPE_FP32, recvBuf, peerCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtRecordEvent(endEvent, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        ACLCHECK(aclrtSynchronizeEvent(endEvent));

        // per-iter BW: accumulate elapsed time for avg, print suppressed
        float elapsedMs = 0.0f;
        ACLCHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        sumElapsedMs += static_cast<double>(elapsedMs);
        // double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        // double bandwidthGBps = (elapsedSec > 1e-9) ? (bufSizeGB / elapsedSec) : 0.0;
        // if (device == 0) {
        //     printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s\n", iter, elapsedMs, bandwidthGBps);
        // }

        ACLCHECK(aclrtMemcpy(hostRecv, bufBytes, recvBuf, bufBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        float* pRecv = static_cast<float*>(hostRecv);
        for (uint32_t src = 0; src < devCount; ++src) {
            for (uint64_t i = 0; i < peerCount; ++i) {
                float expected = SendBlockValue(src, device, iter, i, devCount);
                float actual = pRecv[static_cast<size_t>(src) * peerCount + i];
                if (std::fabs(actual - expected) > 1e-3f) {
                    failed++;
                }
            }
        }
    }

    if (failed > 0) {
        printf("rank %u: FAILED %d element(s) mismatch\n", device, failed);
    } else {
        printf("rank %u: PASSED AllToAll (devCount=%u peer_count=%lu iters=%u)\n",
               device, devCount, static_cast<unsigned long>(peerCount), numIters);
        if (device == 0 && numIters > 0) {
            double avgElapsedMs = sumElapsedMs / static_cast<double>(numIters);
            double elapsedSec = avgElapsedMs / 1000.0;
            double bandwidthGBps = (elapsedSec > 1e-9) ? (bufSizeGB / elapsedSec) : 0.0;
            printf("Average bandwidth over %u iters (event): %.3f GB/s\n", numIters, bandwidthGBps);
        }
    }

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtDestroyEvent(startEvent));
    ACLCHECK(aclrtDestroyEvent(endEvent));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtFreeHost(hostSend));
    ACLCHECK(aclrtFreeHost(hostRecv));
    return failed > 0 ? -1 : 0;
}

static void PrintUsage(const char* prog) {
    printf("Usage: %s [devices] [size_bytes] [iters]\n", prog);
    printf("   or: %s --devices=N --size=B --iters=I\n", prog);
    printf("  devices     number of NPUs (default 2)\n");
    printf("  size_bytes  bytes for each rank's send/recv buffer (default 4096)\n");
    printf("  iters       number of AllToAll iterations (default 1)\n");
}

int main(int argc, char* argv[]) {
    uint32_t devCount = 2;
    uint64_t sizeBytes = 1024 * sizeof(float);
    uint32_t numIters = 1;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = static_cast<uint32_t>(atoi(argv[i] + 10));
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            sizeBytes = static_cast<uint64_t>(atoll(argv[i] + 7));
        } else if (strncmp(argv[i], "--iters=", 8) == 0) {
            numIters = static_cast<uint32_t>(atoi(argv[i] + 8));
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }
    if (argc >= 2 && argv[1][0] != '-') {
        devCount = static_cast<uint32_t>(atoi(argv[1]));
    }
    if (argc >= 3 && argv[2][0] != '-') {
        sizeBytes = static_cast<uint64_t>(atoll(argv[2]));
    }
    if (argc >= 4 && argv[3][0] != '-') {
        numIters = static_cast<uint32_t>(atoi(argv[3]));
    }

    if (devCount == 0 || sizeBytes == 0 || numIters == 0 ||
        sizeBytes % sizeof(float) != 0 || (sizeBytes / sizeof(float)) % devCount != 0) {
        printf("Invalid parameters: devices=%u size=%lu iters=%u (size must be positive and divisible by 4 * devices)\n",
               devCount, (unsigned long)sizeBytes, numIters);
        PrintUsage(argv[0]);
        return -1;
    }
    const uint64_t totalCount = sizeBytes / sizeof(float);
    const uint64_t peerCount = totalCount / devCount;

    ACLCHECK(aclInit(nullptr));
    uint32_t availableDevs = 0;
    ACLCHECK(aclrtGetDeviceCount(&availableDevs));
    if (devCount > availableDevs) {
        printf("Requested %u devices but only %u available\n", devCount, availableDevs);
        aclFinalize();
        return -1;
    }

    printf("AllToAll param test: devices=%u size=%lu bytes total_count=%lu peer_count=%lu iters=%u (%.2f MB send/recv per rank)\n",
           devCount, (unsigned long)sizeBytes, static_cast<unsigned long>(totalCount),
           static_cast<unsigned long>(peerCount), numIters,
           sizeBytes / (1024.0 * 1024.0));

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void* rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo* rootInfo = static_cast<HcclRootInfo*>(rootInfoBuf);
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    pthread_barrier_init(&g_barrier, nullptr, devCount);

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    std::atomic<int> anyFailed{0};
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        args[i].peerCount = peerCount;
        args[i].numIters = numIters;
        threads[i] = std::thread([&args, &anyFailed](size_t idx) {
            int r = Sample(static_cast<void*>(&args[idx]));
            if (r != 0) {
                anyFailed.store(1);
            }
        }, i);
    }
    for (auto& t : threads) {
        t.join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());
    pthread_barrier_destroy(&g_barrier);

    if (anyFailed.load()) {
        printf("AllToAll Test FAILED.\n");
        return 1;
    }
    printf("AllToAll Test PASSED.\n");
    return 0;
}
