/**
 * AllGather configurable parameter test (CPU wall-clock timing)
 *
 * Features:
 *   - Specify NPU count, send count per rank (element count), iteration count
 *   - Multiple AllGather rounds with correctness verification after each round
 *   - Use HostTimer (CPU wall-clock) for timing, print bandwidth per iter and average bandwidth
 *
 * Build (in tests/hccl_test/allgather):
 *   ./build.sh allgather_param_wall.cc hccl_allgather_param_wall
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <cmath>
#include <atomic>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, ret);  \
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

// Simple CPU wall-clock timer (milliseconds)
class HostTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    void end() { end_time = std::chrono::high_resolution_clock::now(); }
    float getElapsedMs() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0f;
    }
};

struct ThreadContext {
    HcclRootInfo* rootInfo;
    uint32_t device;
    uint32_t devCount;
    uint64_t sendCount;   // elements per rank
    uint32_t numIters;    // AllGather iteration count
};

// Expected value: rank j, iter round, element i
static float ExpectedValue(uint32_t rank, uint32_t iter, uint64_t i) {
    return 1000000 * rank + 1000 * iter + i;
}

static const uint32_t WARMUP_ITERS = 3;

int Sample(void* arg) {
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    const uint32_t device = ctx->device;
    const uint32_t devCount = ctx->devCount;
    const uint64_t sendCount = ctx->sendCount;
    const uint32_t numIters = ctx->numIters;

    const uint64_t recvCount = devCount * sendCount;
    const size_t sendSize = sendCount * sizeof(float);
    const size_t recvSize = recvCount * sizeof(float);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;

    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));

    void* hostSend = nullptr;
    void* hostRecv = nullptr;
    ACLCHECK(aclrtMallocHost(&hostSend, sendSize));
    ACLCHECK(aclrtMallocHost(&hostRecv, recvSize));

    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    HostTimer timer;

    int failed = 0;
    const double sendSizeGB = sendSize / (1024.0 * 1024.0 * 1024.0);

    // Warmup: 3 rounds, not counted, comm only
    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(device, w, i);
        }

        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
    }
    if (device == 0) {
        printf("Warmup done (%u iters)\n", WARMUP_ITERS);
    }

    double sumBandwidthGBps = 0.0;
    const uint32_t measureIters = numIters;

    // Formal test: numIters rounds, CPU wall-clock timing, output bandwidth
    for (uint32_t iter = 0; iter < numIters; ++iter) {
        if (device == 0) {
            printf("normal test on iter %u\n", iter);
        }
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(device, iter, i);
        }
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));

        timer.start();
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        timer.end();

        float elapsedMs = timer.getElapsedMs();
        double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;
        if (device == 0) {
            printf("iter %u: %.3f ms (wall), bandwidth: %.3f GB/s\n", iter, elapsedMs, bandwidthGBps);
            sumBandwidthGBps += bandwidthGBps;
        }

        ACLCHECK(aclrtMemcpy(hostRecv, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
        float* pRecv = static_cast<float*>(hostRecv);
        for (uint32_t j = 0; j < devCount; ++j) {
            for (uint64_t i = 0; i < sendCount; ++i) {
                float expected = ExpectedValue(j, iter, i);
                float actual = pRecv[j * sendCount + i];
                if (std::fabs(actual - expected) > 1e-3f) {
                    failed++;
                }
            }
        }
    }

    if (failed > 0) {
        printf("rank %u: FAILED %d element(s) mismatch in correctness check\n", device, failed);
    } else {
        printf("rank %u: PASSED (devCount=%u sendCount=%lu iters=%u)\n",
               device, devCount, (unsigned long)sendCount, numIters);
        if (device == 0 && measureIters > 0) {
            double avgBandwidth = sumBandwidthGBps / static_cast<double>(measureIters);
            printf("Average bandwidth over %u iters (wall): %.3f GB/s\n", measureIters, avgBandwidth);
        }
    }

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtFreeHost(hostSend));
    ACLCHECK(aclrtFreeHost(hostRecv));
    return failed > 0 ? -1 : 0;
}

static void PrintUsage(const char* prog) {
    printf("Usage: %s [devices] [send_count] [iters]\n", prog);
    printf("   or: %s --devices=N --send-count=K --iters=I\n", prog);
    printf("  devices     number of NPUs (default 2)\n");
    printf("  send_count  float elements per rank (default 1024)\n");
    printf("  iters       number of AllGather iterations (default 1)\n");
}

int main(int argc, char* argv[]) {
    uint32_t devCount = 2;
    uint64_t sendCount = 1024;
    uint32_t numIters = 1;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = static_cast<uint32_t>(atoi(argv[i] + 10));
        } else if (strncmp(argv[i], "--send-count=", 13) == 0) {
            sendCount = static_cast<uint64_t>(atoll(argv[i] + 13));
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
        sendCount = static_cast<uint64_t>(atoll(argv[2]));
    }
    if (argc >= 4 && argv[3][0] != '-') {
        numIters = static_cast<uint32_t>(atoi(argv[3]));
    }

    if (devCount == 0 || sendCount == 0 || numIters == 0) {
        printf("Invalid parameters: devices=%u send_count=%lu iters=%u\n",
               devCount, (unsigned long)sendCount, numIters);
        PrintUsage(argv[0]);
        return -1;
    }

    ACLCHECK(aclInit(nullptr));
    uint32_t availableDevs = 0;
    ACLCHECK(aclrtGetDeviceCount(&availableDevs));
    if (devCount > availableDevs) {
        printf("Requested %u devices but only %u available\n", devCount, availableDevs);
        aclFinalize();
        return -1;
    }
    printf("AllGather param (wall timer) test: devices=%u send_count=%lu iters=%u (%.2f MB per rank per iter)\n",
           devCount, (unsigned long)sendCount, numIters,
           sendCount * sizeof(float) / (1024.0 * 1024.0));

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void* rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo* rootInfo = static_cast<HcclRootInfo*>(rootInfoBuf);
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    std::atomic<int> anyFailed{0};
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        args[i].sendCount = sendCount;
        args[i].numIters = numIters;
        threads[i] = std::thread([&args, &anyFailed](size_t idx) {
            int r = Sample(static_cast<void*>(&args[idx]));
            if (r != 0) anyFailed.store(1);
        }, i);
    }
    for (auto& t : threads) {
        t.join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());

    if (anyFailed.load()) {
        printf("Test FAILED (at least one rank reported mismatch).\n");
        return 1;
    }
    printf("Test PASSED (all ranks correctness check ok).\n");
    return 0;
}

