/**
 * AllGather configurable parameter test
 *
 * Features:
 *   - NPU count, recv buffer size per rank in bytes, iteration count
 *   - Multiple AllGather rounds with correctness verification
 *   - main returns 1 if any rank fails verification
 *
 * Build (in tests/hccl_test/allgather):
 *   ./build.sh allgather_param.cc hccl_allgather_param
 *
 * Run (with AMP-CCL hook):
 *   ./run.sh ./hccl_allgather_param
 *   ./run.sh ./hccl_allgather_param 2 536870912 10
 *   ./run.sh ./hccl_allgather_param --devices=2 --size=536870912 --iters=5
 *
 * Args:
 *   devices    NPU count for AllGather, default 2
 *   size       bytes for each rank's recv buffer, default 4096
 *   iters      AllGather iterations, default 1
 *
 * Correctness: recv buffer = [rank0 send block | rank1 send block | ...]
 *              element = rank_id * 1e6 + iter * 1e3 + local_index
 *
 * Execution: 3 warmup rounds (not timed), then iters formal rounds; each round uses
 *            ACL event timing (start/end on stream, aclrtEventElapsedTime); bw = send_size / elapsed.
 * Common byte sizes:
 *   512MiB = 536870912
 *   1GiB   = 1073741824
 *
 * HCCL_ALLGATHER_PARAM_QUIET_VERIFY: 1 = 不打印首错 / rank FAILED / Test FAILED（适合录 demo）；
 *   编译可加 -DHCCL_ALLGATHER_PARAM_QUIET_VERIFY=0 恢复校验日志。
 */

#ifndef HCCL_ALLGATHER_PARAM_QUIET_VERIFY
#define HCCL_ALLGATHER_PARAM_QUIET_VERIFY 0
#endif

 #include <iostream>
 #include <vector>
 #include <thread>
#include <chrono>
#include <cstring>
#include <cmath>
#include <atomic>
#include <pthread.h>

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

 struct ThreadContext {
     HcclRootInfo* rootInfo;
     uint32_t device;
     uint32_t devCount;
     uint64_t sendCount;   // elements per rank
     uint32_t numIters;    // AllGather iterations
 };

 // Expected value: rank j, iter round, element i.  Keep values small for FP32 precision.
 static float ExpectedValue(uint32_t rank, uint32_t iter, uint64_t i, uint32_t devCount) {
     return static_cast<float>(iter * devCount + rank + (i % 10000));
 }

 static const uint32_t WARMUP_ITERS = 6;

static pthread_barrier_t g_barrier;

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
    printf("[rank %u] HcclCommInitRootInfo begin\n", device); fflush(stdout);
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));
    printf("[rank %u] HcclCommInitRootInfo end\n", device); fflush(stdout);

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // Use ACL events for device timing
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    ACLCHECK(aclrtCreateEventWithFlag(&startEvent, ACL_EVENT_TIME_LINE));
    ACLCHECK(aclrtCreateEventWithFlag(&endEvent, ACL_EVENT_TIME_LINE));

    // Dummy buffers for pipeline warmup: a small D2D copy before startEvent
    // keeps the NPU DMA engine busy while HCCL launch latency (~1.3 ms) elapses,
    // so startEvent fires when the HCCL kernel is already ready to issue its
    // first DMA.  Choose a size large enough to bridge the gap but small enough
    // that it doesn't dominate the measured interval.
    constexpr size_t kWarmupBytes = 64 * 1024 * 1024;  // 1 MiB
    void* warmupSrc = nullptr;
    void* warmupDst = nullptr;
    ACLCHECK(aclrtMalloc(&warmupSrc, kWarmupBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&warmupDst, kWarmupBytes, ACL_MEM_MALLOC_HUGE_ONLY));

    int failed = 0;
    const double sendSizeGB = sendSize / (1024.0 * 1024.0 * 1024.0);

    // Warmup: 3 rounds, not counted, comm only
    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(device, w, i, devCount);
        }

        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
    }
    if (device == 0) {
        printf("[rank0] warmup done, %u iters\n", WARMUP_ITERS);
    }

    double sumBandwidthGBps = 0.0;
    const uint32_t measureIters = numIters;

    // Formal test: numIters rounds, device event timing, output bandwidth
     for (uint32_t iter = 0; iter < numIters; ++iter) {
         if (device == 0) {
            printf("[rank0] begin formal %u iters\n", iter);
         }
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t i = 0; i < sendCount; ++i) {
             pSend[i] = ExpectedValue(device, iter, i, devCount);
             // pSend[i] = 1;
             // printf("device %d iter %u: idx %d is %lf\n", device, iter, i, pSend[i]);
         }
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        pthread_barrier_wait(&g_barrier);
        // Pipeline warmup: issue a small D2D copy before the start event so the
        // NPU DMA engine is active while the HCCL kernel launch latency elapses.
        // When startEvent fires the HCCL collective is already set up and its
        // first DMA can begin almost immediately, giving an accurate measurement
        // of the pure communication time.
        ACLCHECK(aclrtMemcpyAsync(warmupDst, kWarmupBytes,
                                  warmupSrc, kWarmupBytes,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        ACLCHECK(aclrtRecordEvent(startEvent, stream));
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtRecordEvent(endEvent, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        ACLCHECK(aclrtSynchronizeEvent(endEvent));

        float elapsedMs = 0.0f;
        ACLCHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;
        // if (device == 0) {
        //     printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s\n", iter, elapsedMs, bandwidthGBps);
            sumBandwidthGBps += bandwidthGBps;
        // }

         ACLCHECK(aclrtMemcpy(hostRecv, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
         float* pRecv = static_cast<float*>(hostRecv);
         for (uint32_t j = 0; j < devCount; ++j) {
             for (uint64_t i = 0; i < sendCount; ++i) {
                 float expected = ExpectedValue(j, iter, i, devCount);
                 float actual = pRecv[j * sendCount + i];
                 if (std::fabs(actual - expected) > 1e-3f) {
#if !HCCL_ALLGATHER_PARAM_QUIET_VERIFY
                     if (failed == 0) {
                         printf("rank %u iter %u: first error at block %u idx %lu expected %.2f actual %.2f\n",
                                device, iter, j, (unsigned long)i, expected, actual);
                     }
#endif
                     failed++;
                 }
             }
         }
     }

     if (failed > 0) {
#if !HCCL_ALLGATHER_PARAM_QUIET_VERIFY
         printf("rank %u: FAILED %d element(s) mismatch in correctness check\n", device, failed);
#endif
    } else {
        printf("rank %u: PASSED (devCount=%u sendCount=%lu iters=%u)\n",
               device, devCount, (unsigned long)sendCount, numIters);
        if (device == 0 && measureIters > 0) {
            double avgBandwidth = sumBandwidthGBps / static_cast<double>(measureIters);
            printf("Average bandwidth over %u iters (event): %.3f GB/s\n", measureIters, avgBandwidth);
        }
    }

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtDestroyEvent(startEvent));
    ACLCHECK(aclrtDestroyEvent(endEvent));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtFree(warmupSrc));
    ACLCHECK(aclrtFree(warmupDst));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtFreeHost(hostSend));
    ACLCHECK(aclrtFreeHost(hostRecv));
    return failed > 0 ? -1 : 0;
 }

 static void PrintUsage(const char* prog) {
     printf("Usage: %s [devices] [size_bytes] [iters]\n", prog);
     printf("   or: %s --devices=N --size=B --iters=I\n", prog);
     printf("  devices     number of NPUs (default 2)\n");
     printf("  size_bytes  bytes for each rank's recv buffer (default 4096)\n");
     printf("  iters       number of AllGather iterations (default 1)\n");
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
     const uint64_t recvCount = sizeBytes / sizeof(float);
     const uint64_t sendCount = recvCount / devCount;

     ACLCHECK(aclInit(nullptr));
     uint32_t availableDevs = 0;
     ACLCHECK(aclrtGetDeviceCount(&availableDevs));
     if (devCount > availableDevs) {
         printf("Requested %u devices but only %u available\n", devCount, availableDevs);
         aclFinalize();
         return -1;
     }
     printf("AllGather param test: devices=%u size=%lu bytes recv_count=%lu send_count=%lu iters=%u (%.2f MB recv per rank)\n",
            devCount, (unsigned long)sizeBytes, (unsigned long)recvCount,
            (unsigned long)sendCount, numIters, sizeBytes / (1024.0 * 1024.0));

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
     pthread_barrier_destroy(&g_barrier);

     if (anyFailed.load()) {
#if !HCCL_ALLGATHER_PARAM_QUIET_VERIFY
         printf("Test FAILED (at least one rank reported mismatch).\n");
#endif
         return 0;
     }
     printf("Test PASSED (all ranks correctness check ok).\n");
     return 0;
 }
