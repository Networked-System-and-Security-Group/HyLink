/**
 * ReduceScatter configurable parameter test
 *
 * Features:
 *   - NPU count, send buffer size per rank in bytes, iteration count
 *   - Multiple ReduceScatter rounds with correctness verification
 *   - main returns 1 if any rank fails verification
 *
 * Build (in tests/hccl_test/reducescatter):
 *   ./build.sh reducescatter_param.cc hccl_reducescatter_param
 *
 * Run (with AMP-CCL hook):
 *   ./run.sh ./hccl_reducescatter_param
 *   ./run.sh ./hccl_reducescatter_param 2 536870912 10
 *   ./run.sh ./hccl_reducescatter_param --devices=2 --size=536870912 --iters=5
 *
 * Args:
 *   devices     NPU count for ReduceScatter, default 2
 *   size        bytes for each rank's send buffer, default 4096
 *   iters       ReduceScatter iterations, default 1
 *
 * Layout:
 *   - ReduceScatter 定长版本：每个 rank 最终收到 recv_count 个元素。
 *   - HcclReduceScatter 接口：
 *       sendBuf 大小 = devCount * recvCount 元素
 *       recvBuf 大小 = recvCount 元素
 *   - 每个 rank 的 sendBuf 视为一个长度为 devCount * recvCount 的向量：
 *       全局下标 g ∈ [0, devCount * recvCount)
 *   - ReduceScatter 的语义：
 *       先对所有 rank 的 sendBuf 逐元素规约（这里用 SUM），得到一个长度为 devCount * recvCount 的结果向量，
 *       然后把它切成 devCount 段：
 *         rank r 拿结果向量中 [r * recv_count, (r+1) * recv_count) 这一段。
 *
 * Correctness:
 *   每个 rank 在 sendBuf 里填充：
 *     value(rank, iter, g) = rank * 1e6 + iter * 1e3 + g
 *   规约算子是 SUM，所以规约后在下标 g 的值应该是：
 *     sum_{k=0}^{devCount-1} (k * 1e6 + iter * 1e3 + g)
 *   rank r 的 recvBuf[i] 对应全局下标：
 *     g = r * recv_count + i
 */

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
     uint64_t recvCount;  // elements per rank in recvBuf
     uint32_t numIters;   // ReduceScatter iterations
 };

 // 规约后，全局下标 g 的期望值（sum over all ranks）
 static float ExpectedReducedValue(uint32_t devCount, uint32_t iter, uint64_t g)
 {
     double sum = 0.0;
     for (uint32_t k = 0; k < devCount; ++k) {
         sum += static_cast<double>(iter * devCount + k + (g % 10000));
     }
     return static_cast<float>(sum);
 }

 static const uint32_t WARMUP_ITERS = 6;

static pthread_barrier_t g_barrier;

 int Sample(void* arg)
 {
     ThreadContext* ctx = static_cast<ThreadContext*>(arg);
     const uint32_t device = ctx->device;
     const uint32_t devCount = ctx->devCount;
     const uint64_t recvCount = ctx->recvCount;
     const uint32_t numIters = ctx->numIters;

     if (recvCount == 0 || devCount == 0) {
         printf("rank %u: invalid config, recvCount=%lu devCount=%u\n",
                device, (unsigned long)recvCount, devCount);
         return -1;
     }

     const uint64_t totalCount = devCount * recvCount; // sendBuf length per rank
     const size_t sendSize = totalCount * sizeof(float);
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

     // timing events
     aclrtEvent startEvent = nullptr;
     aclrtEvent endEvent = nullptr;
     ACLCHECK(aclrtCreateEventWithFlag(&startEvent, ACL_EVENT_TIME_LINE));
     ACLCHECK(aclrtCreateEventWithFlag(&endEvent, ACL_EVENT_TIME_LINE));

    int failed = 0;
    const double sendSizeGB = sendSize / (1024.0 * 1024.0 * 1024.0);
    double sumElapsedMs = 0.0;

    // warmup (not timed)
     for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t g = 0; g < totalCount; ++g) {
             pSend[g] = static_cast<float>(w * devCount + device + (g % 10000));
         }
         ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
         HCCLCHECK(HcclReduceScatter(sendBuf,
                                     recvBuf,
                                     recvCount,
                                     HCCL_DATA_TYPE_FP32,
                                     HCCL_REDUCE_SUM,
                                     hcclComm,
                                     stream));
         ACLCHECK(aclrtSynchronizeStream(stream));
     }
     if (device == 0) {
         printf("[rank0] ReduceScatter warmup done, %u iters\n", WARMUP_ITERS);
     }

     // formal iters
     for (uint32_t iter = 0; iter < numIters; ++iter) {
         if (device == 0) {
             printf("[rank0] begin formal %u iters\n", iter);
         }
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t g = 0; g < totalCount; ++g) {
             pSend[g] = static_cast<float>(iter * devCount + device + (g % 10000));
         }

         ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
         pthread_barrier_wait(&g_barrier);
         ACLCHECK(aclrtRecordEvent(startEvent, stream));
         HCCLCHECK(HcclReduceScatter(sendBuf,
                                     recvBuf,
                                     recvCount,
                                     HCCL_DATA_TYPE_FP32,
                                     HCCL_REDUCE_SUM,
                                     hcclComm,
                                     stream));
         ACLCHECK(aclrtRecordEvent(endEvent, stream));
         ACLCHECK(aclrtSynchronizeStream(stream));
         ACLCHECK(aclrtSynchronizeEvent(endEvent));

         float elapsedMs = 0.0f;
         ACLCHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
         sumElapsedMs += static_cast<double>(elapsedMs);

         ACLCHECK(aclrtMemcpy(hostRecv, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
         float* pRecv = static_cast<float*>(hostRecv);

         // rank r owns global indices [r * recvCount, (r + 1) * recvCount)
         const uint64_t baseIndex = static_cast<uint64_t>(device) * recvCount;
         for (uint64_t i = 0; i < recvCount; ++i) {
             uint64_t g = baseIndex + i;
             float expected = ExpectedReducedValue(devCount, iter, g);
             float actual = pRecv[i];
             if (std::fabs(actual - expected) > 1e-2f) {
                 if (failed < 10) {
                     printf("rank %u iter %u: first errors at local idx %lu (global %lu), expected %.2f actual %.2f\n",
                            device, iter, (unsigned long)i, (unsigned long)g, expected, actual);
                 }
                 failed++;
             }
         }
     }

    if (failed > 0) {
        printf("rank %u: FAILED %d element(s) mismatch in ReduceScatter correctness check\n", device, failed);
    } else {
        printf("rank %u: PASSED ReduceScatter (devCount=%u recvCount=%lu iters=%u)\n",
               device, devCount, (unsigned long)recvCount, numIters);
        if (device == 0 && numIters > 0) {
            double avgElapsedMs = sumElapsedMs / static_cast<double>(numIters);
            double elapsedSec = avgElapsedMs / 1000.0;
            double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;
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

 static void PrintUsage(const char* prog)
 {
     printf("Usage: %s [devices] [size_bytes] [iters]\n", prog);
     printf("   or: %s --devices=N --size=B --iters=I\n", prog);
     printf("  devices     number of NPUs (default 2)\n");
     printf("  size_bytes  bytes for each rank's send buffer (default 4096)\n");
     printf("  iters       number of ReduceScatter iterations (default 1)\n");
 }

 int main(int argc, char* argv[])
 {
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
     const uint64_t recvCount = totalCount / devCount;

     ACLCHECK(aclInit(nullptr));
     uint32_t availableDevs = 0;
     ACLCHECK(aclrtGetDeviceCount(&availableDevs));
     if (devCount > availableDevs) {
         printf("Requested %u devices but only %u available\n", devCount, availableDevs);
         aclFinalize();
         return -1;
     }

     printf("ReduceScatter param test: devices=%u size=%lu bytes send_count=%lu recv_count=%lu iters=%u (recv %.2f MB, sendBuf %.2f MB per rank)\n",
            devCount, (unsigned long)sizeBytes, (unsigned long)totalCount,
            (unsigned long)recvCount, numIters,
            recvCount * sizeof(float) / (1024.0 * 1024.0),
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
         args[i].recvCount = recvCount;
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
         printf("ReduceScatter Test FAILED (at least one rank reported mismatch).\n");
         return 1;
     }
     printf("ReduceScatter Test PASSED (all ranks correctness check ok).\n");
     return 0;
 }
