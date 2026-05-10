/**
 * NCCL AllReduce configurable parameter test
 *
 * Corresponding HCCL: tests/hccl_test/allreduce/allreduce_param.cc
 *
 * Build:
 *   ./build.sh allreduce_param.cc nccl_allreduce_param
 *
 * Run:
 *   ./run.sh ./nccl_allreduce_param
 *   ./run.sh ./nccl_allreduce_param 2 536870912 10
 *   ./run.sh ./nccl_allreduce_param --devices=2 --size=536870912 --iters=10
 */

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cmath>
#include <atomic>

#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd)                                                                 \
    do {                                                                               \
        cudaError_t e = (cmd);                                                         \
        if (e != cudaSuccess) {                                                        \
            printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            return e;                                                                  \
        }                                                                              \
    } while (0)

#define NCCLCHECK(cmd)                                                                 \
    do {                                                                               \
        ncclResult_t r = (cmd);                                                        \
        if (r != ncclSuccess) {                                                        \
            printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            return r;                                                                  \
        }                                                                              \
    } while (0)

struct ThreadContext {
    int device;
    int devCount;
    uint64_t sendCount;
    uint32_t numIters;
    ncclUniqueId uniqueId;
};

static float ExpectedSendValue(int ranks, int rank, uint32_t iter, uint64_t i) {
    return 100.0f * static_cast<float>(ranks) +
           10.0f * static_cast<float>(rank) +
           static_cast<float>(iter % 10) +
           0.001f * static_cast<float>(i % 1000);
}

static float ExpectedRecvValue(int devCount, uint32_t iter, uint64_t i) {
    float sum = 0.0f;
    for (int r = 0; r < devCount; ++r) {
        sum += ExpectedSendValue(devCount, r, iter, i);
    }
    return sum;
}

static const uint32_t WARMUP_ITERS = 3;

static int Sample(void* arg) {
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    const int device = ctx->device;
    const int devCount = ctx->devCount;
    const uint64_t sendCount = ctx->sendCount;
    const uint32_t numIters = ctx->numIters;

    const size_t bufSize = sendCount * sizeof(float);
    CUDACHECK(cudaSetDevice(device));

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, devCount, ctx->uniqueId, device));

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    CUDACHECK(cudaMalloc(&sendBuf, bufSize));
    CUDACHECK(cudaMalloc(&recvBuf, bufSize));

    void* hostSend = nullptr;
    void* hostRecv = nullptr;
    CUDACHECK(cudaMallocHost(&hostSend, bufSize));
    CUDACHECK(cudaMallocHost(&hostRecv, bufSize));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&endEvent));

    int failed = 0;
    double sumBandwidthGBps = 0.0;
    const double bufSizeGB = static_cast<double>(bufSize) / (1024.0 * 1024.0 * 1024.0);
    const uint32_t measureIters = numIters;

    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedSendValue(devCount, device, w, i);
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, bufSize, cudaMemcpyHostToDevice));
        NCCLCHECK(ncclAllReduce(sendBuf, recvBuf, sendCount, ncclFloat32, ncclSum, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    if (device == 0) {
        printf("Warmup done (%u iters)\n", WARMUP_ITERS);
    }

    for (uint32_t iter = 0; iter < numIters; ++iter) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedSendValue(devCount, device, iter, i);
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, bufSize, cudaMemcpyHostToDevice));
        CUDACHECK(cudaEventRecord(startEvent, stream));
        NCCLCHECK(ncclAllReduce(sendBuf, recvBuf, sendCount, ncclFloat32, ncclSum, comm, stream));
        CUDACHECK(cudaEventRecord(endEvent, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        float elapsedMs = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));
        double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (bufSizeGB / elapsedSec) : 0.0;
        if (device == 0) {
            printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s\n", iter, elapsedMs, bandwidthGBps);
            sumBandwidthGBps += bandwidthGBps;
        }

        CUDACHECK(cudaMemcpy(hostRecv, recvBuf, bufSize, cudaMemcpyDeviceToHost));
        float* pRecv = static_cast<float*>(hostRecv);
        for (uint64_t i = 0; i < sendCount; ++i) {
            float expected = ExpectedRecvValue(devCount, iter, i);
            if (std::fabs(pRecv[i] - expected) > 1e-2f) {
                failed++;
            }
        }
    }

    if (failed > 0) {
        printf("rank %d: FAILED %d element(s) mismatch\n", device, failed);
    } else {
        printf("rank %d: PASSED (devCount=%d sendCount=%lu iters=%u)\n",
               device, devCount, static_cast<unsigned long>(sendCount), numIters);
        if (device == 0 && measureIters > 0) {
            double avgBandwidth = sumBandwidthGBps / static_cast<double>(measureIters);
            printf("Average bandwidth over %u iters (event): %.3f GB/s\n", measureIters, avgBandwidth);
        }
    }

    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(endEvent));
    CUDACHECK(cudaFree(sendBuf));
    CUDACHECK(cudaFree(recvBuf));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFreeHost(hostSend));
    CUDACHECK(cudaFreeHost(hostRecv));
    return failed > 0 ? -1 : 0;
}

static void PrintUsage(const char* prog) {
    printf("Usage: %s [devices] [size_bytes] [iters]\n", prog);
    printf("   or: %s --devices=N --size=B --iters=I\n", prog);
    printf("  devices     number of GPUs (default 2)\n");
    printf("  size_bytes  bytes for each rank's send/recv buffer (default 4096)\n");
    printf("  iters       number of AllReduce iterations (default 1)\n");
}

int main(int argc, char* argv[]) {
    int devCount = 2;
    uint64_t sizeBytes = 1024 * sizeof(float);
    uint32_t numIters = 1;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = atoi(argv[i] + 10);
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
        devCount = atoi(argv[1]);
    }
    if (argc >= 3 && argv[2][0] != '-') {
        sizeBytes = static_cast<uint64_t>(atoll(argv[2]));
    }
    if (argc >= 4 && argv[3][0] != '-') {
        numIters = static_cast<uint32_t>(atoi(argv[3]));
    }

    if (devCount <= 0 || sizeBytes == 0 || numIters == 0 || sizeBytes % sizeof(float) != 0) {
        printf("Invalid parameters: devices=%d size=%lu iters=%u (size must be positive and 4-byte aligned)\n",
               devCount, static_cast<unsigned long>(sizeBytes), numIters);
        PrintUsage(argv[0]);
        return -1;
    }
    const uint64_t sendCount = sizeBytes / sizeof(float);

    int availableDevs = 0;
    CUDACHECK(cudaGetDeviceCount(&availableDevs));
    if (devCount > availableDevs) {
        printf("Requested %d devices but only %d available\n", devCount, availableDevs);
        return -1;
    }

    printf("NCCL AllReduce param test: devices=%d size=%lu bytes send_count=%lu iters=%u (%.2f MB send/recv per rank)\n",
           devCount, static_cast<unsigned long>(sizeBytes),
           static_cast<unsigned long>(sendCount), numIters,
           sizeBytes / (1024.0 * 1024.0));

    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    std::atomic<int> anyFailed{0};
    for (int i = 0; i < devCount; ++i) {
        args[i].device = i;
        args[i].devCount = devCount;
        args[i].sendCount = sendCount;
        args[i].numIters = numIters;
        args[i].uniqueId = uniqueId;
        threads[i] = std::thread([&args, &anyFailed](size_t idx) {
            int r = Sample(static_cast<void*>(&args[idx]));
            if (r != 0) {
                anyFailed.store(1);
            }
        }, static_cast<size_t>(i));
    }
    for (auto& t : threads) {
        t.join();
    }

    if (anyFailed.load()) {
        printf("Test FAILED.\n");
        return 1;
    }
    printf("Test PASSED.\n");
    return 0;
}
