/**
 * NCCL ReduceScatter configurable parameter test
 * HCCL counterpart: tests/hccl_test/reducescatter/reducescatter_param.cc
 * Build: ./build.sh reducescatter_param.cc nccl_reducescatter_param
 */

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cmath>
#include <atomic>

#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd) \
    do { \
        cudaError_t e = (cmd); \
        if (e != cudaSuccess) { \
            printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            return e; \
        } \
    } while (0)

#define NCCLCHECK(cmd) \
    do { \
        ncclResult_t r = (cmd); \
        if (r != ncclSuccess) { \
            printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            return r; \
        } \
    } while (0)

struct ThreadContext {
    int device;
    int devCount;
    uint64_t recvCount;
    uint32_t numIters;
    ncclUniqueId uniqueId;
};

static float SendValue(int ranks, int rank, uint32_t iter, uint64_t g) {
    return 100.0f * static_cast<float>(ranks) +
           10.0f * static_cast<float>(rank) +
           static_cast<float>(iter % 10) +
           0.001f * static_cast<float>(g % 1000);
}

static float ExpectedReducedValue(int devCount, uint32_t iter, uint64_t g) {
    float sum = 0.0f;
    for (int k = 0; k < devCount; ++k) {
        sum += SendValue(devCount, k, iter, g);
    }
    return sum;
}

static const uint32_t WARMUP_ITERS = 3;

static int Sample(void* arg) {
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    const int device = ctx->device;
    const int devCount = ctx->devCount;
    const uint64_t recvCount = ctx->recvCount;
    const uint32_t numIters = ctx->numIters;

    const uint64_t totalCount = static_cast<uint64_t>(devCount) * recvCount;
    const size_t sendSize = totalCount * sizeof(float);
    const size_t recvSize = recvCount * sizeof(float);

    CUDACHECK(cudaSetDevice(device));
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, devCount, ctx->uniqueId, device));

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    CUDACHECK(cudaMalloc(&sendBuf, sendSize));
    CUDACHECK(cudaMalloc(&recvBuf, recvSize));

    void* hostSend = nullptr;
    void* hostRecv = nullptr;
    CUDACHECK(cudaMallocHost(&hostSend, sendSize));
    CUDACHECK(cudaMallocHost(&hostRecv, recvSize));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&endEvent));

    int failed = 0;
    double sumBandwidthGBps = 0.0;
    const double recvSizeGB = static_cast<double>(recvSize) / (1024.0 * 1024.0 * 1024.0);
    const uint32_t measureIters = numIters;

    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t g = 0; g < totalCount; ++g) {
            pSend[g] = SendValue(devCount, device, w, g);
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
        NCCLCHECK(ncclReduceScatter(sendBuf, recvBuf, recvCount, ncclFloat32, ncclSum, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    if (device == 0) {
        printf("Warmup done (%u iters)\n", WARMUP_ITERS);
    }

    for (uint32_t iter = 0; iter < numIters; ++iter) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t g = 0; g < totalCount; ++g) {
            pSend[g] = SendValue(devCount, device, iter, g);
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
        CUDACHECK(cudaEventRecord(startEvent, stream));
        NCCLCHECK(ncclReduceScatter(sendBuf, recvBuf, recvCount, ncclFloat32, ncclSum, comm, stream));
        CUDACHECK(cudaEventRecord(endEvent, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        float elapsedMs = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));
        double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (recvSizeGB / elapsedSec) : 0.0;
        if (device == 0) {
            printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s\n", iter, elapsedMs, bandwidthGBps);
            sumBandwidthGBps += bandwidthGBps;
        }

        CUDACHECK(cudaMemcpy(hostRecv, recvBuf, recvSize, cudaMemcpyDeviceToHost));
        float* pRecv = static_cast<float*>(hostRecv);
        const uint64_t baseIndex = static_cast<uint64_t>(device) * recvCount;
        for (uint64_t i = 0; i < recvCount; ++i) {
            uint64_t g = baseIndex + i;
            float expected = ExpectedReducedValue(devCount, iter, g);
            if (std::fabs(pRecv[i] - expected) > 1e-2f) {
                failed++;
            }
        }
    }

    if (failed > 0) {
        printf("rank %d: FAILED %d element(s) mismatch\n", device, failed);
    } else {
        printf("rank %d: PASSED (devCount=%d recvCount=%lu iters=%u)\n",
               device, devCount, static_cast<unsigned long>(recvCount), numIters);
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
    printf("  size_bytes  bytes for each rank's send buffer (default 4096)\n");
    printf("  iters       number of ReduceScatter iterations (default 1)\n");
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

    if (devCount <= 0 || sizeBytes == 0 || numIters == 0 ||
        sizeBytes % sizeof(float) != 0 || (sizeBytes / sizeof(float)) % devCount != 0) {
        printf("Invalid parameters: devices=%d size=%lu iters=%u (size must be positive and divisible by 4 * devices)\n",
               devCount, static_cast<unsigned long>(sizeBytes), numIters);
        PrintUsage(argv[0]);
        return -1;
    }
    const uint64_t totalCount = sizeBytes / sizeof(float);
    const uint64_t recvCount = totalCount / devCount;

    int availableDevs = 0;
    CUDACHECK(cudaGetDeviceCount(&availableDevs));
    if (devCount > availableDevs) {
        printf("Requested %d devices but only %d available\n", devCount, availableDevs);
        return -1;
    }

    printf("NCCL ReduceScatter param test: devices=%d size=%lu bytes send_count=%lu recv_count=%lu iters=%u (recv %.2f MB, sendBuf %.2f MB per rank)\n",
           devCount, static_cast<unsigned long>(sizeBytes),
           static_cast<unsigned long>(totalCount), static_cast<unsigned long>(recvCount),
           numIters, recvCount * sizeof(float) / (1024.0 * 1024.0),
           sizeBytes / (1024.0 * 1024.0));

    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    std::atomic<int> anyFailed{0};
    for (int i = 0; i < devCount; ++i) {
        args[i].device = i;
        args[i].devCount = devCount;
        args[i].recvCount = recvCount;
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
