/**
 * NCCL AllToAll configurable parameter test
 *
 * Corresponding HCCL: tests/hccl_test/alltoall/alltoall_param.cc
 *
 * Requires NCCL with ncclAlltoAll (2.28+).
 *
 * Build:
 *   ./build.sh alltoall_param.cc nccl_alltoall_param
 *
 * Run:
 *   ./run.sh ./nccl_alltoall_param 2 536870912 5
 *   ./run.sh ./nccl_alltoall_param --devices=2 --size=536870912 --iters=5
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
    uint64_t peerCount;
    uint32_t numIters;
    ncclUniqueId uniqueId;
};

static float SendBlockValue(int ranks, int src, int dst, uint32_t iter, uint64_t i) {
    return 1000.0f * static_cast<float>(ranks) +
           100.0f * static_cast<float>(src) +
           10.0f * static_cast<float>(dst) +
           static_cast<float>(iter % 10) +
           0.001f * static_cast<float>(i % 1000);
}

static const uint32_t WARMUP_ITERS = 3;

static int Sample(void* arg) {
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    const int device = ctx->device;
    const int devCount = ctx->devCount;
    const uint64_t peerCount = ctx->peerCount;
    const uint32_t numIters = ctx->numIters;

    const uint64_t totalCount = static_cast<uint64_t>(devCount) * peerCount;
    const size_t bufBytes = totalCount * sizeof(float);

    CUDACHECK(cudaSetDevice(device));
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, devCount, ctx->uniqueId, device));

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    CUDACHECK(cudaMalloc(&sendBuf, bufBytes));
    CUDACHECK(cudaMalloc(&recvBuf, bufBytes));

    void* hostSend = nullptr;
    void* hostRecv = nullptr;
    CUDACHECK(cudaMallocHost(&hostSend, bufBytes));
    CUDACHECK(cudaMallocHost(&hostRecv, bufBytes));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&endEvent));

    int failed = 0;
    double sumBandwidthGBps = 0.0;
    const double bufSizeGB = static_cast<double>(bufBytes) / (1024.0 * 1024.0 * 1024.0);
    const uint32_t measureIters = numIters;

    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (int dst = 0; dst < devCount; ++dst) {
            for (uint64_t i = 0; i < peerCount; ++i) {
                pSend[static_cast<size_t>(dst) * peerCount + i] =
                    SendBlockValue(devCount, device, dst, w, i);
            }
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, bufBytes, cudaMemcpyHostToDevice));
        NCCLCHECK(ncclAlltoAll(sendBuf, recvBuf, peerCount, ncclFloat32, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    if (device == 0) {
        printf("Warmup done (%u iters)\n", WARMUP_ITERS);
    }

    for (uint32_t iter = 0; iter < numIters; ++iter) {
        float* pSend = static_cast<float*>(hostSend);
        for (int dst = 0; dst < devCount; ++dst) {
            for (uint64_t i = 0; i < peerCount; ++i) {
                pSend[static_cast<size_t>(dst) * peerCount + i] =
                    SendBlockValue(devCount, device, dst, iter, i);
            }
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, bufBytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaEventRecord(startEvent, stream));
        NCCLCHECK(ncclAlltoAll(sendBuf, recvBuf, peerCount, ncclFloat32, comm, stream));
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

        CUDACHECK(cudaMemcpy(hostRecv, recvBuf, bufBytes, cudaMemcpyDeviceToHost));
        float* pRecv = static_cast<float*>(hostRecv);
        for (int src = 0; src < devCount; ++src) {
            for (uint64_t i = 0; i < peerCount; ++i) {
                float expected = SendBlockValue(devCount, src, device, iter, i);
                float actual = pRecv[static_cast<size_t>(src) * peerCount + i];
                if (std::fabs(actual - expected) > 1e-3f) {
                    failed++;
                }
            }
        }
    }

    if (failed > 0) {
        printf("rank %d: FAILED %d element(s) mismatch\n", device, failed);
    } else {
        printf("rank %d: PASSED AllToAll (devCount=%d peer_count=%lu iters=%u)\n",
               device, devCount, static_cast<unsigned long>(peerCount), numIters);
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
    printf("  iters       number of AllToAll iterations (default 1)\n");
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
    const uint64_t peerCount = totalCount / devCount;

    int availableDevs = 0;
    CUDACHECK(cudaGetDeviceCount(&availableDevs));
    if (devCount > availableDevs) {
        printf("Requested %d devices but only %d available\n", devCount, availableDevs);
        return -1;
    }

    printf("NCCL AllToAll param test: devices=%d size=%lu bytes total_count=%lu peer_count=%lu iters=%u (%.2f MB send/recv per rank)\n",
           devCount, static_cast<unsigned long>(sizeBytes),
           static_cast<unsigned long>(totalCount), static_cast<unsigned long>(peerCount),
           numIters, sizeBytes / (1024.0 * 1024.0));

    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    std::atomic<int> anyFailed{0};
    for (int i = 0; i < devCount; ++i) {
        args[i].device = i;
        args[i].devCount = devCount;
        args[i].peerCount = peerCount;
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
