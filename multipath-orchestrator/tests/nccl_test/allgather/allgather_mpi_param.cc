/**
 * NCCL AllGather MPI multi-process configurable parameter test.
 *
 * Parameters intentionally match allgather_param.cc:
 *   ./allgather_mpi_param [devices] [send_count] [iters]
 *   ./allgather_mpi_param --devices=N --send-count=K --iters=I
 *
 * MPI mode uses one OS process per NCCL rank/GPU. Therefore --devices must
 * equal MPI_COMM_WORLD size for the intended run.
 */

#include <mpi.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

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

static int EnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::atoi(v) : fallback;
}

static int LocalRank(int worldRank) {
    int v = EnvInt("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (v >= 0) return v;
    v = EnvInt("MV2_COMM_WORLD_LOCAL_RANK", -1);
    if (v >= 0) return v;
    v = EnvInt("SLURM_LOCALID", -1);
    if (v >= 0) return v;
    v = EnvInt("PMI_LOCAL_RANK", -1);
    if (v >= 0) return v;
    v = EnvInt("MPI_LOCALRANKID", -1);
    if (v >= 0) return v;
    return worldRank;
}

// Expected value: rank j, iter round, element i
static float ExpectedValue(int rank, uint32_t iter, uint64_t i) {
    return 1000000.0f * static_cast<float>(rank) +
           1000.0f * static_cast<float>(iter) +
           static_cast<float>(i);
}

static const uint32_t WARMUP_ITERS = 3;

static void PrintUsage(const char* prog) {
    printf("Usage: %s [devices] [send_count] [iters]\n", prog);
    printf("   or: %s --devices=N --send-count=K --iters=I\n", prog);
    printf("  devices     number of MPI ranks / GPUs (default 2)\n");
    printf("  send_count  float elements per rank (default 1024)\n");
    printf("  iters       number of AllGather iterations (default 1)\n");
}

static int RunRank(int rank, int nranks, int device, uint64_t sendCount, uint32_t numIters,
                   const ncclUniqueId& uniqueId, double* avgBandwidthGBpsOut) {
    const uint64_t recvCount = static_cast<uint64_t>(nranks) * sendCount;
    const size_t sendSize = sendCount * sizeof(float);
    const size_t recvSize = recvCount * sizeof(float);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;

    CUDACHECK(cudaSetDevice(device));
    printf("rank %d: init nccl comm on cuda device %d\n", rank, device);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nranks, uniqueId, rank));
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
    const double sendSizeGB = static_cast<double>(sendSize) / (1024.0 * 1024.0 * 1024.0);

    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(rank, w, i);
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
        NCCLCHECK(ncclAllGather(sendBuf, recvBuf, sendCount, ncclFloat32, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    if (rank == 0) {
        printf("Warmup done (%u iters)\n", WARMUP_ITERS);
    }

    double sumBandwidthGBps = 0.0;
    const uint32_t measureIters = numIters;

    for (uint32_t iter = 0; iter < numIters; ++iter) {
        if (rank == 0) {
            printf("normal test on iter %u\n", iter);
        }
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(rank, iter, i);
        }
        CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));

        MPI_Barrier(MPI_COMM_WORLD);
        CUDACHECK(cudaEventRecord(startEvent, stream));
        NCCLCHECK(ncclAllGather(sendBuf, recvBuf, sendCount, ncclFloat32, comm, stream));
        CUDACHECK(cudaEventRecord(endEvent, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        float localElapsedMs = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&localElapsedMs, startEvent, endEvent));
        double localElapsedMsD = static_cast<double>(localElapsedMs);
        double maxElapsedMs = 0.0;
        MPI_Allreduce(&localElapsedMsD, &maxElapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double elapsedSec = maxElapsedMs / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;
        if (rank == 0) {
            printf("iter %u: %.3f ms (event max), bandwidth: %.3f GB/s\n",
                   iter, maxElapsedMs, bandwidthGBps);
            sumBandwidthGBps += bandwidthGBps;
        }

        CUDACHECK(cudaMemcpy(hostRecv, recvBuf, recvSize, cudaMemcpyDeviceToHost));
        float* pRecv = static_cast<float*>(hostRecv);
        for (int j = 0; j < nranks; ++j) {
            for (uint64_t i = 0; i < sendCount; ++i) {
                float expected = ExpectedValue(j, iter, i);
                float actual = pRecv[static_cast<size_t>(j) * sendCount + i];
                if (std::fabs(actual - expected) > 1e-3f) {
                    failed++;
                }
            }
        }
    }

    if (failed > 0) {
        printf("rank %d: FAILED %d element(s) mismatch in correctness check\n", rank, failed);
    } else {
        printf("rank %d: PASSED (devCount=%d sendCount=%lu iters=%u)\n",
               rank, nranks, static_cast<unsigned long>(sendCount), numIters);
        if (rank == 0 && measureIters > 0) {
            double avgBandwidth = sumBandwidthGBps / static_cast<double>(measureIters);
            *avgBandwidthGBpsOut = avgBandwidth;
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int devCount = 2;
    uint64_t sendCount = 1024;
    uint32_t numIters = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = std::atoi(argv[i] + 10);
        } else if (std::strncmp(argv[i], "--send-count=", 13) == 0) {
            sendCount = static_cast<uint64_t>(std::atoll(argv[i] + 13));
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            numIters = static_cast<uint32_t>(std::atoi(argv[i] + 8));
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            if (rank == 0) PrintUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    if (argc >= 2 && argv[1][0] != '-') {
        devCount = std::atoi(argv[1]);
    }
    if (argc >= 3 && argv[2][0] != '-') {
        sendCount = static_cast<uint64_t>(std::atoll(argv[2]));
    }
    if (argc >= 4 && argv[3][0] != '-') {
        numIters = static_cast<uint32_t>(std::atoi(argv[3]));
    }

    int invalid = (devCount <= 0 || sendCount == 0 || numIters == 0 || devCount != worldSize) ? 1 : 0;
    if (invalid) {
        if (rank == 0) {
            printf("Invalid parameters: devices=%d world_size=%d send_count=%lu iters=%u\n",
                   devCount, worldSize, static_cast<unsigned long>(sendCount), numIters);
            PrintUsage(argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    int availableDevs = 0;
    CUDACHECK(cudaGetDeviceCount(&availableDevs));
    if (availableDevs <= 0) {
        printf("rank %d: no CUDA device available\n", rank);
        MPI_Finalize();
        return -1;
    }
    int device = LocalRank(rank) % availableDevs;

    if (rank == 0) {
        printf("NCCL AllGather MPI param test: devices=%d send_count=%lu iters=%u (%.2f MB per rank per iter)\n",
               devCount, static_cast<unsigned long>(sendCount), numIters,
               sendCount * sizeof(float) / (1024.0 * 1024.0));
    }

    ncclUniqueId uniqueId;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&uniqueId));
    }
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    double avgBandwidth = 0.0;
    int localRet = RunRank(rank, worldSize, device, sendCount, numIters, uniqueId, &avgBandwidth);
    int anyFailed = 0;
    int localFailed = localRet != 0 ? 1 : 0;
    MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        if (anyFailed) {
            printf("Test FAILED (at least one rank reported mismatch).\n");
        } else {
            printf("Test PASSED (all ranks correctness check ok).\n");
        }
    }

    MPI_Finalize();
    return anyFailed ? 1 : 0;
}
