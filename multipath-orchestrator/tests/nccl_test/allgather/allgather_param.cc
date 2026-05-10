/**
 * NCCL AllGather configurable parameter test
 *
 * Features:
 *   - Specify GPU count, send count per rank (element count), iteration count
 *   - Multiple AllGather rounds with correctness verification after each round
 *   - main returns 1 if any rank fails verification
 *
 * Corresponding HCCL version: tests/hccl_test/allgather/allgather_param.cc
 *
 * Build (in tests/nccl_test/allgather):
 *   ./build.sh allgather_param.cc nccl_allgather_param
 *
 * Run (with AMP-CCL NCCL hook):
 *   ./run.sh ./nccl_allgather_param
 *   ./run.sh ./nccl_allgather_param 2 4096 10
 *   ./run.sh ./nccl_allgather_param --devices=2 --send-count=1024 --iters=5
 */

 #include <iostream>
 #include <vector>
 #include <thread>
 #include <chrono>
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
    uint64_t sendCount;   // elements per rank
    uint32_t numIters;    // AllGather iteration count
    ncclUniqueId uniqueId; // NCCL shared unique ID for all ranks
 };
 
 // Expected value: rank j, iter round, element i
 static float ExpectedValue(int rank, uint32_t iter, uint64_t i) {
     return 1000000.0f * static_cast<float>(rank) +
            1000.0f * static_cast<float>(iter) +
            static_cast<float>(i);
 }
 
 static const uint32_t WARMUP_ITERS = 3;
 
 int Sample(void* arg) {
     ThreadContext* ctx = static_cast<ThreadContext*>(arg);
     const int device = ctx->device;
     const int devCount = ctx->devCount;
     const uint64_t sendCount = ctx->sendCount;
     const uint32_t numIters = ctx->numIters;
 
     
 
     const uint64_t recvCount = static_cast<uint64_t>(devCount) * sendCount;
     const size_t sendSize = sendCount * sizeof(float);
     const size_t recvSize = recvCount * sizeof(float);
 
     void* sendBuf = nullptr;
     void* recvBuf = nullptr;
 
     CUDACHECK(cudaSetDevice(device));
    printf("rank %d: init nccl comm\n", device);
     // Initialize NCCL communicator for this rank in each thread
     ncclComm_t comm;
     NCCLCHECK(ncclCommInitRank(&comm, devCount, ctx->uniqueId, device));
     CUDACHECK(cudaMalloc(&sendBuf, sendSize));
     CUDACHECK(cudaMalloc(&recvBuf, recvSize));
 
     void* hostSend = nullptr;
     void* hostRecv = nullptr;
     CUDACHECK(cudaMallocHost(&hostSend, sendSize));
     CUDACHECK(cudaMallocHost(&hostRecv, recvSize));
 
     cudaStream_t stream;
     CUDACHECK(cudaStreamCreate(&stream));
 
     // Use CUDA events for device timing
     cudaEvent_t startEvent = nullptr;
     cudaEvent_t endEvent = nullptr;
     CUDACHECK(cudaEventCreate(&startEvent));
     CUDACHECK(cudaEventCreate(&endEvent));
 
     int failed = 0;
     const double sendSizeGB = static_cast<double>(sendSize) / (1024.0 * 1024.0 * 1024.0);
 
     // Warmup: 3 rounds, not counted, comm only
     for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t i = 0; i < sendCount; ++i) {
             pSend[i] = ExpectedValue(device, w, i);
         }
 
         CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
         NCCLCHECK(ncclAllGather(sendBuf, recvBuf, sendCount, ncclFloat32, comm, stream));
         CUDACHECK(cudaStreamSynchronize(stream));
     }
     if (device == 0) {
         printf("Warmup done (%u iters)\n", WARMUP_ITERS);
     }
 
     double sumBandwidthGBps = 0.0;
     const uint32_t measureIters = numIters;
 
     // Formal test: numIters rounds, device event timing, output bandwidth
     for (uint32_t iter = 0; iter < numIters; ++iter) {
         if (device == 0) {
             printf("normal test on iter %u\n", iter);
         }
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t i = 0; i < sendCount; ++i) {
             pSend[i] = ExpectedValue(device, iter, i);
         }
         CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
         CUDACHECK(cudaEventRecord(startEvent, stream));
         NCCLCHECK(ncclAllGather(sendBuf, recvBuf, sendCount, ncclFloat32, comm, stream));
         CUDACHECK(cudaEventRecord(endEvent, stream));
         CUDACHECK(cudaStreamSynchronize(stream));
 
         float elapsedMs = 0.0f;
         CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));
         double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
         double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;
        //  if (device == 0) {
        //      printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s\n", iter, elapsedMs, bandwidthGBps);
        //      sumBandwidthGBps += bandwidthGBps;
        //  }
 
         CUDACHECK(cudaMemcpy(hostRecv, recvBuf, recvSize, cudaMemcpyDeviceToHost));
         float* pRecv = static_cast<float*>(hostRecv);
         for (int j = 0; j < devCount; ++j) {
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
         printf("rank %d: FAILED %d element(s) mismatch in correctness check\n", device, failed);
     } else {
         printf("rank %d: PASSED (devCount=%d sendCount=%lu iters=%u)\n",
                device, devCount, static_cast<unsigned long>(sendCount), numIters);
        //  if (device == 0 && measureIters > 0) {
        //      double avgBandwidth = sumBandwidthGBps / static_cast<double>(measureIters);
        //      printf("Average bandwidth over %u iters (event): %.3f GB/s\n", measureIters, avgBandwidth);
        //  }
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
     printf("Usage: %s [devices] [send_count] [iters]\n", prog);
     printf("   or: %s --devices=N --send-count=K --iters=I\n", prog);
     printf("  devices     number of GPUs (default 2)\n");
     printf("  send_count  float elements per rank (default 1024)\n");
     printf("  iters       number of AllGather iterations (default 1)\n");
 }
 
 int main(int argc, char* argv[]) {
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
             PrintUsage(argv[0]);
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
 
     if (devCount <= 0 || sendCount == 0 || numIters == 0) {
         printf("Invalid parameters: devices=%d send_count=%lu iters=%u\n",
                devCount, static_cast<unsigned long>(sendCount), numIters);
         PrintUsage(argv[0]);
         return -1;
     }
 
     int availableDevs = 0;
     CUDACHECK(cudaGetDeviceCount(&availableDevs));
     if (devCount > availableDevs) {
         printf("Requested %d devices but only %d available\n", devCount, availableDevs);
         return -1;
     }
 
     printf("NCCL AllGather param test: devices=%d send_count=%lu iters=%u (%.2f MB per rank per iter)\n",
            devCount, static_cast<unsigned long>(sendCount), numIters,
            sendCount * sizeof(float) / (1024.0 * 1024.0));
 
     // Create global NCCL Unique ID, each thread calls ncclCommInitRank internally
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
             if (r != 0) anyFailed.store(1);
         }, static_cast<size_t>(i));
     }
     for (auto& t : threads) {
         t.join();
     }
 
     if (anyFailed.load()) {
         printf("Test FAILED (at least one rank reported mismatch).\n");
         return 1;
     }
     printf("Test PASSED (all ranks correctness check ok).\n");
     return 0;
 }
 
 