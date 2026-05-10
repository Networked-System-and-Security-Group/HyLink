/**
 * NCCL AllGather + PCIe background traffic interference experiment framework (AmpCCL)
 *
 * Features:
 *   - Main thread: N AllGather iterations, output bandwidth/time per iter
 *   - Independent interference thread: issue PCIe D2H copies at period T (ms), simulate dynamic PCIe congestion (e.g. checkpointing)
 *   - Time-driven: interference every T ms, not tied to iter; no accumulation (next wave starts after previous ends)
 *   - Record time intervals for each interference and collective, for post-hoc overlap analysis
 *   - Bandwidth inference: iters overlapping interference marked as "affected", compare affected vs unaffected bandwidth
 *
 * Corresponding HCCL version: tests/hccl_test/allgather/allgather_with_pcie.cc
 *
 * Build: ./build.sh allgather_with_pcie.cc nccl_allgather_with_pcie
 * Run: bash ./run.sh ./nccl_allgather_with_pcie --devices=2 --send-count=268435456 --iters=50 --interference-ranges=10-20,30-40 --interference-size=67108864
 */

 #include <iostream>
 #include <vector>
 #include <thread>
 #include <chrono>
 #include <cstring>
 #include <cmath>
 #include <atomic>
 #include <mutex>
 #include <condition_variable>
 
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
 
 using Clock = std::chrono::steady_clock;
 using Ms = std::chrono::duration<double, std::milli>;
 
/** Time interval and bandwidth for one collective (for overlap analysis) */
 struct CollectiveInterval {
     uint32_t iter;
     double start_ms;   // ms relative to test start
     double end_ms;
     double bandwidth_gbps;
 };

/** Time interval for one interference event */
struct InterferenceInterval {
    uint32_t iter;     // AllGather iter it belongs to
    double start_ms;
    double end_ms;
};

/** Interference active range [start_iter, end_iter] (inclusive) */
struct InterferenceRange {
    uint32_t start_iter;
    uint32_t end_iter;
};

/** Global: experiment start time (set in main, all threads use for relative time) */
static Clock::time_point g_t0;
 
 /** Collective timeline (rank0 only writes) */
 static std::vector<CollectiveInterval> g_collective_times;
 static std::mutex g_collective_mutex;
 
/** Interference timeline (interference thread writes) */
 static std::vector<InterferenceInterval> g_interference_times;
 static std::mutex g_interference_mutex;
 
 /** Signal interference thread to exit */
 static std::atomic<bool> g_stop{false};
/** Current AllGather iter (updated by rank0 in thread) */
static std::atomic<uint32_t> g_current_iter{0};
 
 /** Check if two time ranges overlap: [a,b] and [c,d], overlap iff a < d && b > c */
 static bool TimeRangesOverlap(double a, double b, double c, double d) {
     return a < d && b > c;
 }
 
 struct ThreadContext {
     int device;
     int devCount;
     uint64_t sendCount;
     uint32_t numIters;
     ncclUniqueId uniqueId;  // NCCL shared unique ID for all ranks
 };
 
 static float ExpectedValue(int rank, uint32_t iter, uint64_t i) {
     return 1000000.0f * static_cast<float>(rank) +
            1000.0f * static_cast<float>(iter) +
            static_cast<float>(i);
 }
 
 static const uint32_t WARMUP_ITERS = 3;
 
 /** Wait for period or stop: use condition_variable to wake interference thread early when main ends */
 static std::mutex g_interference_sleep_mutex;
 static std::condition_variable g_interference_sleep_cv;
 
static bool IsIterInRanges(uint32_t iter, const std::vector<InterferenceRange>& ranges) {
    for (const auto& r : ranges) {
        if (iter >= r.start_iter && iter <= r.end_iter) return true;
    }
    return false;
}

static int InterferenceWorkerWithCv(
    int device,
    void* dev_buffer,
    void* host_buffer,
    cudaStream_t stream,
    size_t copy_size,
    const std::vector<InterferenceRange>& ranges) {
    uint32_t wave_id = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        uint32_t iter = g_current_iter.load(std::memory_order_relaxed);
        bool active = IsIterInRanges(iter, ranges);

        if (active) {
            auto t_start = Clock::now();
            double start_ms = std::chrono::duration_cast<Ms>(t_start - g_t0).count();

            CUDACHECK(cudaSetDevice(device));
            CUDACHECK(cudaMemcpyAsync(
                host_buffer, dev_buffer, copy_size,
                cudaMemcpyDeviceToHost, stream));

            CUDACHECK(cudaStreamSynchronize(stream));

            auto t_end = Clock::now();
            double end_ms = std::chrono::duration_cast<Ms>(t_end - g_t0).count();

            {
                std::lock_guard<std::mutex> lock(g_interference_mutex);
                g_interference_times.push_back({iter, start_ms, end_ms});
            }
            wave_id++;
        } else {
            // Current iter not in any interference range, brief wait to avoid busy spin
            std::unique_lock<std::mutex> lock(g_interference_sleep_mutex);
            g_interference_sleep_cv.wait_for(
                lock,
                std::chrono::milliseconds(1),
                []() { return g_stop.load(std::memory_order_relaxed); });
        }
    }
}
 
 int Sample(void* arg) {
     ThreadContext* ctx = static_cast<ThreadContext*>(arg);
     const int device = ctx->device;
     const int devCount = ctx->devCount;
     const uint64_t sendCount = ctx->sendCount;
     const uint32_t numIters = ctx->numIters;
 
     // Initialize NCCL communicator for this rank in each thread (must be after cudaSetDevice)
     CUDACHECK(cudaSetDevice(device));
     ncclComm_t comm;
     NCCLCHECK(ncclCommInitRank(&comm, devCount, ctx->uniqueId, device));
 
     const uint64_t recvCount = static_cast<uint64_t>(devCount) * sendCount;
     const size_t sendSize = sendCount * sizeof(float);
     const size_t recvSize = recvCount * sizeof(float);
 
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
     const double sendSizeGB = static_cast<double>(sendSize) / (1024.0 * 1024.0 * 1024.0);
 
     /* Warmup */
     for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t i = 0; i < sendCount; ++i) {
             pSend[i] = ExpectedValue(device, w, i);
         }
         CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
         NCCLCHECK(ncclAllGather(sendBuf, recvBuf, sendCount, ncclFloat32, comm, stream));
         CUDACHECK(cudaStreamSynchronize(stream));
         CUDACHECK(cudaMemcpy(hostRecv, recvBuf, recvSize, cudaMemcpyDeviceToHost));
         (void)w;
     }
 
     if (device == 0) {
         printf("[rank0] warmup done, %u iters\n", WARMUP_ITERS);
     }
 
    /* Formal test: record wall-clock interval + event time/bandwidth per iter, rank0 only writes g_collective_times */
     for (uint32_t iter = 0; iter < numIters; ++iter) {
        if (device == 0) {
            printf("[rank0] begin formal %u iters\n", iter);
            g_current_iter.store(iter, std::memory_order_relaxed);
        }
         float* pSend = static_cast<float*>(hostSend);
         for (uint64_t i = 0; i < sendCount; ++i) {
             pSend[i] = ExpectedValue(device, iter, i);
         }
         CUDACHECK(cudaMemcpy(sendBuf, hostSend, sendSize, cudaMemcpyHostToDevice));
 
         auto wall_start = Clock::now();
         CUDACHECK(cudaEventRecord(startEvent, stream));
         NCCLCHECK(ncclAllGather(sendBuf, recvBuf, sendCount, ncclFloat32, comm, stream));
         CUDACHECK(cudaEventRecord(endEvent, stream));
         CUDACHECK(cudaStreamSynchronize(stream));
         auto wall_end = Clock::now();
 
         float elapsedMs = 0.0f;
         CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));
         double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
         double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;
 
         double start_ms = std::chrono::duration_cast<Ms>(wall_start - g_t0).count();
         double end_ms = std::chrono::duration_cast<Ms>(wall_end - g_t0).count();
 
         if (device == 0) {
             {
                 std::lock_guard<std::mutex> lock(g_collective_mutex);
                 g_collective_times.push_back({iter, start_ms, end_ms, bandwidthGBps});
             }
             // printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s, wall [%.2f, %.2f] ms\n",
             //        iter, elapsedMs, bandwidthGBps, start_ms, end_ms);
         }
 
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
         printf("rank %d: FAILED %d element(s) mismatch\n", device, failed);
     } else {
         printf("rank %d: PASSED (devCount=%d sendCount=%lu iters=%u)\n",
                device, devCount, static_cast<unsigned long>(sendCount), numIters);
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
    printf("Usage: %s [options]\n", prog);
    printf("  --devices=N                     number of GPUs (default 4)\n");
    printf("  --send-count=K                  float elements per rank\n");
    printf("  --iters=I                       AllGather iterations\n");
    printf("  --interference-ranges=a-b,c-d   PCIe interference active iter ranges (e.g. 100-150,200-230)\n");
    printf("  --interference-size=S           D2H copy size in bytes per device per wave (default 256M)\n");
 }
 
int main(int argc, char* argv[]) {
    int devCount = 4;
    uint64_t sendCount = 1024;
    uint32_t numIters = 1;
    size_t interference_size = 256ULL * 1024 * 1024; /* 256MB */
    std::vector<InterferenceRange> ranges;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = std::atoi(argv[i] + 10);
        } else if (std::strncmp(argv[i], "--send-count=", 13) == 0) {
            sendCount = static_cast<uint64_t>(std::atoll(argv[i] + 13));
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            numIters = static_cast<uint32_t>(std::atoi(argv[i] + 8));
        } else if (std::strncmp(argv[i], "--interference-ranges=", 22) == 0) {
            const char* p = argv[i] + 22;
            while (*p) {
                char* end;
                unsigned long a = std::strtoul(p, &end, 10);
                if (*end == '-') {
                    char* end2;
                    unsigned long b = std::strtoul(end + 1, &end2, 10);
                    if (b >= a) {
                        ranges.push_back({static_cast<uint32_t>(a), static_cast<uint32_t>(b)});
                    }
                    p = end2;
                } else {
                    break;
                }
                if (*p == ',') ++p;
            }
        } else if (std::strncmp(argv[i], "--interference-size=", 20) == 0) {
            interference_size = static_cast<size_t>(std::atoll(argv[i] + 20));
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (devCount <= 0 || numIters == 0 || interference_size == 0) {
        printf("Invalid parameters\n");
        PrintUsage(argv[0]);
        return -1;
    }
 
     int availableDevs = 0;
     CUDACHECK(cudaGetDeviceCount(&availableDevs));
     if (devCount > availableDevs) {
         printf("Requested %d devices but only %d available\n", devCount, availableDevs);
         return -1;
     }
 
    printf("NCCL AllGather+PCIe interference: devices=%d send_count=%lu iters=%u size=%zu bytes\n",
           devCount, static_cast<unsigned long>(sendCount), numIters,
           interference_size);
    if (!ranges.empty()) {
        printf("Interference ranges (iter): ");
        for (const auto& r : ranges) {
            printf("[%u,%u] ", r.start_iter, r.end_iter);
        }
        printf("\n");
    }
 
    /* Allocate per-device device and host buffers for interference thread (D2H only, content arbitrary) */
    std::vector<void*> dev_buffers(devCount, nullptr);
    std::vector<void*> host_buffers(devCount, nullptr);
    std::vector<cudaStream_t> interference_streams(devCount, nullptr);
    for (int d = 0; d < devCount; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMalloc(&dev_buffers[d], interference_size));
        CUDACHECK(cudaMallocHost(&host_buffers[d], interference_size));
        CUDACHECK(cudaStreamCreate(&interference_streams[d]));
    }
 
     printf("Allocating NCCL communicators...\n");
     // Create global NCCL Unique ID, each thread calls ncclCommInitRank internally
     ncclUniqueId uniqueId;
     NCCLCHECK(ncclGetUniqueId(&uniqueId));
 
    g_stop.store(false);
    g_t0 = Clock::now();

    /* Create one interference thread per device, simulate per-card PCIe D2H interference */
    std::vector<std::thread> interference_threads;
    interference_threads.reserve(devCount);
    for (int d = 0; d < devCount; ++d) {
        interference_threads.emplace_back(
            InterferenceWorkerWithCv,
            d,
            dev_buffers[d],
            host_buffers[d],
            interference_streams[d],
            interference_size,
            std::ref(ranges));
    }
 
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

    g_stop.store(true);
    g_interference_sleep_cv.notify_all();
    for (auto& it : interference_threads) {
        it.join();
    }
 
     /* Free interference buffers */
     for (int d = 0; d < devCount; ++d) {
         CUDACHECK(cudaSetDevice(d));
         CUDACHECK(cudaFree(dev_buffers[d]));
         CUDACHECK(cudaFreeHost(host_buffers[d]));
         CUDACHECK(cudaStreamDestroy(interference_streams[d]));
     }
 
     /* ---------- Output timeline and overlap analysis (rank0 has collective records) ---------- */
     printf("\n=== Time intervals (relative ms from test start) ===\n");
     {
         std::lock_guard<std::mutex> lock(g_collective_mutex);
         // printf("Collective (iter, start_ms, end_ms, bandwidth_GB/s):\n");
         // for (const auto& c : g_collective_times) {
         //     printf("  iter %u: [%.2f, %.2f] ms, %.3f GB/s\n", c.iter, c.start_ms, c.end_ms, c.bandwidth_gbps);
         // }
     }
    {
        std::lock_guard<std::mutex> lock(g_interference_mutex);
        printf("Interference (iter, start_ms, end_ms):\n");
        for (const auto& i : g_interference_times) {
            printf("  iter %u: [%.2f, %.2f] ms\n", i.iter, i.start_ms, i.end_ms);
        }
    }
 
     /* Determine if each iter overlaps any interference, and compute affected/unaffected bandwidth */
     {
         std::lock_guard<std::mutex> lock_c(g_collective_mutex);
         std::lock_guard<std::mutex> lock_i(g_interference_mutex);
         std::vector<double> bw_affected, bw_unaffected;
         printf("\n=== Overlap analysis (affected = collective interval overlaps any interference) ===\n");
         for (const auto& c : g_collective_times) {
             bool affected = false;
             for (const auto& in : g_interference_times) {
                 if (TimeRangesOverlap(c.start_ms, c.end_ms, in.start_ms, in.end_ms)) {
                     affected = true;
                     break;
                 }
             }
             if (affected) {
                 bw_affected.push_back(c.bandwidth_gbps);
                 // printf("  iter %u: AFFECTED, %.3f GB/s\n", c.iter, c.bandwidth_gbps);
             } else {
                 bw_unaffected.push_back(c.bandwidth_gbps);
                 // printf("  iter %u: not affected, %.3f GB/s\n", c.iter, c.bandwidth_gbps);
             }
         }
         double avg_aff = 0.0, avg_unaff = 0.0;
         if (!bw_affected.empty()) {
             for (double b : bw_affected) avg_aff += b;
             avg_aff /= static_cast<double>(bw_affected.size());
         }
         if (!bw_unaffected.empty()) {
             for (double b : bw_unaffected) avg_unaff += b;
             avg_unaff /= static_cast<double>(bw_unaffected.size());
         }
         // printf("\nSummary: affected iters %zu (avg BW %.3f GB/s), unaffected %zu (avg BW %.3f GB/s)\n",
         //        bw_affected.size(), avg_aff, bw_unaffected.size(), avg_unaff);
     }
 
     if (anyFailed.load()) {
         printf("Test FAILED (correctness check failed).\n");
         return 1;
     }
     printf("Test PASSED.\n");
     return 0;
 }
 
 