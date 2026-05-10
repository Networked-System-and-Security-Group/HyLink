/**
 * HCCL AllGather + PCIe background traffic interference experiment (AmpCCL)
 *
 * Features:
 *   - One MPI process per NPU, with one background interference thread
 *   - Main process thread: N AllGather iterations, output max-rank event time
 *   - Independent interference stream: async PCIe D2H copies when current iter
 *     falls within configured ranges, simulate dynamic PCIe congestion
 *   - Range-driven: interference active during specified iter ranges only
 *   - Record time intervals for each interference and collective, for post-hoc
 *     overlap analysis and affected/unaffected bandwidth comparison
 *
 * Corresponding NCCL version: tests/nccl_test/allgather/allgather_with_pcie.cc
 *
 * Build: ./build.sh allgather_with_pcie.cc hccl_allgather_with_pcie
 * Run:
 *   NP=2 ./run.sh ./hccl_allgather_with_pcie --devices=2 --send-count=268435456 --iters=50 \
 *       --interference-ranges=10-20,30-40 --interference-size=67108864
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
#include <algorithm>
#include <cstdint>
#include <mpi.h>
#include <unistd.h>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        const aclError status = (ret);                                                         \
        if (status != ACL_SUCCESS) {                                                           \
            printf("acl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, status); \
            return status;                                                                     \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                         \
    do {                                                                                       \
        const HcclResult status = (ret);                                                        \
        if (status != HCCL_SUCCESS) {                                                          \
            printf("hccl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, status); \
            return status;                                                                     \
        }                                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

/** Time interval and bandwidth for one collective (for overlap analysis) */
struct CollectiveInterval {
    uint32_t iter;
    double start_ms;   // ms relative to test start
    double end_ms;
    double bandwidth_gbps;
    uint32_t device;
    double event_ms;
};

/** Time interval for one interference event */
struct InterferenceInterval {
    uint32_t iter;     // AllGather iter it belongs to
    double start_ms;
    double end_ms;
    int device;
};

/** Interference active range [start_iter, end_iter] (inclusive) */
struct InterferenceRange {
    uint32_t start_iter;
    uint32_t end_iter;
};

/** Global: experiment start time (set in main, all threads use for relative time) */
static Clock::time_point g_t0;

/** Local process's collective timeline */
static std::vector<CollectiveInterval> g_collective_times;
static std::mutex g_collective_mutex;

/** Interference timeline (interference threads write) */
static std::vector<InterferenceInterval> g_interference_times;
static std::mutex g_interference_mutex;

/** Signal interference threads to exit */
static std::atomic<bool> g_stop{false};

/** Local formal iteration; each MPI rank updates its own copy */
// Sentinel prevents ranges starting at zero from interfering with warmup.
static std::atomic<uint32_t> g_current_iter{UINT32_MAX};

static bool TimeRangesOverlap(double a, double b, double c, double d) {
    return a < d && b > c;
}

static bool IsIterInRanges(uint32_t iter, const std::vector<InterferenceRange>& ranges) {
    for (const auto& r : ranges) {
        if (iter >= r.start_iter && iter <= r.end_iter) return true;
    }
    return false;
}

struct RankContext {
    HcclRootInfo* rootInfo;
    uint32_t device;
    uint32_t rank;
    uint32_t devCount;
    uint64_t sendCount;
    uint32_t numIters;
    uint32_t warmupIters;
    std::atomic<int>* interferenceError;
    const std::vector<InterferenceRange>* ranges;
};

static float ExpectedValue(uint32_t rank, uint32_t iter, uint64_t i, uint32_t nranks) {
    return static_cast<float>(iter * nranks + i % 10000 + rank);
}

/** Condition variable for interference thread sleep/wake */
static std::mutex g_interference_sleep_mutex;
static std::condition_variable g_interference_sleep_cv;

static int InterferenceWorkerWithCv(
    int device,
    void* dev_buffer,
    void* host_buffer,
    aclrtStream stream,
    size_t copy_size,
    const std::vector<InterferenceRange>& ranges) {

    while (!g_stop.load(std::memory_order_relaxed)) {
        uint32_t iter = g_current_iter.load(std::memory_order_relaxed);
        bool active = iter != UINT32_MAX && IsIterInRanges(iter, ranges);

        if (active) {
            auto t_start = Clock::now();
            double start_ms = std::chrono::duration_cast<Ms>(t_start - g_t0).count();

            ACLCHECK(aclrtSetDevice(device));
            // Match the CUDA reference: one async copy on an independent stream,
            // drained before submitting the next wave (no unbounded queue).
            ACLCHECK(aclrtMemcpyAsync(host_buffer, copy_size, dev_buffer, copy_size,
                                      ACL_MEMCPY_DEVICE_TO_HOST, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));

            auto t_end = Clock::now();
            double end_ms = std::chrono::duration_cast<Ms>(t_end - g_t0).count();

            {
                std::lock_guard<std::mutex> lock(g_interference_mutex);
                g_interference_times.push_back({iter, start_ms, end_ms, device});
            }
        } else {
            // Current iter not in any interference range — brief wait to avoid busy spin
            std::unique_lock<std::mutex> lock(g_interference_sleep_mutex);
            g_interference_sleep_cv.wait_for(
                lock,
                std::chrono::milliseconds(1),
                []() { return g_stop.load(std::memory_order_relaxed); });
        }
    }
    return 0;
}

int Sample(void* arg) {
    RankContext* ctx = static_cast<RankContext*>(arg);
    const uint32_t device = ctx->device;
    const uint32_t rank = ctx->rank;
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
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, rank, &hcclComm));

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    ACLCHECK(aclrtCreateEventWithFlag(&startEvent, ACL_EVENT_TIME_LINE));
    ACLCHECK(aclrtCreateEventWithFlag(&endEvent, ACL_EVENT_TIME_LINE));

    int failed = 0;
    const double sendSizeGB = sendSize / (1024.0 * 1024.0 * 1024.0);

    /* Warmup */
    for (uint32_t w = 0; w < ctx->warmupIters; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(rank, w, i, devCount);
        }
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        MPI_Barrier(MPI_COMM_WORLD);
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        ACLCHECK(aclrtMemcpy(hostRecv, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
        (void)w;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank0] warmup done, %u iters\n", ctx->warmupIters);
    }

    /* Formal test: record wall-clock interval + event time/bandwidth per iter */
    for (uint32_t iter = 0; iter < numIters; ++iter) {
        if (ctx->interferenceError->load() != 0) return 1;
        g_current_iter.store(iter, std::memory_order_release);
        if (rank == 0) {
            printf("[rank0] begin formal %u iters\n", iter);
        }

        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(rank, iter, i, devCount);
        }
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        // Input preparation and cross-rank skew must not inflate event timing.
        MPI_Barrier(MPI_COMM_WORLD);

        auto wall_start = Clock::now();
        ACLCHECK(aclrtRecordEvent(startEvent, stream));
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtRecordEvent(endEvent, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        auto wall_end = Clock::now();

        float elapsedMs = 0.0f;
        ACLCHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;

        double start_ms = std::chrono::duration_cast<Ms>(wall_start - g_t0).count();
        double end_ms = std::chrono::duration_cast<Ms>(wall_end - g_t0).count();

        {
            std::lock_guard<std::mutex> lock(g_collective_mutex);
            g_collective_times.push_back({iter, start_ms, end_ms, bandwidthGBps, device, elapsedMs});
        }

        double local_event_ms = elapsedMs;
        double max_event_ms = 0.0;
        MPI_Allreduce(&local_event_ms, &max_event_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("iter=%u event_ms=%.6f bw_GBps=%.6f\n", iter, max_event_ms,
                   max_event_ms > 0.0 ? static_cast<double>(recvSize) / (max_event_ms * 1e6) : 0.0);
            fflush(stdout);
        }

        ACLCHECK(aclrtMemcpy(hostRecv, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
        float* pRecv = static_cast<float*>(hostRecv);
        for (uint32_t j = 0; j < devCount; ++j) {
            for (uint64_t i = 0; i < sendCount; ++i) {
                float expected = ExpectedValue(j, iter, i, devCount);
                float actual = pRecv[j * sendCount + i];
                if (std::fabs(actual - expected) > 1e-3f) {
                    failed++;
                }
            }
        }
    }

    if (failed > 0) {
        printf("rank %u: FAILED %d element(s) mismatch\n", rank, failed);
    } else {
        printf("rank %u: PASSED (send_count=%lu iters=%u)\n", rank,
               static_cast<unsigned long>(sendCount), numIters);
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
    printf("Usage: mpirun -np N %s [options] (single node, one process per NPU)\n", prog);
    printf("  --devices=N                     optional check: must equal MPI world size\n");
    printf("  --send-count=K                  float elements per rank\n");
    printf("  --iters=I                       AllGather iterations\n");
    printf("  --warmup=W                      warmup iterations (default 6)\n");
    printf("  --interference-ranges=a-b,c-d   inclusive ranges [a,b], matching the CUDA reference\n");
    printf("  --interference-size=S           D2H copy size in bytes per device per wave (default 256M)\n");
}

static int RunExperiment(int argc, char* argv[]) {
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    uint32_t devCount = 0;
    uint64_t sendCount = 1024;
    uint32_t numIters = 1;
    uint32_t warmupIters = 6;
    size_t interference_size = 256ULL * 1024 * 1024; /* 256MB */
    std::vector<InterferenceRange> ranges;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = static_cast<uint32_t>(std::atoi(argv[i] + 10));
        } else if (std::strncmp(argv[i], "--send-count=", 13) == 0) {
            sendCount = static_cast<uint64_t>(std::atoll(argv[i] + 13));
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            numIters = static_cast<uint32_t>(std::atoi(argv[i] + 8));
        } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
            warmupIters = static_cast<uint32_t>(std::atoi(argv[i] + 9));
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
            if (worldRank == 0) PrintUsage(argv[0]);
            return 0;
        }
    }

    if ((devCount != 0 && devCount != static_cast<uint32_t>(worldSize)) ||
        sendCount == 0 || numIters == 0 || interference_size == 0) {
        printf("Invalid parameters\n");
        PrintUsage(argv[0]);
        return -1;
    }
    devCount = static_cast<uint32_t>(worldSize);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
    if (localSize != worldSize) {
        fprintf(stderr, "This experiment supports a single node only\n");
        return 1;
    }

    ACLCHECK(aclInit(nullptr));
    uint32_t availableDevs = 0;
    ACLCHECK(aclrtGetDeviceCount(&availableDevs));
    if (devCount > availableDevs) {
        printf("Requested %u devices but only %u available\n", devCount, availableDevs);
        aclFinalize();
        return -1;
    }

    const int device = localRank;
    ACLCHECK(aclrtSetDevice(device));
    printf("rank %d/%d pid=%ld using device %d\n", worldRank, worldSize,
           static_cast<long>(getpid()), device);
    if (worldRank == 0) {
        printf("HCCL AllGather+PCIe interference: execution=mpi mode=async dedicated_stream=1 max_inflight=1 devices=%u send_count=%lu iters=%u interference_size=%zu bytes\n",
               devCount, (unsigned long)sendCount, numIters, interference_size);
        printf("Interference ranges (iter): ");
        for (const auto& r : ranges) {
            printf("[%u,%u] ", r.start_iter, r.end_iter);
        }
        printf("\n");
    }

    // Allocate only this process's background buffers/stream.
    void* dev_buffer = nullptr;
    void* host_buffer = nullptr;
    aclrtStream interference_stream = nullptr;
    if (!ranges.empty()) {
        ACLCHECK(aclrtMalloc(&dev_buffer, interference_size, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMallocHost(&host_buffer, interference_size));
        ACLCHECK(aclrtCreateStream(&interference_stream));
    }

    HcclRootInfo rootInfo{};
    if (worldRank == 0) HCCLCHECK(HcclGetRootInfo(&rootInfo));
    MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

    g_stop.store(false);
    // CLOCK_MONOTONIC/steady_clock is shared by processes on this single node.
    // Broadcast one epoch so cross-rank timeline differences remain meaningful.
    int64_t epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
    MPI_Bcast(&epoch_ns, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    g_t0 = Clock::time_point(std::chrono::duration_cast<Clock::duration>(
        std::chrono::nanoseconds(epoch_ns)));

    /* One local background worker; all MPI calls stay on the main thread. */
    std::atomic<int> anyFailed{0};
    std::thread interference_thread;
    if (!ranges.empty()) {
        interference_thread = std::thread([&]() {
            const int status = InterferenceWorkerWithCv(
                device, dev_buffer, host_buffer, interference_stream, interference_size, ranges);
            if (status != ACL_SUCCESS) {
                anyFailed.store(1);
                g_stop.store(true);
                g_interference_sleep_cv.notify_all();
            }
        });
    }

    RankContext context{&rootInfo, static_cast<uint32_t>(device), static_cast<uint32_t>(worldRank),
                        devCount, sendCount, numIters, warmupIters, &anyFailed, &ranges};
    const int sample_status = Sample(&context);
    if (sample_status != 0) MPI_Abort(MPI_COMM_WORLD, 1);

    g_stop.store(true);
    g_interference_sleep_cv.notify_all();
    if (interference_thread.joinable()) interference_thread.join();

    if (!ranges.empty()) {
        ACLCHECK(aclrtDestroyStream(interference_stream));
        ACLCHECK(aclrtFree(dev_buffer));
        ACLCHECK(aclrtFreeHost(host_buffer));
    }

    ACLCHECK(aclFinalize());

    // Print buffered timelines only after all workers finish, avoiding I/O in
    // the measured copy loop. Host intervals include API/synchronization time;
    // overlap is a host-interval proxy, not a measurement of DMA engine busy.
    for (int output_rank = 0; output_rank < worldSize; ++output_rank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (worldRank != output_rank) continue;
        for (const auto& c : g_collective_times) {
            printf("collective device=%u iter=%u start_ms=%.6f end_ms=%.6f event_ms=%.6f\n",
                   c.device, c.iter, c.start_ms, c.end_ms, c.event_ms);
        }
        for (const auto& interval : g_interference_times) {
            printf("interference device=%d iter=%u start_ms=%.6f end_ms=%.6f bytes=%zu\n",
                   interval.device, interval.iter, interval.start_ms, interval.end_ms, interference_size);
        }
        size_t copies = 0;
        double active_ms = 0.0;
        for (const auto& interval : g_interference_times) {
            if (interval.device == device) {
                ++copies;
                active_ms += interval.end_ms - interval.start_ms;
            }
        }
        printf("interference_summary device=%d copies=%zu copy_active_ms=%.6f copy_interval_GBps=%.6f\n",
               device, copies, active_ms,
               active_ms > 0.0 ? copies * static_cast<double>(interference_size) / (active_ms * 1e6) : 0.0);
        fflush(stdout);
    }

    int localFailed = anyFailed.load(), globalFailed = 0;
    MPI_Allreduce(&localFailed, &globalFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (worldRank == 0) printf("Test %s.\n", globalFailed ? "FAILED" : "PASSED");
    return globalFailed ? 1 : 0;
}

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    const int status = RunExperiment(argc, argv);
    if (status != 0) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    return status;
}
