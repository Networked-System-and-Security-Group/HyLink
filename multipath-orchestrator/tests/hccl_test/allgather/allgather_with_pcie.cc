/**
 * AllGather + PCIe background traffic interference experiment framework (AmpCCL)
 *
 * Features:
 *   - Main thread: N AllGather iterations, output bandwidth/time per iter
 *   - Independent interference thread: issue PCIe D2H copies at period T (ms), simulate dynamic PCIe congestion (e.g. checkpointing)
 *   - Time-driven: interference every T ms, not tied to iter; no accumulation (next wave starts after previous ends)
 *   - Record time intervals for each interference and collective, for post-hoc overlap analysis
 *   - Bandwidth inference: iters overlapping interference marked as "affected", compare affected vs unaffected bandwidth
 *
 * Constraints:
 *   - No collective in main loop as interference; interference only from background thread via aclrtMemcpy D2H
 *   - 4-card test; each card performs one D2H per interference period (fixed size, configurable)
 *
 * Build: ./build.sh allgather_with_pcie.cc hccl_allgather_with_pcie
 * Run: ./run.sh ./hccl_allgather_with_pcie --devices=4 --send-count=67108864 --iters=20 --interference-period-ms=50 --interference-size=268435456
 *
 * Args:
 *   devices                 NPU count for AllGather, default 4
 *   send_count              float elements per rank
 *   iters                   AllGather iteration count
 *   interference-period-ms  interference period T (ms), trigger D2H every T ms
 *   interference-size       bytes per D2H copy per card per wave (fixed)
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
    uint32_t wave_id;
    double start_ms;
    double end_ms;
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

/** Check if two time ranges overlap: [a,b] and [c,d], overlap iff a < d && b > c */
static bool TimeRangesOverlap(double a, double b, double c, double d) {
    return a < d && b > c;
}

struct ThreadContext {
    HcclRootInfo* rootInfo;
    uint32_t device;
    uint32_t devCount;
    uint64_t sendCount;
    uint32_t numIters;
    /* Interference allocated by main, not in ThreadContext */
};

static float ExpectedValue(uint32_t rank, uint32_t iter, uint64_t i) {
    return 1000000.f * rank + 1000.f * iter + static_cast<float>(i);
}

static const uint32_t WARMUP_ITERS = 3;

/** Wait for period or stop: use condition_variable to wake interference thread early when main ends */
static std::mutex g_interference_sleep_mutex;
static std::condition_variable g_interference_sleep_cv;

static void InterferenceWorkerWithCv(uint32_t devCount,
                                     std::vector<void*>& dev_buffers,
                                     std::vector<void*>& host_buffers,
                                     size_t copy_size,
                                     double period_ms) {
    uint32_t wave_id = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        auto t_start = Clock::now();
        double start_ms = std::chrono::duration_cast<Ms>(t_start - g_t0).count();

        for (uint32_t d = 0; d < devCount; ++d) {
            aclError e = aclrtSetDevice(static_cast<int32_t>(d));
            if (e != ACL_SUCCESS) {
                printf("[interference] aclrtSetDevice(%u) failed: %d\n", d, e);
                continue;
            }
            e = aclrtMemcpy(host_buffers[d], copy_size, dev_buffers[d], copy_size, ACL_MEMCPY_DEVICE_TO_HOST);
            if (e != ACL_SUCCESS) {
                printf("[interference] aclrtMemcpy D2H device %u failed: %d\n", d, e);
            }
        }

        auto t_end = Clock::now();
        double end_ms = std::chrono::duration_cast<Ms>(t_end - g_t0).count();

        {
            std::lock_guard<std::mutex> lock(g_interference_mutex);
            g_interference_times.push_back({wave_id, start_ms, end_ms});
        }
        wave_id++;

        /* Next wave starts T ms after this start, avoid accumulation, period = start-to-start = T */
        auto next_start = t_start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(period_ms));
        auto wait_duration = next_start - Clock::now();
        if (wait_duration.count() > 0) {
            std::unique_lock<std::mutex> lock(g_interference_sleep_mutex);
            g_interference_sleep_cv.wait_for(lock, wait_duration,
                []() { return g_stop.load(std::memory_order_relaxed); });
        }
    }
}

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

    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    ACLCHECK(aclrtCreateEventWithFlag(&startEvent, ACL_EVENT_TIME_LINE));
    ACLCHECK(aclrtCreateEventWithFlag(&endEvent, ACL_EVENT_TIME_LINE));

    int failed = 0;
    const double sendSizeGB = sendSize / (1024.0 * 1024.0 * 1024.0);

    /* Warmup */
    for (uint32_t w = 0; w < WARMUP_ITERS; ++w) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(device, w, i);
        }
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        ACLCHECK(aclrtMemcpy(hostRecv, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
        (void)w;
    }

    if (device == 0) {
        printf("[rank0] warmup done, %u iters\n", WARMUP_ITERS);
    }

    /* Formal test: record wall-clock interval + event time/bandwidth per iter, rank0 only writes g_collective_times */
    for (uint32_t iter = 0; iter < numIters; ++iter) {
        float* pSend = static_cast<float*>(hostSend);
        for (uint64_t i = 0; i < sendCount; ++i) {
            pSend[i] = ExpectedValue(device, iter, i);
        }
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostSend, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));

        auto wall_start = Clock::now();
        ACLCHECK(aclrtRecordEvent(startEvent, stream));
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
        ACLCHECK(aclrtRecordEvent(endEvent, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        ACLCHECK(aclrtSynchronizeEvent(endEvent));
        auto wall_end = Clock::now();

        float elapsedMs = 0.0f;
        ACLCHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
        double bandwidthGBps = (elapsedSec > 1e-9) ? (sendSizeGB / elapsedSec) : 0.0;

        double start_ms = std::chrono::duration_cast<Ms>(wall_start - g_t0).count();
        double end_ms = std::chrono::duration_cast<Ms>(wall_end - g_t0).count();

        if (device == 0) {
            {
                std::lock_guard<std::mutex> lock(g_collective_mutex);
                g_collective_times.push_back({iter, start_ms, end_ms, bandwidthGBps});
            }
            printf("iter %u: %.3f ms (event), bandwidth: %.3f GB/s, wall [%.2f, %.2f] ms\n",
                   iter, elapsedMs, bandwidthGBps, start_ms, end_ms);
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
        printf("rank %u: FAILED %d element(s) mismatch\n", device, failed);
    } else {
        printf("rank %u: PASSED (devCount=%u sendCount=%lu iters=%u)\n",
               device, devCount, (unsigned long)sendCount, numIters);
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
    printf("Usage: %s [options]\n", prog);
    printf("  --devices=N                     number of NPUs (default 4)\n");
    printf("  --send-count=K                  float elements per rank\n");
    printf("  --iters=I                      AllGather iterations\n");
    printf("  --interference-period-ms=T      PCIe interference period in ms (default 100)\n");
    printf("  --interference-size=S           D2H copy size in bytes per device per wave (default 256M)\n");
}

int main(int argc, char* argv[]) {
    uint32_t devCount = 4;
    uint64_t sendCount = 1024;
    uint32_t numIters = 1;
    double interference_period_ms = 100.0;
    size_t interference_size = 256ULL * 1024 * 1024; /* 256MB */

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--devices=", 10) == 0) {
            devCount = static_cast<uint32_t>(atoi(argv[i] + 10));
        } else if (strncmp(argv[i], "--send-count=", 13) == 0) {
            sendCount = static_cast<uint64_t>(atoll(argv[i] + 13));
        } else if (strncmp(argv[i], "--iters=", 8) == 0) {
            numIters = static_cast<uint32_t>(atoi(argv[i] + 8));
        } else if (strncmp(argv[i], "--interference-period-ms=", 26) == 0) {
            interference_period_ms = atof(argv[i] + 26);
        } else if (strncmp(argv[i], "--interference-size=", 20) == 0) {
            interference_size = static_cast<size_t>(atoll(argv[i] + 20));
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (devCount == 0 || numIters == 0 || interference_period_ms <= 0 || interference_size == 0) {
        printf("Invalid parameters\n");
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

    printf("AllGather+PCIe interference: devices=%u send_count=%lu iters=%u period=%.1f ms size=%zu bytes\n",
           devCount, (unsigned long)sendCount, numIters, interference_period_ms, interference_size);

    /* Allocate per-device device and host buffers for interference thread (D2H only, content arbitrary) */
    std::vector<void*> dev_buffers(devCount, nullptr);
    std::vector<void*> host_buffers(devCount, nullptr);
    for (uint32_t d = 0; d < devCount; ++d) {
        ACLCHECK(aclrtSetDevice(static_cast<int32_t>(d)));
        ACLCHECK(aclrtMalloc(&dev_buffers[d], interference_size, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMallocHost(&host_buffers[d], interference_size));
    }

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void* rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo* rootInfo = static_cast<HcclRootInfo*>(rootInfoBuf);
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    g_stop.store(false);
    g_t0 = Clock::now();

    /* Start interference thread first (time-driven), then collective threads */
    std::thread interference_thread(InterferenceWorkerWithCv,
                                   devCount,
                                   std::ref(dev_buffers),
                                   std::ref(host_buffers),
                                   interference_size,
                                   interference_period_ms);

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

    g_stop.store(true);
    g_interference_sleep_cv.notify_all();
    interference_thread.join();

    /* Free interference buffers */
    for (uint32_t d = 0; d < devCount; ++d) {
        ACLCHECK(aclrtSetDevice(static_cast<int32_t>(d)));
        ACLCHECK(aclrtFree(dev_buffers[d]));
        ACLCHECK(aclrtFreeHost(host_buffers[d]));
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());

    /* ---------- Output timeline and overlap analysis (rank0 has collective records) ---------- */
    printf("\n=== Time intervals (relative ms from test start) ===\n");
    {
        std::lock_guard<std::mutex> lock(g_collective_mutex);
        printf("Collective (iter, start_ms, end_ms, bandwidth_GB/s):\n");
        for (const auto& c : g_collective_times) {
            printf("  iter %u: [%.2f, %.2f] ms, %.3f GB/s\n", c.iter, c.start_ms, c.end_ms, c.bandwidth_gbps);
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_interference_mutex);
        printf("Interference (wave_id, start_ms, end_ms):\n");
        for (const auto& i : g_interference_times) {
            printf("  wave %u: [%.2f, %.2f] ms\n", i.wave_id, i.start_ms, i.end_ms);
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
                printf("  iter %u: AFFECTED, %.3f GB/s\n", c.iter, c.bandwidth_gbps);
            } else {
                bw_unaffected.push_back(c.bandwidth_gbps);
                printf("  iter %u: not affected, %.3f GB/s\n", c.iter, c.bandwidth_gbps);
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
        printf("\nSummary: affected iters %zu (avg BW %.3f GB/s), unaffected %zu (avg BW %.3f GB/s)\n",
               bw_affected.size(), avg_aff, bw_unaffected.size(), avg_unaff);
    }

    if (anyFailed.load()) {
        printf("Test FAILED (correctness check failed).\n");
        return 1;
    }
    printf("Test PASSED.\n");
    return 0;
}
