#include <mpi.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if ((ret) != ACL_SUCCESS) {                                                            \
            std::fprintf(stderr, "acl interface return err %s:%d, retcode: %d\n",              \
                         __FILE__, __LINE__, static_cast<int>(ret));                            \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

enum class DmaMode {
    D2H,
    H2D,
    MIXED
};

static const char* ModeName(DmaMode mode) {
    switch (mode) {
        case DmaMode::D2H: return "d2h";
        case DmaMode::H2D: return "h2d";
        case DmaMode::MIXED: return "mixed";
    }
    return "unknown";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DmaMode mode = DmaMode::D2H;
    size_t size_mb = 128;
    uint32_t iters = 50;
    uint32_t warmup = 5;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            std::string m = argv[i] + 7;
            if (m == "d2h") mode = DmaMode::D2H;
            else if (m == "h2d") mode = DmaMode::H2D;
            else if (m == "mixed") mode = DmaMode::MIXED;
        } else if (std::strncmp(argv[i], "--size-mb=", 10) == 0) {
            size_mb = std::strtoul(argv[i] + 10, nullptr, 10);
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            iters = std::strtoul(argv[i] + 8, nullptr, 10);
        } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
            warmup = std::strtoul(argv[i] + 9, nullptr, 10);
        }
    }

    size_t bytes = size_mb * 1024ULL * 1024ULL;

    ACLCHECK(aclInit(nullptr));
    uint32_t count = 0;
    ACLCHECK(aclrtGetDeviceCount(&count));
    int dev_id = rank % static_cast<int>(count);
    ACLCHECK(aclrtSetDevice(dev_id));

    void *host_buf = nullptr, *dev_buf = nullptr;
    void *host_buf2 = nullptr, *dev_buf2 = nullptr;

    ACLCHECK(aclrtMallocHost(&host_buf, bytes));
    ACLCHECK(aclrtMalloc(&dev_buf, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    std::memset(host_buf, 0x5a, bytes);
    ACLCHECK(aclrtMemset(dev_buf, bytes, 0x3c, bytes));

    if (mode == DmaMode::MIXED) {
        ACLCHECK(aclrtMallocHost(&host_buf2, bytes));
        ACLCHECK(aclrtMalloc(&dev_buf2, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
        std::memset(host_buf2, 0x5a, bytes);
        ACLCHECK(aclrtMemset(dev_buf2, bytes, 0x3c, bytes));
    }

    aclrtStream stream = nullptr;
    aclrtStream stream2 = nullptr;
    ACLCHECK(aclrtCreateStream(&stream));
    if (mode == DmaMode::MIXED) {
        ACLCHECK(aclrtCreateStream(&stream2));
    }

    aclrtEvent start_event = nullptr, end_event = nullptr;
    ACLCHECK(aclrtCreateEventWithFlag(&start_event, ACL_EVENT_TIME_LINE));
    ACLCHECK(aclrtCreateEventWithFlag(&end_event, ACL_EVENT_TIME_LINE));

    if (rank == 0) {
        std::printf("MPI DMA Benchmark: mode=%s nranks=%d size_mb=%zu bytes=%zu iters=%u warmup=%u\n",
                    ModeName(mode), world_size, size_mb, bytes, iters, warmup);
        std::fflush(stdout);
    }
    std::printf("rank %d/%d assigned device %d\n", rank, world_size, dev_id);
    std::fflush(stdout);

    // Warmup
    for (uint32_t w = 0; w < warmup; ++w) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mode == DmaMode::D2H) {
            ACLCHECK(aclrtMemcpyAsync(host_buf, bytes, dev_buf, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
        } else if (mode == DmaMode::H2D) {
            ACLCHECK(aclrtMemcpyAsync(dev_buf, bytes, host_buf, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
        } else {
            ACLCHECK(aclrtMemcpyAsync(host_buf, bytes, dev_buf, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
            ACLCHECK(aclrtMemcpyAsync(dev_buf2, bytes, host_buf2, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream2));
            ACLCHECK(aclrtSynchronizeStream(stream));
            ACLCHECK(aclrtSynchronizeStream(stream2));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double total_max_ms = 0.0;
    const uint64_t bytes_per_rank = (mode == DmaMode::MIXED) ? bytes * 2 : bytes;

    for (uint32_t iter = 0; iter < iters; ++iter) {
        MPI_Barrier(MPI_COMM_WORLD);

        ACLCHECK(aclrtRecordEvent(start_event, stream));
        if (mode == DmaMode::D2H) {
            ACLCHECK(aclrtMemcpyAsync(host_buf, bytes, dev_buf, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
            ACLCHECK(aclrtRecordEvent(end_event, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
        } else if (mode == DmaMode::H2D) {
            ACLCHECK(aclrtMemcpyAsync(dev_buf, bytes, host_buf, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
            ACLCHECK(aclrtRecordEvent(end_event, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
        } else {
            ACLCHECK(aclrtMemcpyAsync(host_buf, bytes, dev_buf, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
            ACLCHECK(aclrtMemcpyAsync(dev_buf2, bytes, host_buf2, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream2));
            ACLCHECK(aclrtSynchronizeStream(stream2));
            ACLCHECK(aclrtRecordEvent(end_event, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
        }

        ACLCHECK(aclrtSynchronizeEvent(end_event));
        float local_ms = 0.0f;
        ACLCHECK(aclrtEventElapsedTime(&local_ms, start_event, end_event));

        double d_local_ms = static_cast<double>(local_ms);
        double max_ms = 0.0;
        double min_ms = 0.0;
        MPI_Allreduce(&d_local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_local_ms, &min_ms, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        total_max_ms += max_ms;

        if (rank == 0) {
            double iter_sec = max_ms / 1000.0;
            // Per-rank bandwidth based on slowest rank
            double rank_bw = (iter_sec > 1e-9) ? (static_cast<double>(bytes_per_rank) / 1e9 / iter_sec) : 0.0;
            // Total aggregate bandwidth across all ranks
            double agg_bw = rank_bw * world_size;
            std::printf("iter=%u max_time_ms=%.3f min_time_ms=%.3f rank_bw_GBps=%.3f agg_bw_GBps=%.3f\n",
                        iter, max_ms, min_ms, rank_bw, agg_bw);
            std::fflush(stdout);
        }
    }

    if (rank == 0 && iters > 0) {
        double avg_ms = total_max_ms / static_cast<double>(iters);
        double avg_sec = avg_ms / 1000.0;
        double avg_rank_bw = (avg_sec > 1e-9) ? (static_cast<double>(bytes_per_rank) / 1e9 / avg_sec) : 0.0;
        std::printf("=== Summary: avg_max_time_ms=%.3f avg_rank_bw_GBps=%.3f avg_agg_bw_GBps=%.3f ===\n",
                    avg_ms, avg_rank_bw, avg_rank_bw * world_size);
        std::fflush(stdout);
    }

    ACLCHECK(aclrtDestroyEvent(start_event));
    ACLCHECK(aclrtDestroyEvent(end_event));
    ACLCHECK(aclrtDestroyStream(stream));
    if (stream2) ACLCHECK(aclrtDestroyStream(stream2));
    ACLCHECK(aclrtFree(dev_buf));
    ACLCHECK(aclrtFreeHost(host_buf));
    if (dev_buf2) ACLCHECK(aclrtFree(dev_buf2));
    if (host_buf2) ACLCHECK(aclrtFreeHost(host_buf2));
    ACLCHECK(aclFinalize());
    MPI_Finalize();
    return 0;
}
