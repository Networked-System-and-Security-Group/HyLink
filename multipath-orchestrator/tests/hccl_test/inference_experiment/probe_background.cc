// Standalone two-rank probe of the exact same synchronous background workers.
#define main benchmark_main
#include "../mpi_primitives/hccl_mpi_primitives.cc"
#undef main

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Options opt;
    if (!ParseArgs(argc, argv, &opt)) { MPI_Finalize(); return 1; }
    opt.pcieSizeBytes = 256ULL * 1024 * 1024;
    opt.pcieRanges = {{0, 1}};
    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(rank));
    std::array<PcieWorkerState, 2> workers;
    const size_t count = opt.pcieDirection == "both" ? 2 : 1;
    for (size_t n = 0; n < count; ++n) {
        auto& w = workers[n];
        w.kind = opt.pcieDirection == "h2d" || n == 1
            ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
        w.metrics.resize(1);
        ACLCHECK(aclrtMallocHost(&w.host, opt.pcieSizeBytes));
        ACLCHECK(aclrtMalloc(&w.device, opt.pcieSizeBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        std::memset(w.host, 0x5a, opt.pcieSizeBytes);
        ACLCHECK(aclrtMemset(w.device, opt.pcieSizeBytes, 0xa5, opt.pcieSizeBytes));
    }
    std::atomic<bool> stop{false}, formal{true};
    std::atomic<uint32_t> iter{0};
    std::mutex mutex;
    std::condition_variable cv;
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t n = 0; n < count; ++n) {
        workers[n].thread = std::thread([&, n]() {
            auto& w = workers[n];
            w.status.store(RunPcieInterference(rank, w.device, w.host, opt, stop, iter,
                           formal, mutex, cv, w.metrics, w.kind));
        });
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));
    stop.store(true);
    cv.notify_all();
    for (size_t n = 0; n < count; ++n) workers[n].thread.join();
    int failed = 0, total = 0;
    for (size_t n = 0; n < count; ++n) {
        auto& w = workers[n];
        unsigned char expected = 0xa5;
        if (w.kind == ACL_MEMCPY_HOST_TO_DEVICE) {
            expected = 0x5a;
            std::memset(w.host, 0, opt.pcieSizeBytes);
            ACLCHECK(aclrtMemcpy(w.host, opt.pcieSizeBytes, w.device,
                                 opt.pcieSizeBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        bool verified = true;
        const auto* data = static_cast<const unsigned char*>(w.host);
        for (size_t i = 0; i < opt.pcieSizeBytes; ++i) {
            if (data[i] != expected) { verified = false; break; }
        }
        const auto& m = w.metrics[0];
        const double bytes = static_cast<double>(m.successes) * opt.pcieSizeBytes;
        std::printf("background_only rank=%d direction=%s status=%d attempts=%llu successes=%llu "
                    "seconds=%.6f GBps=%.6f verified=%d\n", rank,
                    w.kind == ACL_MEMCPY_DEVICE_TO_HOST ? "d2h" : "h2d", w.status.load(),
                    static_cast<unsigned long long>(m.attempts),
                    static_cast<unsigned long long>(m.successes), m.copy_seconds,
                    m.copy_seconds > 0 ? bytes / m.copy_seconds / 1e9 : 0,
                    static_cast<int>(verified));
        failed += w.status.load() != ACL_SUCCESS || m.successes == 0 || !verified;
        ACLCHECK(aclrtFree(w.device));
        ACLCHECK(aclrtFreeHost(w.host));
    }
    MPI_Allreduce(&failed, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ACLCHECK(aclFinalize());
    MPI_Finalize();
    return total != 0;
}
