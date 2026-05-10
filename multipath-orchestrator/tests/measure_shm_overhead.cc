#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

// Minimal shm barrier implementation matching ShmParamStore::BarrierSync
struct SharedHeader {
    std::atomic<uint32_t> barrier_count{0};
    std::atomic<uint32_t> barrier_sense{0};
};

void barrier_sync(SharedHeader* hdr, int nranks) {
    uint32_t my_sense = 1u - hdr->barrier_sense.load(std::memory_order_seq_cst);
    uint32_t count = hdr->barrier_count.fetch_add(1u, std::memory_order_seq_cst) + 1u;
    if (count == static_cast<uint32_t>(nranks)) {
        hdr->barrier_count.store(0u, std::memory_order_seq_cst);
        hdr->barrier_sense.store(my_sense, std::memory_order_seq_cst);
    } else {
        while (hdr->barrier_sense.load(std::memory_order_seq_cst) != my_sense) {
            sched_yield();
        }
    }
}

// Measure barrier latency in a multi-threaded setting
void measure_barrier_overhead(int nthreads, int iters) {
    const char* shm_name = "/test_barrier_overhead";
    shm_unlink(shm_name);

    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        std::cerr << "shm_open failed\n";
        return;
    }

    size_t shm_size = sizeof(SharedHeader);
    ftruncate(fd, shm_size);
    void* base = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::cerr << "mmap failed\n";
        close(fd);
        return;
    }

    SharedHeader* hdr = static_cast<SharedHeader*>(base);
    new (hdr) SharedHeader();

    std::vector<double> latencies_us;
    std::atomic<bool> start{false};

    auto worker = [&](int tid) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            barrier_sync(hdr, nthreads);
            auto t1 = std::chrono::high_resolution_clock::now();

            if (tid == 0) {
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                latencies_us.push_back(us);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker, t);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    start.store(true, std::memory_order_release);

    for (auto& th : threads) {
        th.join();
    }

    // Compute statistics
    double sum = 0.0;
    for (double v : latencies_us) sum += v;
    double mean = sum / latencies_us.size();

    std::vector<double> sorted = latencies_us;
    std::sort(sorted.begin(), sorted.end());
    double min_val = sorted.front();
    double max_val = sorted.back();
    double median = sorted[sorted.size() / 2];
    double p95 = sorted[static_cast<size_t>(sorted.size() * 0.95)];

    double var = 0.0;
    for (double v : latencies_us) {
        var += (v - mean) * (v - mean);
    }
    double stdev = std::sqrt(var / latencies_us.size());

    std::cout << "BarrierSync overhead (nthreads=" << nthreads << ", iters=" << iters << "):\n";
    std::cout << "  mean=" << mean << " us, median=" << median << " us\n";
    std::cout << "  min=" << min_val << " us, max=" << max_val << " us, p95=" << p95 << " us\n";
    std::cout << "  stdev=" << stdev << " us, cv=" << (stdev/mean*100.0) << "%\n";

    munmap(base, shm_size);
    close(fd);
    shm_unlink(shm_name);
}

int main() {
    std::cout << "Measuring ShmParamStore::BarrierSync overhead...\n\n";

    measure_barrier_overhead(4, 100);
    std::cout << "\n";
    measure_barrier_overhead(8, 100);
    std::cout << "\n";

    return 0;
}
