// Exercise production worker control flow; wrap only external ACL device APIs.
#define main benchmark_main
#include "../mpi_primitives/hccl_mpi_primitives.cc"
#undef main
#include <cassert>

static int stage = 0, calls = 0;
static aclrtMemcpyKind expected_kind = ACL_MEMCPY_DEVICE_TO_HOST;
static std::atomic<bool>* stop_flag = nullptr;
extern "C" aclError __wrap_aclrtSetDevice(int32_t) {
    return stage == 1 ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtMemcpy(void*, size_t capacity, const void*, size_t bytes, aclrtMemcpyKind kind) {
    assert(capacity == 64 && bytes == 64 && kind == expected_kind);
    ++calls;
    if (calls == 3) stop_flag->store(true);
    return stage == 2 ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}
int main() {
    Options opt;
    opt.pcieSizeBytes = 64;
    opt.pcieRanges = {{7, 10}};
    std::atomic<bool> stop{false}, formal{true};
    std::atomic<uint32_t> iteration{7};
    std::mutex mutex;
    std::condition_variable cv;
    stop_flag = &stop;
    char host[64], device[64];
    for (aclrtMemcpyKind kind : {ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEMCPY_HOST_TO_DEVICE}) {
      expected_kind = kind;
      for (stage = 0; stage <= 2; ++stage) {
        stop.store(false); calls = 0;
        std::vector<PcieStageMetrics> metrics(1);
        int status = RunPcieInterference(0, device, host, opt, stop, iteration, formal, mutex, cv, metrics, kind);
        if (stage == 0) {
            assert(status == ACL_SUCCESS && calls == 3);
            assert(metrics[0].attempts == 3 && metrics[0].successes == 3);
            assert(metrics[0].copy_seconds >= 0);
        } else {
            assert(status == ACL_ERROR_INVALID_PARAM);
            assert(metrics[0].successes == 0);
            assert(calls == (stage == 1 ? 0 : 1));
        }
    }
    }
    expected_kind = ACL_MEMCPY_DEVICE_TO_HOST;
    char program[] = "benchmark", flag[] = "--pcie-direction=both";
    char* argv[] = {program, flag};
    Options parsed;
    assert(ParseArgs(2, argv, &parsed) && parsed.pcieDirection == "both");
    stage = 0; calls = 0; stop.store(false); formal.store(false);
    std::vector<PcieStageMetrics> inactive(1);
    std::thread worker([&]() {
        assert(RunPcieInterference(0, device, host, opt, stop, iteration, formal, mutex, cv, inactive) == ACL_SUCCESS);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop.store(true); cv.notify_all(); worker.join();
    assert(calls == 0 && inactive[0].attempts == 0);
}
