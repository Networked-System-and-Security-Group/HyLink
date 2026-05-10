// Exercise the real worker; replace only external ACL device calls at link time.
#define main hccl_experiment_main
#include "../hccl_test/allgather/allgather_with_pcie.cc"
#undef main

#include <cassert>
#include <cstdio>

static int async_calls = 0;
static int sync_copy_calls = 0;
static int stream_waits = 0;
static int failure_stage = 0;
static aclrtStream expected_stream = reinterpret_cast<aclrtStream>(0x1234);
static char device_data[64], host_data[64];

extern "C" aclError __wrap_aclrtSetDevice(int32_t) {
    if (failure_stage == 1) { g_stop.store(true); return ACL_ERROR_INVALID_PARAM; }
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind) {
    ++sync_copy_calls;
    g_stop.store(true);
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtMemcpyAsync(void* dst, size_t capacity, const void* src,
                                           size_t bytes, aclrtMemcpyKind kind, aclrtStream stream) {
    assert(dst == host_data && src == device_data && capacity == 64 && bytes == 64);
    assert(kind == ACL_MEMCPY_DEVICE_TO_HOST && stream == expected_stream);
    ++async_calls;
    if (failure_stage == 2) { g_stop.store(true); return ACL_ERROR_INVALID_PARAM; }
    return ACL_SUCCESS;
}
extern "C" aclError __wrap_aclrtSynchronizeStream(aclrtStream stream) {
    assert(stream == expected_stream && async_calls == stream_waits + 1);
    ++stream_waits;
    g_stop.store(true);
    if (failure_stage == 3) return ACL_ERROR_INVALID_PARAM;
    return ACL_SUCCESS;
}

int main() {
    g_stop.store(false);
    g_current_iter.store(3);
    g_t0 = Clock::now();
    const std::vector<InterferenceRange> ranges{{3, 3}};
    InterferenceWorkerWithCv(0, device_data, host_data, expected_stream, 64, ranges);
    if (async_calls != 1 || stream_waits != 1 || sync_copy_calls != 0) {
        std::fprintf(stderr, "FAIL: background D2H must use its dedicated async stream and drain each copy\n");
        return 1;
    }
    assert(g_interference_times.size() == 1);
    assert(g_interference_times[0].iter == 3);
    assert(g_interference_times[0].end_ms >= g_interference_times[0].start_ms);
    std::puts("PASS: production interference worker uses dedicated async D2H and waits for completion");
    for (int stage = 1; stage <= 3; ++stage) {
        failure_stage = stage;
        async_calls = stream_waits = sync_copy_calls = 0;
        g_interference_times.clear();
        g_stop.store(false);
        const int status = InterferenceWorkerWithCv(0, device_data, host_data, expected_stream, 64, ranges);
        if (status != ACL_ERROR_INVALID_PARAM || !g_interference_times.empty()) {
            std::fprintf(stderr, "FAIL: failed interference stage %d must propagate error without recording success\n", stage);
            return 1;
        }
    }
    std::puts("PASS: device selection, async copy and stream wait errors propagate without false success");
    failure_stage = 0;
    async_calls = stream_waits = sync_copy_calls = 0;
    g_interference_times.clear();
    g_stop.store(false);
    g_current_iter.store(UINT32_MAX);  // No formal iteration has started.
    const std::vector<InterferenceRange> including_zero{{0, UINT32_MAX}};
    std::thread worker([&]() {
        InterferenceWorkerWithCv(0, device_data, host_data, expected_stream, 64, including_zero);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    g_stop.store(true);
    g_interference_sleep_cv.notify_all();
    worker.join();
    if (async_calls != 0 || stream_waits != 0 || !g_interference_times.empty()) {
        std::fprintf(stderr, "FAIL: warmup must not start interference even when ranges include iteration zero\n");
        return 1;
    }
    std::puts("PASS: no background traffic before formal iterations, including ranges starting at zero");
}
