#include "cann_runtime_test_common.h"

#include <dlfcn.h>

namespace {

using ValueOp = aclError (*)(void*, uint64_t, uint32_t, aclrtStream);

bool IsUnsupported(aclError result) {
    return result == ACL_ERROR_API_NOT_SUPPORT || result == ACL_ERROR_FEATURE_UNSUPPORTED ||
           result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
}

bool IsHelpRequest(int argc, char** argv) {
    return argc == 2 &&
           (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0);
}

void DestroyEvent(aclrtEvent* event) {
    if (*event) (void)aclrtDestroyEvent(*event);
    *event = nullptr;
}

void DestroyStream(aclrtStream* stream) {
    if (*stream) (void)aclrtDestroyStream(*stream);
    *stream = nullptr;
}

void FreeDevice(void** pointer) {
    if (*pointer) (void)aclrtFree(*pointer);
    *pointer = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    cann_test::Options options;
    if (!cann_test::ParseOptions(argc, argv, &options)) return IsHelpRequest(argc, argv) ? 0 : 2;

    constexpr size_t kRingSize = 2;
    constexpr uint64_t kIterations = 6;
    aclrtStream producer = nullptr;
    aclrtStream consumer_a = nullptr;
    aclrtStream consumer_b = nullptr;
    aclrtEvent done_a = nullptr;
    aclrtEvent done_b = nullptr;
    void* source = nullptr;
    void* destination = nullptr;
    void* device_words = nullptr;
    void* ascendcl = nullptr;
    ValueOp write_value = nullptr;
    ValueOp wait_value = nullptr;
    uint64_t* lane_a = nullptr;
    uint64_t* lane_b = nullptr;
    aclrtEventRecordedStatus state_a = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    aclrtEventRecordedStatus state_b = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    bool initialized = false;
    int status = 1;

    if (!cann_test::Check(aclInit(nullptr), "aclInit(nullptr)", __FILE__, __LINE__)) goto cleanup;
    initialized = true;
    if (!cann_test::Check(aclrtSetDevice(options.device), "aclrtSetDevice", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateStream(&producer), "aclrtCreateStream(producer)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateStream(&consumer_a), "aclrtCreateStream(consumer_a)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateStream(&consumer_b), "aclrtCreateStream(consumer_b)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateEvent(&done_a), "aclrtCreateEvent(done_a)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateEvent(&done_b), "aclrtCreateEvent(done_b)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMalloc(&source, options.bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(source)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMalloc(&destination, options.bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(destination)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMalloc(&device_words, 2 * kRingSize * sizeof(uint64_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(words)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMemset(device_words, 2 * kRingSize * sizeof(uint64_t), 0, 2 * kRingSize * sizeof(uint64_t)), "aclrtMemset(words)", __FILE__, __LINE__)) goto cleanup;

    ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    write_value = reinterpret_cast<ValueOp>(dlsym(ascendcl ? ascendcl : RTLD_DEFAULT, "aclrtValueWrite"));
    wait_value = reinterpret_cast<ValueOp>(dlsym(ascendcl ? ascendcl : RTLD_DEFAULT, "aclrtValueWait"));
    if (!write_value || !wait_value) {
        std::printf("SKIP: aclrtValueWait/aclrtValueWrite is not exported\n");
        status = cann_test::kSkipUnsupported;
        goto cleanup;
    }
    lane_a = static_cast<uint64_t*>(device_words);
    lane_b = lane_a + kRingSize;
    for (uint64_t seq = 1; seq <= kIterations; ++seq) {
        aclError result = cann_test::EnqueueD2DWork(producer, destination, source, options.bytes, options.iterations);
        if (!cann_test::Check(result, "EnqueueD2DWork", __FILE__, __LINE__)) goto cleanup;
        result = write_value(&lane_a[seq % kRingSize], seq, 0, producer);
        if (IsUnsupported(result)) { status = cann_test::kSkipUnsupported; goto cleanup; }
        if (!cann_test::Check(result, "aclrtValueWrite(lane_a)", __FILE__, __LINE__) ||
            !cann_test::Check(wait_value(&lane_a[seq % kRingSize], seq, ACL_STREAM_WAIT_VALUE_GEQ, consumer_a), "aclrtValueWait(lane_a, GEQ)", __FILE__, __LINE__) ||
            !cann_test::Check(write_value(&lane_b[seq % kRingSize], seq, 0, producer), "aclrtValueWrite(lane_b)", __FILE__, __LINE__) ||
            !cann_test::Check(wait_value(&lane_b[seq % kRingSize], seq, ACL_STREAM_WAIT_VALUE_GEQ, consumer_b), "aclrtValueWait(lane_b, GEQ)", __FILE__, __LINE__)) goto cleanup;
    }
    if (!cann_test::Check(aclrtRecordEvent(done_a, consumer_a), "aclrtRecordEvent(done_a)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtRecordEvent(done_b, consumer_b), "aclrtRecordEvent(done_b)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtSynchronizeStream(consumer_a), "aclrtSynchronizeStream(consumer_a)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtSynchronizeStream(consumer_b), "aclrtSynchronizeStream(consumer_b)", __FILE__, __LINE__)) goto cleanup;
    if (!cann_test::Check(aclrtQueryEventStatus(done_a, &state_a), "aclrtQueryEventStatus(done_a)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtQueryEventStatus(done_b, &state_b), "aclrtQueryEventStatus(done_b)", __FILE__, __LINE__) ||
        state_a != ACL_EVENT_RECORDED_STATUS_COMPLETE || state_b != ACL_EVENT_RECORDED_STATUS_COMPLETE) goto cleanup;
    std::printf("ValueWait ring: ring_size=%zu iterations=%llu consumer_a=%d consumer_b=%d\n", kRingSize, static_cast<unsigned long long>(kIterations), static_cast<int>(state_a), static_cast<int>(state_b));
    status = 0;
cleanup:
    if (ascendcl) (void)dlclose(ascendcl);
    DestroyEvent(&done_b); DestroyEvent(&done_a);
    FreeDevice(&device_words); FreeDevice(&destination); FreeDevice(&source);
    DestroyStream(&consumer_b); DestroyStream(&consumer_a); DestroyStream(&producer);
    if (initialized) (void)aclFinalize();
    return status;
}
