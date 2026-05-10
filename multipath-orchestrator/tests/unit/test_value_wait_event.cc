#include "cann_runtime_test_common.h"

#include <dlfcn.h>

namespace {

using ValueOp = aclError (*)(void*, uint64_t, uint32_t, aclrtStream);
constexpr uint32_t kValueWaitGeq = ACL_STREAM_WAIT_VALUE_GEQ;

bool IsUnsupported(aclError result) {
    return result == ACL_ERROR_API_NOT_SUPPORT || result == ACL_ERROR_FEATURE_UNSUPPORTED ||
           result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
}

bool IsHelpRequest(int argc, char** argv) {
    return argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0);
}

void DestroyEvent(aclrtEvent* event) {
    if (*event != nullptr) {
        (void)aclrtDestroyEvent(*event);
        *event = nullptr;
    }
}

void DestroyStream(aclrtStream* stream) {
    if (*stream != nullptr) {
        (void)aclrtDestroyStream(*stream);
        *stream = nullptr;
    }
}

void FreeDevice(void** pointer) {
    if (*pointer != nullptr) {
        (void)aclrtFree(*pointer);
        *pointer = nullptr;
    }
}

}  // namespace

int main(int argc, char** argv) {
    cann_test::Options options;
    if (!cann_test::ParseOptions(argc, argv, &options)) {
        return IsHelpRequest(argc, argv) ? 0 : 2;
    }

    aclrtStream producer = nullptr;
    aclrtStream consumer = nullptr;
    aclrtEvent producer_start = nullptr;
    aclrtEvent producer_done = nullptr;
    aclrtEvent consumer_start = nullptr;
    aclrtEvent consumer_end = nullptr;
    void* source = nullptr;
    void* destination = nullptr;
    void* value_word = nullptr;
    void* ascendcl = nullptr;
    ValueOp wait_value = nullptr;
    ValueOp write_value = nullptr;
    aclError result = ACL_SUCCESS;
    int status = 1;
    bool initialized = false;

    if (!cann_test::Check(aclInit(nullptr), "aclInit(nullptr)", __FILE__, __LINE__)) {
        goto cleanup;
    }
    initialized = true;
    if (!cann_test::Check(aclrtSetDevice(options.device), "aclrtSetDevice(options.device)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateStream(&producer), "aclrtCreateStream(&producer)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateStream(&consumer), "aclrtCreateStream(&consumer)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateEvent(&producer_start), "aclrtCreateEvent(&producer_start)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateEvent(&producer_done), "aclrtCreateEvent(&producer_done)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateEvent(&consumer_start), "aclrtCreateEvent(&consumer_start)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtCreateEvent(&consumer_end), "aclrtCreateEvent(&consumer_end)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMalloc(&source, options.bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(&source)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMalloc(&destination, options.bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(&destination)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMalloc(&value_word, sizeof(uint64_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(&value_word)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtMemset(value_word, sizeof(uint64_t), 0, sizeof(uint64_t)), "aclrtMemset(value_word)", __FILE__, __LINE__)) {
        goto cleanup;
    }

    ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    wait_value = reinterpret_cast<ValueOp>(dlsym(ascendcl != nullptr ? ascendcl : RTLD_DEFAULT,
                                                 "aclrtValueWait"));
    write_value = reinterpret_cast<ValueOp>(dlsym(ascendcl != nullptr ? ascendcl : RTLD_DEFAULT,
                                                  "aclrtValueWrite"));
    if (wait_value == nullptr || write_value == nullptr) {
        std::printf("SKIP: aclrtValueWait/aclrtValueWrite is not exported by this CANN Runtime\n");
        status = cann_test::kSkipUnsupported;
        goto cleanup;
    }

    if (!cann_test::Check(aclrtRecordEvent(producer_start, producer), "aclrtRecordEvent(producer_start, producer)", __FILE__, __LINE__) ||
        !cann_test::Check(cann_test::EnqueueD2DWork(producer, destination, source, options.bytes, options.iterations), "EnqueueD2DWork(producer)", __FILE__, __LINE__)) {
        goto cleanup;
    }
    result = write_value(value_word, 1, 0, producer);
    if (IsUnsupported(result)) {
        std::printf("SKIP: aclrtValueWrite is unsupported on device %d (error %d)\n", options.device, result);
        status = cann_test::kSkipUnsupported;
        goto cleanup;
    }
    if (!cann_test::Check(result, "aclrtValueWrite(value_word, 1, 0, producer)", __FILE__, __LINE__) ||
        !cann_test::Check(write_value(value_word, 2, 0, producer), "aclrtValueWrite(value_word, 2, 0, producer)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtRecordEvent(producer_done, producer), "aclrtRecordEvent(producer_done, producer)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtRecordEvent(consumer_start, consumer), "aclrtRecordEvent(consumer_start, consumer)", __FILE__, __LINE__)) {
        goto cleanup;
    }
    result = wait_value(value_word, 1, kValueWaitGeq, consumer);
    if (IsUnsupported(result)) {
        std::printf("SKIP: aclrtValueWait is unsupported on device %d (error %d)\n", options.device, result);
        status = cann_test::kSkipUnsupported;
        goto cleanup;
    }
    if (!cann_test::Check(result, "aclrtValueWait(value_word, 1, GEQ, consumer)", __FILE__, __LINE__) ||
        !cann_test::Check(aclrtRecordEvent(consumer_end, consumer), "aclrtRecordEvent(consumer_end, consumer)", __FILE__, __LINE__)) {
        goto cleanup;
    }

    {
        const double host_start_ms = cann_test::SteadyNowMs();
        if (!cann_test::Check(aclrtSynchronizeStream(consumer), "aclrtSynchronizeStream(consumer)", __FILE__, __LINE__) ||
            !cann_test::Check(aclrtSynchronizeStream(producer), "aclrtSynchronizeStream(producer)", __FILE__, __LINE__)) {
            goto cleanup;
        }
        const double host_wait_ms = cann_test::SteadyNowMs() - host_start_ms;
        aclrtEventRecordedStatus producer_status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclrtEventRecordedStatus consumer_status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        if (!cann_test::Check(aclrtQueryEventStatus(producer_done, &producer_status), "aclrtQueryEventStatus(producer_done)", __FILE__, __LINE__) ||
            !cann_test::Check(aclrtQueryEventStatus(consumer_end, &consumer_status), "aclrtQueryEventStatus(consumer_end)", __FILE__, __LINE__) ||
            producer_status != ACL_EVENT_RECORDED_STATUS_COMPLETE ||
            consumer_status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            std::fprintf(stderr, "Event completion ordering check failed\n");
            goto cleanup;
        }
        const float producer_ms = cann_test::ElapsedMs(producer_start, producer_done);
        const float consumer_ms = cann_test::ElapsedMs(consumer_start, consumer_end);
        if (producer_ms < 0.0F || consumer_ms < 0.0F) {
            goto cleanup;
        }
        std::printf("ValueWait/ValueWrite event timing (device=%d, bytes=%zu, iterations=%d)\n",
                    options.device, options.bytes, options.iterations);
        std::printf("producer_ms=%.3f\nconsumer_event_ms=%.3f\nconsumer_host_wait_ms=%.3f\n",
                    producer_ms, consumer_ms, host_wait_ms);
    }
    status = 0;

cleanup:
    if (ascendcl != nullptr) {
        (void)dlclose(ascendcl);
    }
    DestroyEvent(&consumer_end);
    DestroyEvent(&consumer_start);
    DestroyEvent(&producer_done);
    DestroyEvent(&producer_start);
    FreeDevice(&value_word);
    FreeDevice(&destination);
    FreeDevice(&source);
    DestroyStream(&consumer);
    DestroyStream(&producer);
    if (initialized) {
        (void)aclFinalize();
    }
    return status;
}
