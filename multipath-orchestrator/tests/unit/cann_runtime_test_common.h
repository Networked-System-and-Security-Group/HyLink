#ifndef ADAPTIVE_CCL_TESTS_UNIT_CANN_RUNTIME_TEST_COMMON_H_
#define ADAPTIVE_CCL_TESTS_UNIT_CANN_RUNTIME_TEST_COMMON_H_

#include <acl/acl.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cann_test {

constexpr int kSkipUnsupported = 77;

struct Options {
    int32_t device = 0;
    size_t bytes = 64ULL * 1024 * 1024;
    int iterations = 32;
};

inline void PrintUsage(const char* program) {
    std::printf("Usage: %s [--device <id>] [--bytes <bytes>] [--iterations <count>]\n",
                program);
}

inline bool ParseUnsigned(const char* text, uint64_t* value) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

inline bool ParseOptions(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage(argv[0]);
            return false;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", argv[i]);
            PrintUsage(argv[0]);
            return false;
        }
        uint64_t value = 0;
        if (!ParseUnsigned(argv[++i], &value)) {
            std::fprintf(stderr, "Invalid numeric value: %s\n", argv[i]);
            return false;
        }
        if (std::strcmp(argv[i - 1], "--device") == 0 && value <= INT32_MAX) {
            options->device = static_cast<int32_t>(value);
        } else if (std::strcmp(argv[i - 1], "--bytes") == 0 && value > 0) {
            options->bytes = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i - 1], "--iterations") == 0 && value > 0 && value <= INT32_MAX) {
            options->iterations = static_cast<int>(value);
        } else {
            std::fprintf(stderr, "Invalid option or value: %s %s\n", argv[i - 1], argv[i]);
            PrintUsage(argv[0]);
            return false;
        }
    }
    return true;
}

inline bool Check(aclError result, const char* expression, const char* file, int line) {
    if (result == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ACL error %d from %s at %s:%d\n", result, expression, file, line);
    return false;
}

inline aclError EnqueueD2DWork(aclrtStream stream, void* dst, void* src, size_t bytes, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        const aclError result = aclrtMemcpyAsync(dst, bytes, src, bytes,
                                                  ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (result != ACL_SUCCESS) {
            return result;
        }
    }
    return ACL_SUCCESS;
}

inline float ElapsedMs(aclrtEvent start, aclrtEvent end) {
    float milliseconds = 0.0F;
    const aclError result = aclrtEventElapsedTime(&milliseconds, start, end);
    if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtEventElapsedTime failed: %d\n", result);
        return -1.0F;
    }
    return milliseconds;
}

inline double SteadyNowMs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

}  // namespace cann_test

#endif  // ADAPTIVE_CCL_TESTS_UNIT_CANN_RUNTIME_TEST_COMMON_H_
