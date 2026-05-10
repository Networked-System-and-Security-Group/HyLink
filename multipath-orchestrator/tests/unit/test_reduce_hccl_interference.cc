#include <acl/acl.h>
#include <hccl/hccl.h>

#include "pccl_aicore_reduce.h"

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kBackgroundBurst = 10;

enum class ReduceImplementation { kAicore, kAclrt };

enum class BackgroundOp { kAllGather, kReduceScatter, kAllReduce };

struct Options {
    int reduce_device = 0;
    size_t reduce_bytes = 1024 * 1024;
    int reduce_iters = 32;
    int background_start = 8;
    BackgroundOp background_op = BackgroundOp::kReduceScatter;
    int background_ranks = 4;
    size_t background_bytes = 1024 * 1024;
    ReduceImplementation reduce_impl = ReduceImplementation::kAicore;
};

struct BufferLayout {
    size_t send_bytes = 0;
    size_t recv_bytes = 0;
};

struct RankContext {
    int device = 0;
    HcclComm comm = nullptr;
    aclrtStream hccl_stream = nullptr;
    aclrtEvent hccl_start = nullptr;
    aclrtEvent hccl_end = nullptr;
    void* hccl_send = nullptr;
    void* hccl_recv = nullptr;
    aclrtStream reduce_stream = nullptr;
    aclrtEvent reduce_start = nullptr;
    aclrtEvent reduce_end = nullptr;
    void* reduce_input = nullptr;
    void* reduce_output = nullptr;
};

void PrintUsage(const char* program) {
    std::printf(
        "Usage: %s [options]\n"
        "  --reduce-device <id>                         default: 0\n"
        "  --reduce-bytes <bytes>                       default: 1048576\n"
        "  --reduce-iters <count>                       default: 32\n"
        "  --background-start <zero-based iteration>    default: 8\n"
        "  --background-op <allgather|reducescatter|allreduce> default: reducescatter\n"
        "  --background-ranks <2..8>                    default: 4 (devices 0..N-1)\n"
        "  --background-bytes <bytes>                   default: 1048576\n"
        "  --reduce-impl <aicore|aclrt>                 default: aicore\n",
        program);
}

bool ParseUnsigned(const char* text, uint64_t* value) {
    if (text == nullptr || text[0] == '\0') return false;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == nullptr || *end != '\0') return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage(argv[0]);
            return false;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", argv[i]);
            return false;
        }
        const char* option = argv[i++];
        const char* value = argv[i];
        uint64_t parsed = 0;
        if (std::strcmp(option, "--background-op") == 0) {
            if (std::strcmp(value, "allgather") == 0) options->background_op = BackgroundOp::kAllGather;
            else if (std::strcmp(value, "reducescatter") == 0) options->background_op = BackgroundOp::kReduceScatter;
            else if (std::strcmp(value, "allreduce") == 0) options->background_op = BackgroundOp::kAllReduce;
            else {
                std::fprintf(stderr, "Invalid --background-op: %s\n", value);
                return false;
            }
            continue;
        }
        if (std::strcmp(option, "--reduce-impl") == 0) {
            if (std::strcmp(value, "aicore") == 0) options->reduce_impl = ReduceImplementation::kAicore;
            else if (std::strcmp(value, "aclrt") == 0) options->reduce_impl = ReduceImplementation::kAclrt;
            else {
                std::fprintf(stderr, "Invalid --reduce-impl: %s\n", value);
                return false;
            }
            continue;
        }
        if (!ParseUnsigned(value, &parsed)) {
            std::fprintf(stderr, "Invalid numeric value: %s\n", value);
            return false;
        }
        if (std::strcmp(option, "--reduce-device") == 0 && parsed <= INT32_MAX) {
            options->reduce_device = static_cast<int>(parsed);
        } else if (std::strcmp(option, "--reduce-bytes") == 0 && parsed > 0) {
            options->reduce_bytes = static_cast<size_t>(parsed);
        } else if (std::strcmp(option, "--reduce-iters") == 0 && parsed > 0 && parsed <= INT32_MAX) {
            options->reduce_iters = static_cast<int>(parsed);
        } else if (std::strcmp(option, "--background-start") == 0 && parsed <= INT32_MAX) {
            options->background_start = static_cast<int>(parsed);
        } else if (std::strcmp(option, "--background-ranks") == 0 && parsed >= 2 && parsed <= 8) {
            options->background_ranks = static_cast<int>(parsed);
        } else if (std::strcmp(option, "--background-bytes") == 0 && parsed > 0) {
            options->background_bytes = static_cast<size_t>(parsed);
        } else {
            std::fprintf(stderr, "Invalid option or value: %s %s\n", option, value);
            return false;
        }
    }
    return true;
}

const char* OpName(BackgroundOp op) {
    switch (op) {
        case BackgroundOp::kAllGather: return "allgather";
        case BackgroundOp::kReduceScatter: return "reducescatter";
        case BackgroundOp::kAllReduce: return "allreduce";
    }
    return "unknown";
}

aclError LaunchReduce(ReduceImplementation implementation, void* output, const void* input,
                      size_t bytes, aclrtStream stream) {
    if (implementation == ReduceImplementation::kAclrt) {
        return aclrtReduceAsync(output, input, bytes / sizeof(float),
                                ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_FLOAT, stream, nullptr);
    }
    return pcclAicoreReduceAsync(output, input, bytes, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM,
                                 ACL_FLOAT, stream);
}

const char* ReduceImplementationName(ReduceImplementation implementation) {
    return implementation == ReduceImplementation::kAclrt ? "aclrt" : "aicore";
}

bool GetLayout(const Options& options, BufferLayout* layout) {
    if (options.reduce_bytes % sizeof(float) != 0 || options.background_bytes % sizeof(float) != 0 ||
        options.reduce_device < 0 || options.reduce_device >= options.background_ranks ||
        options.background_start < 0 || options.background_start >= options.reduce_iters) {
        return false;
    }
    if (options.background_op == BackgroundOp::kAllReduce) {
        *layout = {options.background_bytes, options.background_bytes};
        return true;
    }
    if (options.background_bytes % static_cast<size_t>(options.background_ranks) != 0) return false;
    if (options.background_op == BackgroundOp::kAllGather) {
        *layout = {options.background_bytes / static_cast<size_t>(options.background_ranks), options.background_bytes};
    } else {
        *layout = {options.background_bytes, options.background_bytes / static_cast<size_t>(options.background_ranks)};
    }
    return true;
}

bool CheckAcl(aclError result, const char* call, int device, std::atomic<int>* failed) {
    if (result == ACL_SUCCESS) return true;
    std::fprintf(stderr, "device %d: ACL error %d from %s\n", device, result, call);
    failed->store(1);
    return false;
}

bool CheckHccl(HcclResult result, const char* call, int device, std::atomic<int>* failed) {
    if (result == HCCL_SUCCESS) return true;
    std::fprintf(stderr, "device %d: HCCL error %d from %s\n", device, result, call);
    failed->store(1);
    return false;
}

bool InitializeRank(RankContext* context, const Options& options, const BufferLayout& layout,
                    const HcclRootInfo* root_info, std::atomic<int>* failed) {
    const int device = context->device;
    if (!CheckAcl(aclrtSetDevice(device), "aclrtSetDevice", device, failed) ||
        !CheckHccl(HcclCommInitRootInfo(options.background_ranks, root_info, static_cast<uint32_t>(device), &context->comm),
                   "HcclCommInitRootInfo", device, failed) ||
        !CheckAcl(aclrtCreateStream(&context->hccl_stream), "aclrtCreateStream(hccl)", device, failed) ||
        !CheckAcl(aclrtCreateEvent(&context->hccl_start), "aclrtCreateEvent(hccl_start)", device, failed) ||
        !CheckAcl(aclrtCreateEvent(&context->hccl_end), "aclrtCreateEvent(hccl_end)", device, failed) ||
        !CheckAcl(aclrtMalloc(&context->hccl_send, layout.send_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(hccl_send)", device, failed) ||
        !CheckAcl(aclrtMalloc(&context->hccl_recv, layout.recv_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(hccl_recv)", device, failed)) {
        return false;
    }
    std::vector<float> host_send(layout.send_bytes / sizeof(float), static_cast<float>(device + 1));
    if (!CheckAcl(aclrtMemcpy(context->hccl_send, layout.send_bytes, host_send.data(), layout.send_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(hccl_send)", device, failed)) {
        return false;
    }
    if (device != options.reduce_device) return true;
    std::vector<float> host_reduce(options.reduce_bytes / sizeof(float), 1.0F);
    return CheckAcl(aclrtCreateStream(&context->reduce_stream), "aclrtCreateStream(reduce)", device, failed) &&
           CheckAcl(aclrtCreateEvent(&context->reduce_start), "aclrtCreateEvent(reduce_start)", device, failed) &&
           CheckAcl(aclrtCreateEvent(&context->reduce_end), "aclrtCreateEvent(reduce_end)", device, failed) &&
           CheckAcl(aclrtMalloc(&context->reduce_input, options.reduce_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(reduce_input)", device, failed) &&
           CheckAcl(aclrtMalloc(&context->reduce_output, options.reduce_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(reduce_output)", device, failed) &&
           CheckAcl(aclrtMemcpy(context->reduce_input, options.reduce_bytes, host_reduce.data(), options.reduce_bytes,
                                ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(reduce_input)", device, failed) &&
           CheckAcl(aclrtMemset(context->reduce_output, options.reduce_bytes, 0, options.reduce_bytes),
                    "aclrtMemset(reduce_output)", device, failed);
}

bool QueueCollective(RankContext* context, const Options& options, const BufferLayout& layout, std::atomic<int>* failed) {
    const uint64_t elements = options.background_bytes / sizeof(float);
    switch (options.background_op) {
        case BackgroundOp::kAllReduce:
            return CheckHccl(HcclAllReduce(context->hccl_send, context->hccl_recv, elements, HCCL_DATA_TYPE_FP32,
                                            HCCL_REDUCE_SUM, context->comm, context->hccl_stream),
                             "HcclAllReduce", context->device, failed);
        case BackgroundOp::kReduceScatter:
            return CheckHccl(HcclReduceScatter(context->hccl_send, context->hccl_recv,
                                                layout.recv_bytes / sizeof(float), HCCL_DATA_TYPE_FP32,
                                                HCCL_REDUCE_SUM, context->comm, context->hccl_stream),
                             "HcclReduceScatter", context->device, failed);
        case BackgroundOp::kAllGather:
            return CheckHccl(HcclAllGather(context->hccl_send, context->hccl_recv,
                                            layout.send_bytes / sizeof(float), HCCL_DATA_TYPE_FP32,
                                            context->comm, context->hccl_stream),
                             "HcclAllGather", context->device, failed);
    }
    return false;
}

void DestroyRank(RankContext* context) {
    (void)aclrtSetDevice(context->device);
    if (context->reduce_end) (void)aclrtDestroyEvent(context->reduce_end);
    if (context->reduce_start) (void)aclrtDestroyEvent(context->reduce_start);
    if (context->hccl_end) (void)aclrtDestroyEvent(context->hccl_end);
    if (context->hccl_start) (void)aclrtDestroyEvent(context->hccl_start);
    if (context->reduce_input) (void)aclrtFree(context->reduce_input);
    if (context->reduce_output) (void)aclrtFree(context->reduce_output);
    if (context->hccl_send) (void)aclrtFree(context->hccl_send);
    if (context->hccl_recv) (void)aclrtFree(context->hccl_recv);
    if (context->reduce_stream) (void)aclrtDestroyStream(context->reduce_stream);
    if (context->hccl_stream) (void)aclrtDestroyStream(context->hccl_stream);
    if (context->comm) (void)HcclCommDestroy(context->comm);
}

int RunExperiment(const Options& options) {
    BufferLayout layout;
    if (!GetLayout(options, &layout)) {
        std::fprintf(stderr, "Invalid configuration: check FP32 sizes, rank divisibility, target device, and 5-iteration window\n");
        return 2;
    }
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    uint32_t available = 0;
    if (aclrtGetDeviceCount(&available) != ACL_SUCCESS || available < static_cast<uint32_t>(options.background_ranks)) {
        std::fprintf(stderr, "Need %d visible NPUs (devices 0..%d); found %u\n",
                     options.background_ranks, options.background_ranks - 1, available);
        (void)aclFinalize();
        return 1;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(0) failed\n");
        (void)aclFinalize();
        return 1;
    }
    HcclRootInfo* root_info = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void**>(&root_info), sizeof(HcclRootInfo)) != ACL_SUCCESS ||
        HcclGetRootInfo(root_info) != HCCL_SUCCESS) {
        std::fprintf(stderr, "HCCL root-info initialization failed\n");
        if (root_info) (void)aclrtFreeHost(root_info);
        (void)aclFinalize();
        return 1;
    }

    std::printf("Reduce/HCCL interference: reduce_impl=%s reduce_device=%d reduce_bytes=%zu reduce_iters=%d ",
                ReduceImplementationName(options.reduce_impl), options.reduce_device,
                options.reduce_bytes, options.reduce_iters);
    std::printf("background_op=%s background_ranks=%d background_bytes=%zu background_start=%d background_iters=%d\n",
                OpName(options.background_op), options.background_ranks, options.background_bytes,
                options.background_start, kBackgroundBurst);
    std::printf("background_per_rank_send_bytes=%zu background_per_rank_recv_bytes=%zu\n",
                layout.send_bytes, layout.recv_bytes);

    std::vector<RankContext> contexts(static_cast<size_t>(options.background_ranks));
    for (int rank = 0; rank < options.background_ranks; ++rank) contexts[rank].device = rank;
    pthread_barrier_t initialized;
    pthread_barrier_t start;
    pthread_barrier_t iteration;
    pthread_barrier_t initial_collective;
    pthread_barrier_t queued;
    pthread_barrier_init(&initialized, nullptr, static_cast<unsigned>(options.background_ranks + 1));
    pthread_barrier_init(&start, nullptr, static_cast<unsigned>(options.background_ranks + 1));
    pthread_barrier_init(&iteration, nullptr, static_cast<unsigned>(options.background_ranks));
    pthread_barrier_init(&initial_collective, nullptr, static_cast<unsigned>(options.background_ranks));
    pthread_barrier_init(&queued, nullptr, static_cast<unsigned>(options.background_ranks + 1));
    std::atomic<int> failed{0};

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(options.background_ranks));
    for (int rank = 0; rank < options.background_ranks; ++rank) {
        workers.emplace_back([&, rank] {
            RankContext& context = contexts[rank];
            const bool initialized_ok = InitializeRank(&context, options, layout, root_info, &failed);
            pthread_barrier_wait(&initialized);
            pthread_barrier_wait(&start);
            if (!initialized_ok || failed.load() != 0) {
                pthread_barrier_wait(&queued);
                return;
            }
            (void)QueueCollective(&context, options, layout, &failed);
            pthread_barrier_wait(&initial_collective);
            if (context.device == options.reduce_device &&
                !CheckAcl(aclrtRecordEvent(context.reduce_start, context.reduce_stream), "aclrtRecordEvent(reduce_start)", context.device, &failed)) {
                failed.store(1);
            }
            for (int iteration_index = 0; iteration_index < options.reduce_iters; ++iteration_index) {
                if (context.device == options.reduce_device) {
                    (void)CheckAcl(LaunchReduce(options.reduce_impl, context.reduce_output,
                                                context.reduce_input, options.reduce_bytes,
                                                context.reduce_stream),
                                   "reduce dispatch", context.device, &failed);
                }
                if (iteration_index == options.background_start) {
                    (void)CheckAcl(aclrtRecordEvent(context.hccl_start, context.hccl_stream),
                                   "aclrtRecordEvent(hccl_start)", context.device, &failed);
                }
                if (iteration_index == options.background_start) {
                    for (int burst = 0; burst < kBackgroundBurst; ++burst) {
                        (void)QueueCollective(&context, options, layout, &failed);
                    }
                }
                if (iteration_index == options.background_start) {
                    (void)CheckAcl(aclrtRecordEvent(context.hccl_end, context.hccl_stream),
                                   "aclrtRecordEvent(hccl_end)", context.device, &failed);
                }
                pthread_barrier_wait(&iteration);
            }
            if (context.device == options.reduce_device) {
                (void)CheckAcl(aclrtRecordEvent(context.reduce_end, context.reduce_stream), "aclrtRecordEvent(reduce_end)", context.device, &failed);
            }
            pthread_barrier_wait(&queued);
            (void)CheckAcl(aclrtSynchronizeStream(context.hccl_stream), "aclrtSynchronizeStream(hccl)", context.device, &failed);
            if (context.device == options.reduce_device) {
                (void)CheckAcl(aclrtSynchronizeStream(context.reduce_stream), "aclrtSynchronizeStream(reduce)", context.device, &failed);
            }
        });
    }
    pthread_barrier_wait(&initialized);
    const double host_start_ms = [] {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::milli>(now).count();
    }();
    pthread_barrier_wait(&start);
    pthread_barrier_wait(&queued);
    for (auto& worker : workers) worker.join();
    const double host_end_ms = [] {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::milli>(now).count();
    }();

    if (failed.load() == 0) {
        for (const RankContext& context : contexts) {
            float hccl_ms = 0.0F;
            if (aclrtEventElapsedTime(&hccl_ms, context.hccl_start, context.hccl_end) != ACL_SUCCESS) {
                failed.store(1);
                break;
            }
            std::printf("hccl_rank_%d_ms=%.3f\n", context.device, hccl_ms);
        }
        const RankContext& target = contexts[options.reduce_device];
        float reduce_ms = 0.0F;
        if (failed.load() == 0 && aclrtEventElapsedTime(&reduce_ms, target.reduce_start, target.reduce_end) == ACL_SUCCESS) {
            std::printf("reduce_stream_ms=%.3f\ntotal_host_ms=%.3f\n", reduce_ms, host_end_ms - host_start_ms);
        } else {
            failed.store(1);
        }
    }

    for (RankContext& context : contexts) DestroyRank(&context);
    pthread_barrier_destroy(&queued);
    pthread_barrier_destroy(&initial_collective);
    pthread_barrier_destroy(&iteration);
    pthread_barrier_destroy(&start);
    pthread_barrier_destroy(&initialized);
    (void)aclrtSetDevice(0);
    (void)aclrtFreeHost(root_info);
    (void)aclFinalize();
    return failed.load() == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        return argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) ? 0 : 2;
    }
    return RunExperiment(options);
}
