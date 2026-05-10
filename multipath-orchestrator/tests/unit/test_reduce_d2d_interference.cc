#include <acl/acl.h>

#include "pccl_aicore_reduce.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <pthread.h>

namespace {

constexpr int kD2dBurst = 10;

enum class ReduceImplementation { kAicore, kAclrt };

struct Config {
    int ranks = 2;
    int reduce_iters = 32;
    int d2d_start = 8;
    size_t reduce_bytes = 1 << 20;
    size_t d2d_bytes = 1 << 20;
    bool reverse = false;
    ReduceImplementation reduce_impl = ReduceImplementation::kAicore;
};

struct Channel {
    int source = 0;
    int destination = 0;
    aclrtStream stream = nullptr;
    void* source_buffer = nullptr;
    void* destination_buffer = nullptr;
};

bool Check(aclError result, const char* name) {
    if (result == ACL_SUCCESS) return true;
    std::fprintf(stderr, "%s failed: %d\n", name, result);
    return false;
}

bool Parse(int argc, char** argv, Config* config) {
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) return false;
        const char* key = argv[i];
        const char* value = argv[i + 1];
        if (std::strcmp(key, "--d2d-ranks") == 0) config->ranks = std::atoi(value);
        else if (std::strcmp(key, "--reduce-iters") == 0) config->reduce_iters = std::atoi(value);
        else if (std::strcmp(key, "--d2d-start") == 0) config->d2d_start = std::atoi(value);
        else if (std::strcmp(key, "--reduce-bytes") == 0) config->reduce_bytes = std::strtoull(value, nullptr, 10);
        else if (std::strcmp(key, "--d2d-bytes") == 0) config->d2d_bytes = std::strtoull(value, nullptr, 10);
        else if (std::strcmp(key, "--d2d-direction") == 0) config->reverse = std::strcmp(value, "reverse") == 0;
        else if (std::strcmp(key, "--reduce-impl") == 0) {
            if (std::strcmp(value, "aicore") == 0) config->reduce_impl = ReduceImplementation::kAicore;
            else if (std::strcmp(value, "aclrt") == 0) config->reduce_impl = ReduceImplementation::kAclrt;
            else return false;
        }
        else return false;
    }
    return config->ranks >= 1 && config->ranks <= 8 && config->reduce_iters > 0 &&
           config->d2d_start >= 0 && config->d2d_start < config->reduce_iters &&
           config->reduce_bytes > 0 && config->reduce_bytes % sizeof(float) == 0 && config->d2d_bytes > 0;
}

void Usage(const char* program) {
    std::printf("Usage: %s --d2d-ranks N --reduce-iters N --d2d-start N "
                "--reduce-bytes BYTES --d2d-bytes BYTES "
                "[--d2d-direction reverse] [--reduce-impl aicore|aclrt]\n", program);
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

bool EnablePeers(int ranks) {
    for (int device = 1; device < ranks; ++device) {
        int32_t can_access = 0;
        if (!Check(aclrtSetDevice(0), "set peer source device") ||
            !Check(aclrtDeviceCanAccessPeer(&can_access, 0, device), "query peer 0->peer") || !can_access ||
            !Check(aclrtDeviceEnablePeerAccess(device, 0), "enable peer 0->peer")) return false;
        if (!Check(aclrtSetDevice(device), "set peer destination device") ||
            !Check(aclrtDeviceCanAccessPeer(&can_access, device, 0), "query peer peer->0") || !can_access ||
            !Check(aclrtDeviceEnablePeerAccess(0, 0), "enable peer peer->0")) return false;
    }
    return true;
}

bool CreateChannel(const Config& config, int index, Channel* channel) {
    channel->source = config.reverse ? index : 0;
    channel->destination = config.reverse ? 0 : index;
    if (!Check(aclrtSetDevice(channel->source), "set source device") ||
        !Check(aclrtCreateStream(&channel->stream), "create D2D stream") ||
        !Check(aclrtMalloc(&channel->source_buffer, config.d2d_bytes, ACL_MEM_MALLOC_NORMAL_ONLY_P2P), "allocate source")) return false;
    if (!Check(aclrtSetDevice(channel->destination), "set destination device") ||
        !Check(aclrtMalloc(&channel->destination_buffer, config.d2d_bytes, ACL_MEM_MALLOC_NORMAL_ONLY_P2P), "allocate destination")) return false;
    return true;
}

void DestroyChannel(Channel* channel) {
    aclrtSetDevice(channel->source);
    if (channel->source_buffer) aclrtFree(channel->source_buffer);
    if (channel->stream) aclrtDestroyStream(channel->stream);
    aclrtSetDevice(channel->destination);
    if (channel->destination_buffer) aclrtFree(channel->destination_buffer);
}

}  // namespace

int main(int argc, char** argv) {
    Config config;
    if (!Parse(argc, argv, &config)) {
        Usage(argv[0]);
        return 2;
    }
    if (!Check(aclInit(nullptr), "aclInit")) return 1;
    uint32_t available = 0;
    bool ok = Check(aclrtGetDeviceCount(&available), "aclrtGetDeviceCount") &&
              available >= static_cast<uint32_t>(config.ranks) && EnablePeers(config.ranks);

    std::vector<Channel> channels(config.ranks);
    void* reduce_input = nullptr;
    void* reduce_output = nullptr;
    aclrtStream reduce_stream = nullptr;
    std::vector<float> reduce_input_host;
    std::vector<float> reduce_output_host;
    std::vector<float> reduce_result_host;
    if (ok) {
        ok = Check(aclrtSetDevice(0), "set reduce device") &&
             Check(aclrtCreateStream(&reduce_stream), "create reduce stream") &&
             Check(aclrtMalloc(&reduce_input, config.reduce_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate reduce input") &&
             Check(aclrtMalloc(&reduce_output, config.reduce_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate reduce output");
        if (ok) {
            const size_t reduce_elements = config.reduce_bytes / sizeof(float);
            reduce_input_host.assign(reduce_elements, 1.0f);
            reduce_output_host.assign(reduce_elements, 2.0f);
            reduce_result_host.resize(reduce_elements);
            ok = Check(aclrtMemcpy(reduce_input, config.reduce_bytes, reduce_input_host.data(), config.reduce_bytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE), "initialize reduce input") &&
                 Check(aclrtMemcpy(reduce_output, config.reduce_bytes, reduce_output_host.data(), config.reduce_bytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE), "initialize reduce output");
        }
    }
    for (int i = 0; ok && i < config.ranks; ++i) ok = CreateChannel(config, i, &channels[i]);

    pthread_barrier_t initial_copy_done;
    pthread_barrier_init(&initial_copy_done, nullptr, static_cast<unsigned>(config.ranks));
    const char* preload = std::getenv("LD_PRELOAD");
    std::printf("reduce_impl=%s preload=%s\n", ReduceImplementationName(config.reduce_impl),
                preload == nullptr ? "none" : preload);
    const auto workload_start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int rank = 0; ok && rank < config.ranks; ++rank) {
        workers.emplace_back([&, rank] {
            Channel& channel = channels[rank];
            Check(aclrtSetDevice(channel.source), "set initial D2D device");
            Check(aclrtMemcpyAsync(channel.destination_buffer, config.d2d_bytes,
                                   channel.source_buffer, config.d2d_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, channel.stream),
                  "initial D2D copy");
            Check(aclrtSynchronizeStream(channel.stream), "synchronize initial D2D copy");
            pthread_barrier_wait(&initial_copy_done);
            for (int iteration = 0; iteration < config.reduce_iters; ++iteration) {
                if (rank == 0) {
                    Check(aclrtSetDevice(0), "set reduce device");
                    Check(LaunchReduce(config.reduce_impl, reduce_output, reduce_input,
                                       config.reduce_bytes, reduce_stream), "reduce dispatch");
                }
                if (iteration == config.d2d_start) {
                    Check(aclrtSetDevice(channel.source), "set D2D source device");
                    for (int burst = 0; burst < kD2dBurst; ++burst) {
                        Check(aclrtMemcpyAsync(channel.destination_buffer, config.d2d_bytes,
                                               channel.source_buffer, config.d2d_bytes,
                                               ACL_MEMCPY_DEVICE_TO_DEVICE, channel.stream),
                              "D2D burst copy");
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    pthread_barrier_destroy(&initial_copy_done);
    if (ok) {
        aclrtSetDevice(0);
        ok = Check(aclrtSynchronizeStream(reduce_stream), "sync reduce stream");
        if (ok) {
            ok = Check(aclrtMemcpy(reduce_result_host.data(), config.reduce_bytes, reduce_output,
                                   config.reduce_bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read reduce result");
        }
        if (ok) {
            const float expected = 2.0f + static_cast<float>(config.reduce_iters);
            for (size_t i = 0; i < reduce_result_host.size(); ++i) {
                if (std::fabs(reduce_result_host[i] - expected) > 1e-6f) {
                    std::fprintf(stderr, "reduce result mismatch at %zu: got %.6f expected %.6f\n",
                                 i, reduce_result_host[i], expected);
                    ok = false;
                    break;
                }
            }
        }
        for (auto& channel : channels) {
            aclrtSetDevice(channel.source);
            ok = Check(aclrtSynchronizeStream(channel.stream), "sync D2D stream") && ok;
        }
    }
    const auto workload_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(workload_end - workload_start).count();
    std::printf("workload_elapsed_ms=%.3f\n", elapsed_ms);
    for (auto& channel : channels) DestroyChannel(&channel);
    aclrtSetDevice(0);
    if (reduce_input) aclrtFree(reduce_input);
    if (reduce_output) aclrtFree(reduce_output);
    if (reduce_stream) aclrtDestroyStream(reduce_stream);
    aclFinalize();
    return ok ? 0 : 1;
}
