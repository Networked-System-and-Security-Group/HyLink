#include <mpi.h>

#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if ((ret) != ACL_SUCCESS) {                                                            \
            std::fprintf(stderr, "acl interface return err %s:%d, retcode: %d\n",              \
                         __FILE__, __LINE__, static_cast<int>(ret));                            \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                         \
    do {                                                                                       \
        if ((ret) != HCCL_SUCCESS) {                                                           \
            std::fprintf(stderr, "hccl interface return err %s:%d, retcode: %d\n",             \
                         __FILE__, __LINE__, static_cast<int>(ret));                            \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

enum class Op {
    ALLGATHER,
    REDUCESCATTER,
    ALLREDUCE,
    ALLTOALL,
};

struct Options {
    Op op = Op::ALLREDUCE;
    std::string opName = "ar";
    uint64_t size = 1024;
    uint32_t iters = 1;
    uint32_t warmup = 6;
    int device = -1;
    std::string pcieRangesStr;
    std::vector<std::pair<uint32_t, uint32_t>> pcieRanges;
    size_t pcieSizeBytes = 0;
    std::string pcieDirection = "d2h";
    bool reuseInput = false;
    bool help = false;
};

struct PcieStageMetrics {
    uint64_t attempts = 0;
    uint64_t successes = 0;
    double copy_seconds = 0.0;
    std::chrono::steady_clock::time_point first_start{}, last_end{};
};

struct PcieWorkerState {
    void* device = nullptr;
    void* host = nullptr;
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
    std::atomic<int> status{ACL_SUCCESS};
    std::thread thread;
    std::vector<PcieStageMetrics> metrics;
};

static int RunPcieInterference(int device, void* device_buffer, void* host_buffer,
                              const Options& opt, std::atomic<bool>& stop,
                              std::atomic<uint32_t>& iteration, std::atomic<bool>& formal,
                              std::mutex& mutex, std::condition_variable& cv,
                              std::vector<PcieStageMetrics>& metrics,
                              aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST) {
    const aclError selection = aclrtSetDevice(device);
    if (selection != ACL_SUCCESS) return selection;
    while (!stop.load(std::memory_order_acquire)) {
        size_t active = opt.pcieRanges.size();
        if (formal.load(std::memory_order_acquire)) {
            const uint32_t iter = iteration.load(std::memory_order_acquire);
            for (size_t k = 0; k < opt.pcieRanges.size(); ++k) {
                const auto& range = opt.pcieRanges[k];
                if (range.first <= iter && iter < range.second) { active = k; break; }
            }
        }
        if (active != opt.pcieRanges.size()) {
            auto& stage = metrics[active];
            const auto begin = std::chrono::steady_clock::now();
            if (stage.attempts == 0) stage.first_start = begin;
            ++stage.attempts;
            void* dst = kind == ACL_MEMCPY_DEVICE_TO_HOST ? host_buffer : device_buffer;
            const void* src = kind == ACL_MEMCPY_DEVICE_TO_HOST ? device_buffer : host_buffer;
            const aclError status = aclrtMemcpy(dst, opt.pcieSizeBytes, src, opt.pcieSizeBytes, kind);
            stage.last_end = std::chrono::steady_clock::now();
            if (status != ACL_SUCCESS) return status;
            ++stage.successes;
            stage.copy_seconds += std::chrono::duration<double>(stage.last_end - begin).count();
            continue;
        }
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds(1), [&]() { return stop.load(); });
    }
    return ACL_SUCCESS;
}

static bool ParseRanges(const std::string& str, std::vector<std::pair<uint32_t, uint32_t>>* ranges)
{
    ranges->clear();
    if (str.empty()) return true;
    size_t start = 0;
    while (start < str.size()) {
        size_t comma = str.find(',', start);
        std::string token = str.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        size_t dash = token.find('-');
        if (dash == std::string::npos || dash == 0 || dash == token.size() - 1) {
            return false;
        }
        char* endp1 = nullptr;
        char* endp2 = nullptr;
        unsigned long s = std::strtoul(token.c_str(), &endp1, 10);
        unsigned long e = std::strtoul(token.c_str() + dash + 1, &endp2, 10);
        if (*endp1 != '-' || *endp2 != '\0' || s >= e) {
            return false;
        }
        ranges->push_back({static_cast<uint32_t>(s), static_cast<uint32_t>(e)});
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return !ranges->empty();
}

static int EnvInt(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::atoi(value) : fallback;
}

static int GetLocalRankFallback(int worldRank)
{
    int localRank = EnvInt("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (localRank >= 0) {
        return localRank;
    }
    localRank = EnvInt("MV2_COMM_WORLD_LOCAL_RANK", -1);
    if (localRank >= 0) {
        return localRank;
    }
    localRank = EnvInt("SLURM_LOCALID", -1);
    if (localRank >= 0) {
        return localRank;
    }
    localRank = EnvInt("PMI_LOCAL_RANK", -1);
    if (localRank >= 0) {
        return localRank;
    }
    localRank = EnvInt("MPI_LOCALRANKID", -1);
    if (localRank >= 0) {
        return localRank;
    }
    return worldRank;
}

static const char* OpName(Op op)
{
    switch (op) {
        case Op::ALLGATHER:
            return "ag";
        case Op::REDUCESCATTER:
            return "rs";
        case Op::ALLREDUCE:
            return "ar";
        case Op::ALLTOALL:
            return "a2a";
        default:
            return "unknown";
    }
}

static bool ParseOp(const char* value, Op* op, std::string* opName)
{
    if (std::strcmp(value, "ag") == 0 || std::strcmp(value, "allgather") == 0) {
        *op = Op::ALLGATHER;
    } else if (std::strcmp(value, "rs") == 0 || std::strcmp(value, "reducescatter") == 0) {
        *op = Op::REDUCESCATTER;
    } else if (std::strcmp(value, "ar") == 0 || std::strcmp(value, "allreduce") == 0) {
        *op = Op::ALLREDUCE;
    } else if (std::strcmp(value, "a2a") == 0 || std::strcmp(value, "alltoall") == 0) {
        *op = Op::ALLTOALL;
    } else {
        return false;
    }
    *opName = OpName(*op);
    return true;
}

static void PrintUsage(const char* prog)
{
    std::printf("Usage: %s --op=<ag|rs|ar|a2a> --size=N [--iters=N] [--warmup=N] [--device=N]\n", prog);
    std::printf("       [--pcie-ranges=start-end[,start-end...]] [--pcie-size=BYTES]\n");
    std::printf("       [--reuse-input] prepare input once; verify only the final output\n");
    std::printf("       [--pcie-direction=d2h|h2d|both] both uses independent workers and buffers\n");
    std::printf("   or: %s <op> <size> <iters>\n", prog);
    std::printf("  ag:  size is send_count per rank; recv_count = nranks * size\n");
    std::printf("  rs:  size is recv_count per rank; send_count = nranks * size\n");
    std::printf("  ar:  size is send_count/recv_count per rank\n");
    std::printf("  a2a: size is peer_count sent to each rank; send/recv_count = nranks * size\n");
    std::printf("  pcie ranges are left-closed/right-open, e.g. 10-20 means iter 10..19\n");
    std::printf("  pcie interference uses continuous background copies while a range is active\n");
}

static bool ParseArgs(int argc, char** argv, Options* opt)
{
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--op=", 5) == 0) {
            if (!ParseOp(argv[i] + 5, &opt->op, &opt->opName)) {
                std::fprintf(stderr, "Unsupported op: %s\n", argv[i] + 5);
                return false;
            }
        } else if (std::strncmp(argv[i], "--size=", 7) == 0) {
            opt->size = static_cast<uint64_t>(std::strtoull(argv[i] + 7, nullptr, 10));
        } else if (std::strncmp(argv[i], "--count=", 8) == 0) {
            opt->size = static_cast<uint64_t>(std::strtoull(argv[i] + 8, nullptr, 10));
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            opt->iters = static_cast<uint32_t>(std::strtoul(argv[i] + 8, nullptr, 10));
        } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
            opt->warmup = static_cast<uint32_t>(std::strtoul(argv[i] + 9, nullptr, 10));
        } else if (std::strncmp(argv[i], "--device=", 9) == 0) {
            opt->device = std::atoi(argv[i] + 9);
        } else if (std::strncmp(argv[i], "--pcie-ranges=", 14) == 0) {
            opt->pcieRangesStr = argv[i] + 14;
            if (!ParseRanges(opt->pcieRangesStr, &opt->pcieRanges)) {
                std::fprintf(stderr, "Invalid --pcie-ranges: %s\n", opt->pcieRangesStr.c_str());
                return false;
            }
        } else if (std::strncmp(argv[i], "--pcie-size=", 12) == 0) {
            opt->pcieSizeBytes = static_cast<size_t>(std::strtoull(argv[i] + 12, nullptr, 10));
        } else if (std::strncmp(argv[i], "--pcie-direction=", 17) == 0) {
            opt->pcieDirection = argv[i] + 17;
            if (opt->pcieDirection != "d2h" && opt->pcieDirection != "h2d" && opt->pcieDirection != "both") {
                std::fprintf(stderr, "Invalid --pcie-direction: %s\n", opt->pcieDirection.c_str());
                return false;
            }
        } else if (std::strcmp(argv[i], "--reuse-input") == 0) {
            opt->reuseInput = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            opt->help = true;
            return true;
        } else if (argv[i][0] != '-') {
            if (positional == 0) {
                if (!ParseOp(argv[i], &opt->op, &opt->opName)) {
                    std::fprintf(stderr, "Unsupported op: %s\n", argv[i]);
                    return false;
                }
            } else if (positional == 1) {
                opt->size = static_cast<uint64_t>(std::strtoull(argv[i], nullptr, 10));
            } else if (positional == 2) {
                opt->iters = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10));
            } else {
                std::fprintf(stderr, "Unexpected positional argument: %s\n", argv[i]);
                return false;
            }
            ++positional;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    if (opt->help) {
        return true;
    }
    if (opt->size == 0 || opt->iters == 0) {
        std::fprintf(stderr, "Invalid size/iters: size=%lu iters=%u\n",
                     static_cast<unsigned long>(opt->size), opt->iters);
        return false;
    }
    if (!opt->pcieRanges.empty() && opt->pcieSizeBytes == 0) {
        std::fprintf(stderr, "--pcie-size must be positive when --pcie-ranges is set\n");
        return false;
    }
    if (opt->pcieSizeBytes > 0 && opt->pcieRanges.empty()) {
        std::fprintf(stderr, "--pcie-ranges is required when --pcie-size is set\n");
        return false;
    }
    return true;
}

static float BaseValue(uint32_t rank, uint32_t iter, uint64_t i, uint32_t nranks)
{
    return static_cast<float>(iter * nranks + rank + (i % 10000));
}

static float ReducedValue(uint32_t nranks, uint32_t iter, uint64_t globalIndex)
{
    double sum = 0.0;
    for (uint32_t rank = 0; rank < nranks; ++rank) {
        sum += static_cast<double>(BaseValue(rank, iter, globalIndex, nranks));
    }
    return static_cast<float>(sum);
}

static bool NearlyEqual(float actual, float expected)
{
    return std::fabs(actual - expected) <= 1e-2f;
}

static void BufferLayout(const Options& opt, uint32_t nranks, uint64_t* sendElems,
                         uint64_t* recvElems, uint64_t* bandwidthBytes)
{
    const uint64_t size = opt.size;
    if (opt.op == Op::ALLGATHER) {
        *sendElems = size;
        *recvElems = size * nranks;
        *bandwidthBytes = size * sizeof(float);
    } else if (opt.op == Op::REDUCESCATTER) {
        *sendElems = size * nranks;
        *recvElems = size;
        *bandwidthBytes = *sendElems * sizeof(float);
    } else if (opt.op == Op::ALLREDUCE) {
        *sendElems = size;
        *recvElems = size;
        *bandwidthBytes = size * sizeof(float);
    } else {
        *sendElems = size * nranks;
        *recvElems = size * nranks;
        *bandwidthBytes = *sendElems * sizeof(float);
    }
}

static void FillInput(const Options& opt, uint32_t rank, uint32_t nranks,
                      uint32_t iter, float* hostSend)
{
    const uint64_t size = opt.size;
    if (opt.op == Op::ALLTOALL) {
        for (uint32_t dst = 0; dst < nranks; ++dst) {
            for (uint64_t i = 0; i < size; ++i) {
                hostSend[static_cast<uint64_t>(dst) * size + i] = BaseValue(rank, iter, i, nranks);
            }
        }
        return;
    }

    uint64_t sendElems = size;
    if (opt.op == Op::REDUCESCATTER) {
        sendElems = size * nranks;
    }
    for (uint64_t i = 0; i < sendElems; ++i) {
        hostSend[i] = BaseValue(rank, iter, i, nranks);
    }
}

static int VerifyOutput(const Options& opt, uint32_t rank, uint32_t nranks,
                        uint32_t iter, const float* hostRecv)
{
    const uint64_t size = opt.size;
    int failed = 0;

    if (opt.op == Op::ALLGATHER) {
        for (uint32_t src = 0; src < nranks; ++src) {
            for (uint64_t i = 0; i < size; ++i) {
                const float expected = BaseValue(src, iter, i, nranks);
                const float actual = hostRecv[static_cast<uint64_t>(src) * size + i];
                if (!NearlyEqual(actual, expected)) {
                    if (failed < 10) {
                        std::printf("rank %u iter %u: ag mismatch block %u idx %lu expected %.2f actual %.2f\n",
                                    rank, iter, src, static_cast<unsigned long>(i), expected, actual);
                    }
                    ++failed;
                }
            }
        }
    } else if (opt.op == Op::REDUCESCATTER) {
        const uint64_t base = static_cast<uint64_t>(rank) * size;
        for (uint64_t i = 0; i < size; ++i) {
            const uint64_t globalIndex = base + i;
            const float expected = ReducedValue(nranks, iter, globalIndex);
            const float actual = hostRecv[i];
            if (!NearlyEqual(actual, expected)) {
                if (failed < 10) {
                    std::printf("rank %u iter %u: rs mismatch local %lu global %lu expected %.2f actual %.2f\n",
                                rank, iter, static_cast<unsigned long>(i),
                                static_cast<unsigned long>(globalIndex), expected, actual);
                }
                ++failed;
            }
        }
    } else if (opt.op == Op::ALLREDUCE) {
        for (uint64_t i = 0; i < size; ++i) {
            const float expected = ReducedValue(nranks, iter, i);
            const float actual = hostRecv[i];
            if (!NearlyEqual(actual, expected)) {
                if (failed < 10) {
                    std::printf("rank %u iter %u: ar mismatch idx %lu expected %.2f actual %.2f\n",
                                rank, iter, static_cast<unsigned long>(i), expected, actual);
                }
                ++failed;
            }
        }
    } else {
        for (uint32_t src = 0; src < nranks; ++src) {
            for (uint64_t i = 0; i < size; ++i) {
                const float expected = BaseValue(src, iter, i, nranks);
                const float actual = hostRecv[static_cast<uint64_t>(src) * size + i];
                if (!NearlyEqual(actual, expected)) {
                    if (failed < 10) {
                        std::printf("rank %u iter %u: a2a mismatch src %u idx %lu expected %.2f actual %.2f\n",
                                    rank, iter, src, static_cast<unsigned long>(i), expected, actual);
                    }
                    ++failed;
                }
            }
        }
    }
    return failed;
}

static int RunCollective(const Options& opt, HcclComm comm, aclrtStream stream,
                         void* sendBuf, void* recvBuf)
{
    const uint64_t size = opt.size;
    if (opt.op == Op::ALLGATHER) {
        HCCLCHECK(HcclAllGather(sendBuf, recvBuf, size, HCCL_DATA_TYPE_FP32, comm, stream));
    } else if (opt.op == Op::REDUCESCATTER) {
        HCCLCHECK(HcclReduceScatter(sendBuf, recvBuf, size, HCCL_DATA_TYPE_FP32,
                                    HCCL_REDUCE_SUM, comm, stream));
    } else if (opt.op == Op::ALLREDUCE) {
        HCCLCHECK(HcclAllReduce(sendBuf, recvBuf, size, HCCL_DATA_TYPE_FP32,
                                HCCL_REDUCE_SUM, comm, stream));
    } else {
        HCCLCHECK(HcclAlltoAll(sendBuf, size, HCCL_DATA_TYPE_FP32,
                               recvBuf, size, HCCL_DATA_TYPE_FP32, comm, stream));
    }
    return 0;
}

static int RunOneIteration(const Options& opt, uint32_t rank, uint32_t nranks,
                           HcclComm comm, aclrtStream stream,
                           aclrtEvent startEvent, aclrtEvent endEvent,
                           void* sendBuf, void* recvBuf,
                           float* hostSend, float* hostRecv,
                           uint32_t iter, bool verify, double* elapsedMs)
{
    uint64_t sendElems = 0;
    uint64_t recvElems = 0;
    uint64_t bandwidthBytes = 0;
    BufferLayout(opt, nranks, &sendElems, &recvElems, &bandwidthBytes);
    const size_t sendBytes = sendElems * sizeof(float);
    const size_t recvBytes = recvElems * sizeof(float);

    auto t0 = std::chrono::steady_clock::now();
    if (!opt.reuseInput) FillInput(opt, rank, nranks, iter, hostSend);
    auto t1 = std::chrono::steady_clock::now();
    if (!opt.reuseInput) {
        ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend, sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto t2 = std::chrono::steady_clock::now();
    ACLCHECK(aclrtSynchronizeStream(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    if (elapsedMs != nullptr) {
        ACLCHECK(aclrtRecordEvent(startEvent, stream));
    }
    int ret = RunCollective(opt, comm, stream, sendBuf, recvBuf);
    auto t3 = std::chrono::steady_clock::now();
    if (ret != 0) {
        return ret;
    }
    if (elapsedMs != nullptr) {
        ACLCHECK(aclrtRecordEvent(endEvent, stream));
    }
    ACLCHECK(aclrtSynchronizeStream(stream));
    auto t4 = std::chrono::steady_clock::now();
    if (elapsedMs != nullptr) {
        ACLCHECK(aclrtSynchronizeEvent(endEvent));
        float ms = 0.0f;
        ACLCHECK(aclrtEventElapsedTime(&ms, startEvent, endEvent));
        *elapsedMs = static_cast<double>(ms);
    }

    if (!verify) {
        return 0;
    }
    ACLCHECK(aclrtMemcpy(hostRecv, recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    auto t5 = std::chrono::steady_clock::now();
    int errs = VerifyOutput(opt, rank, nranks, iter, hostRecv);
    auto t6 = std::chrono::steady_clock::now();

    double ms_fill = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_h2d = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double ms_coll = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double ms_sync = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double ms_d2h = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double ms_ver = std::chrono::duration<double, std::milli>(t6 - t5).count();
    if (std::getenv("BENCHMARK_DEBUG_TIMING")) {
        std::printf("[RANK%u-MAIN iter=%u] fill=%.2f h2d=%.2f submit=%.2f sync=%.2f d2h=%.2f verify=%.2f\n",
                    rank, iter, ms_fill, ms_h2d, ms_coll, ms_sync, ms_d2h, ms_ver);
        std::fflush(stdout);
    }
    return errs;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    Options opt;
    bool parseOk = ParseArgs(argc, argv, &opt);
    int parseOkInt = parseOk ? 1 : 0;
    MPI_Bcast(&parseOkInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOkInt) {
        if (worldRank == 0) {
            PrintUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if (opt.help) {
        if (worldRank == 0) {
            PrintUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    ACLCHECK(aclInit(nullptr));
    uint32_t availableDevs = 0;
    ACLCHECK(aclrtGetDeviceCount(&availableDevs));
    if (availableDevs == 0) {
        std::fprintf(stderr, "rank %d: no Ascend device available\n", worldRank);
        aclFinalize();
        MPI_Finalize();
        return 1;
    }

    const int localRank = GetLocalRankFallback(worldRank);
    const int device = opt.device >= 0 ? opt.device : (localRank % static_cast<int>(availableDevs));
    ACLCHECK(aclrtSetDevice(device));

    HcclRootInfo rootInfo;
    std::memset(&rootInfo, 0, sizeof(rootInfo));
    if (worldRank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
    }
    MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

    HcclComm comm = nullptr;
    std::printf("[rank %d] HcclCommInitRootInfo begin\n", worldRank);
    std::fflush(stdout);
    HCCLCHECK(HcclCommInitRootInfo(static_cast<uint32_t>(worldSize), &rootInfo,
                                   static_cast<uint32_t>(worldRank), &comm));
    std::printf("[rank %d] HcclCommInitRootInfo end\n", worldRank);
    std::fflush(stdout);

    aclrtStream stream = nullptr;
    ACLCHECK(aclrtCreateStream(&stream));
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    ACLCHECK(aclrtCreateEventWithFlag(&startEvent, ACL_EVENT_TIME_LINE));
    ACLCHECK(aclrtCreateEventWithFlag(&endEvent, ACL_EVENT_TIME_LINE));

    uint64_t sendElems = 0;
    uint64_t recvElems = 0;
    uint64_t bandwidthBytes = 0;
    BufferLayout(opt, static_cast<uint32_t>(worldSize), &sendElems, &recvElems, &bandwidthBytes);
    const uint64_t maxElems = sendElems > recvElems ? sendElems : recvElems;
    const size_t sendBytes = sendElems * sizeof(float);
    const size_t recvBytes = recvElems * sizeof(float);
    const size_t maxBytes = maxElems * sizeof(float);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    void* hostSend = nullptr;
    void* hostRecv = nullptr;
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMallocHost(&hostSend, maxBytes));
    ACLCHECK(aclrtMallocHost(&hostRecv, maxBytes));

    if (worldRank == 0) {
        std::printf("HCCL MPI primitive test: op=%s nranks=%d size=%lu send_elems=%lu recv_elems=%lu iters=%u warmup=%u\n",
                    opt.opName.c_str(), worldSize, static_cast<unsigned long>(opt.size),
                    static_cast<unsigned long>(sendElems), static_cast<unsigned long>(recvElems),
                    opt.iters, opt.warmup);
        std::fflush(stdout);
    }
    std::printf("rank %d/%d using device %d\n", worldRank, worldSize, device);
    std::fflush(stdout);

    if (opt.reuseInput) {
        // Out-of-place collectives never overwrite this fixed input. Use
        // generation zero for both its preparation and the final validation.
        FillInput(opt, static_cast<uint32_t>(worldRank), static_cast<uint32_t>(worldSize),
                  0, static_cast<float*>(hostSend));
        ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend, sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        if (worldRank == 0) {
            std::printf("input_mode=reuse_once validation=final_only input_generation=0\n");
            std::fflush(stdout);
        }
    }

    std::array<PcieWorkerState, 2> pcieWorkers;
    const size_t workerCount = opt.pcieSizeBytes > 0 && !opt.pcieRanges.empty()
        ? (opt.pcieDirection == "both" ? 2 : 1) : 0;
    std::atomic<bool> workerStop{false};
    std::atomic<uint32_t> currentIter{0};
    std::atomic<bool> inFormal{false};
    std::mutex workerSleepMtx;
    std::condition_variable workerSleepCv;
    std::vector<std::chrono::steady_clock::time_point> stageBegin(opt.pcieRanges.size()),
                                                     stageEnd(opt.pcieRanges.size());

    for (size_t n = 0; n < workerCount; ++n) {
        auto& worker = pcieWorkers[n];
        worker.kind = opt.pcieDirection == "h2d" || n == 1
            ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
        worker.metrics.resize(opt.pcieRanges.size());
        ACLCHECK(aclrtMalloc(&worker.device, opt.pcieSizeBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMallocHost(&worker.host, opt.pcieSizeBytes));
        std::memset(worker.host, 0x5a, opt.pcieSizeBytes);
        ACLCHECK(aclrtMemset(worker.device, opt.pcieSizeBytes, 0xa5, opt.pcieSizeBytes));
        worker.thread = std::thread([&, n, device]() {
            auto& w = pcieWorkers[n];
            w.status.store(RunPcieInterference(device, w.device, w.host, opt,
                           workerStop, currentIter, inFormal, workerSleepMtx,
                           workerSleepCv, w.metrics, w.kind), std::memory_order_release);
        });
    }
    if (workerCount > 0) {
        if (worldRank == 0) {
            std::printf("PCIe interference: mode=continuous direction=%s workers=%zu size_bytes=%zu ranges=",
                        opt.pcieDirection.c_str(), workerCount, opt.pcieSizeBytes);
            for (const auto& r : opt.pcieRanges) {
                std::printf("[%u,%u) ", r.first, r.second);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (uint32_t w = 0; w < opt.warmup; ++w) {
        MPI_Barrier(MPI_COMM_WORLD);
        const int failed = RunOneIteration(opt, static_cast<uint32_t>(worldRank),
                                           static_cast<uint32_t>(worldSize), comm, stream,
                                           startEvent, endEvent, sendBuf, recvBuf,
                                           static_cast<float*>(hostSend), static_cast<float*>(hostRecv),
                                           w, false, nullptr);
        if (failed != 0) {
            std::fprintf(stderr, "rank %d: warmup failed\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (worldRank == 0) {
        std::printf("Warmup done (%u iters)\n", opt.warmup);
        std::fflush(stdout);
    }

    int localFailed = 0;
    double sumMaxEventMs = 0.0;
    for (uint32_t iter = 0; iter < opt.iters; ++iter) {
        for (size_t n = 0; n < workerCount; ++n) {
            if (pcieWorkers[n].status.load(std::memory_order_acquire) != ACL_SUCCESS) {
                std::fprintf(stderr, "rank %d: PCIe interference worker=%zu failed status=%d\n",
                             worldRank, n, pcieWorkers[n].status.load());
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        for (size_t k = 0; k < opt.pcieRanges.size(); ++k) {
            if (iter == opt.pcieRanges[k].first) stageBegin[k] = std::chrono::steady_clock::now();
            if (iter == opt.pcieRanges[k].second) stageEnd[k] = std::chrono::steady_clock::now();
        }
        if (worldRank == 0) {
            std::printf("[rank0] begin formal %u iters\n", iter);
            std::fflush(stdout);
        }
        currentIter.store(iter, std::memory_order_release);
        inFormal.store(true, std::memory_order_release);
        MPI_Barrier(MPI_COMM_WORLD);
        double localEventMs = 0.0;
        const int failed = RunOneIteration(opt, static_cast<uint32_t>(worldRank),
                                           static_cast<uint32_t>(worldSize), comm, stream,
                                           startEvent, endEvent, sendBuf, recvBuf,
                                           static_cast<float*>(hostSend), static_cast<float*>(hostRecv),
                                           iter, !opt.reuseInput, &localEventMs);
        MPI_Barrier(MPI_COMM_WORLD);
        double maxEventMs = 0.0;
        MPI_Allreduce(&localEventMs, &maxEventMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        sumMaxEventMs += maxEventMs;
        localFailed += failed;
        if (worldRank == 0) {
            const double iterSec = maxEventMs / 1000.0;
            const double iterBw = (iterSec > 1e-9)
                ? (static_cast<double>(bandwidthBytes) / (1024.0 * 1024.0 * 1024.0) / iterSec)
                : 0.0;
            std::printf("iter=%u event_ms=%.3f bw_GBps=%.3f\n", iter, maxEventMs, iterBw);
            std::fflush(stdout);
        }
    }
    inFormal.store(false, std::memory_order_release);
    for (size_t k = 0; k < opt.pcieRanges.size(); ++k) {
        if (opt.pcieRanges[k].first < opt.iters && opt.pcieRanges[k].second >= opt.iters)
            stageEnd[k] = std::chrono::steady_clock::now();
    }

    // Retire any outstanding background copy before final validation, so the
    // large D2H/CPU check occurs only once and outside all measured iterations.
    if (workerCount > 0) {
        {
            std::lock_guard<std::mutex> lock(workerSleepMtx);
            workerStop.store(true, std::memory_order_release);
        }
        workerSleepCv.notify_all();
        for (size_t n = 0; n < workerCount; ++n) pcieWorkers[n].thread.join();
    }
    for (int reportingRank = 0; reportingRank < worldSize; ++reportingRank) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (worldRank != reportingRank) continue;
    for (size_t n = 0; n < workerCount; ++n) {
      const auto& worker = pcieWorkers[n];
      if (worker.status.load() != ACL_SUCCESS) {
        std::fprintf(stderr, "rank %d: PCIe interference worker=%zu failed status=%d\n",
                     worldRank, n, worker.status.load());
        ++localFailed;
      }
      for (size_t k = 0; k < worker.metrics.size(); ++k) {
        const auto& metrics = worker.metrics[k];
        const double span = metrics.attempts
            ? std::chrono::duration<double>(metrics.last_end - metrics.first_start).count() : 0.0;
        const double active = opt.pcieRanges[k].first < opt.iters
            ? std::chrono::duration<double>(stageEnd[k] - stageBegin[k]).count() : 0.0;
        const double bytes = static_cast<double>(metrics.successes) * opt.pcieSizeBytes;
        std::printf("pcie_background rank=%d direction=%s stage=%zu range=%u-%u attempts=%llu successes=%llu status=%d "
                    "bytes=%.0f copy_seconds=%.6f span_seconds=%.6f active_seconds=%.6f "
                    "copy_GBps=%.6f active_GBps=%.6f\n", worldRank,
                    worker.kind == ACL_MEMCPY_DEVICE_TO_HOST ? "d2h" : "h2d", k,
                    opt.pcieRanges[k].first, opt.pcieRanges[k].second,
                    static_cast<unsigned long long>(metrics.attempts),
                    static_cast<unsigned long long>(metrics.successes), worker.status.load(), bytes,
                    metrics.copy_seconds, span, active,
                    metrics.copy_seconds > 0 ? bytes / metrics.copy_seconds / 1e9 : 0.0,
                    active > 0 ? bytes / active / 1e9 : 0.0);
        if (opt.pcieRanges[k].first < opt.iters && metrics.successes == 0) ++localFailed;
      }
    }
      std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (opt.reuseInput) {
        ACLCHECK(aclrtMemcpy(hostRecv, recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        localFailed += VerifyOutput(opt, static_cast<uint32_t>(worldRank),
                                    static_cast<uint32_t>(worldSize), 0,
                                    static_cast<float*>(hostRecv));
    }

    int globalFailed = 0;
    MPI_Allreduce(&localFailed, &globalFailed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (localFailed == 0) {
        std::printf("rank %d: PASSED (op=%s size=%lu iters=%u)\n",
                    worldRank, opt.opName.c_str(), static_cast<unsigned long>(opt.size), opt.iters);
    } else {
        std::printf("rank %d: FAILED %d element(s) mismatch\n", worldRank, localFailed);
    }
    std::fflush(stdout);

    if (worldRank == 0 && opt.iters > 0) {
        const double avgMs = sumMaxEventMs / static_cast<double>(opt.iters);
        const double elapsedSec = avgMs / 1000.0;
        const double bandwidthGBps = (elapsedSec > 1e-9)
            ? (static_cast<double>(bandwidthBytes) / (1024.0 * 1024.0 * 1024.0) / elapsedSec)
            : 0.0;
        std::printf("Average bandwidth over %u iters (event): %.3f GB/s\n",
                    opt.iters, bandwidthGBps);
        std::printf("result: %s op=%s nranks=%d size=%lu avg_time=%.3f ms bandwidth=%.3f GB/s global_mismatches=%d\n",
                    globalFailed == 0 ? "PASSED" : "FAILED", opt.opName.c_str(), worldSize,
                    static_cast<unsigned long>(opt.size), avgMs, bandwidthGBps, globalFailed);
    }

    for (size_t n = 0; n < workerCount; ++n) {
        ACLCHECK(aclrtFree(pcieWorkers[n].device));
        ACLCHECK(aclrtFreeHost(pcieWorkers[n].host));
    }

    HCCLCHECK(HcclCommDestroy(comm));
    ACLCHECK(aclrtDestroyEvent(startEvent));
    ACLCHECK(aclrtDestroyEvent(endEvent));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtFreeHost(hostSend));
    ACLCHECK(aclrtFreeHost(hostRecv));
    ACLCHECK(aclFinalize());
    MPI_Finalize();
    return globalFailed == 0 ? 0 : 1;
}
