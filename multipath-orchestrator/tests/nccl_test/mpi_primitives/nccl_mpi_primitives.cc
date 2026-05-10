#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>

#define CUDACHECK(cmd)                                                                 \
    do {                                                                               \
        cudaError_t e = (cmd);                                                         \
        if (e != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(e));                                       \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

#define NCCLCHECK(cmd)                                                                 \
    do {                                                                               \
        ncclResult_t r = (cmd);                                                        \
        if (r != ncclSuccess) {                                                        \
            std::fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__,       \
                         ncclGetErrorString(r));                                       \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

static int EnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::atoi(v) : fallback;
}

static int GetLocalRankFallback(int world_rank) {
    int v = EnvInt("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (v >= 0) return v;
    v = EnvInt("MV2_COMM_WORLD_LOCAL_RANK", -1);
    if (v >= 0) return v;
    v = EnvInt("SLURM_LOCALID", -1);
    if (v >= 0) return v;
    v = EnvInt("PMI_LOCAL_RANK", -1);
    if (v >= 0) return v;
    v = EnvInt("MPI_LOCALRANKID", -1);
    if (v >= 0) return v;
    return world_rank;
}

static float BaseValue(int rank, uint32_t iter, uint64_t i) {
    return 1000000.0f * static_cast<float>(rank) +
           1000.0f * static_cast<float>(iter) + static_cast<float>(i);
}

static float AllToAllValue(int src, int dst, uint32_t iter, uint64_t i) {
    return 1000000.0f * static_cast<float>(src) +
           10000.0f * static_cast<float>(dst) +
           1000.0f * static_cast<float>(iter) + static_cast<float>(i);
}

static float ReducedValue(int nranks, uint32_t iter, uint64_t global_i) {
    double sum = 0.0;
    for (int r = 0; r < nranks; ++r) {
        sum += static_cast<double>(BaseValue(r, iter, global_i));
    }
    return static_cast<float>(sum);
}

static bool NearlyEqual(float a, float b) {
    return std::fabs(a - b) <= 1e-2f;
}

static uint64_t PayloadBytes(const std::string& op, uint64_t count, int nranks) {
    const uint64_t elem_size = sizeof(float);
    if (op == "allgather" || op == "reducescatter" || op == "alltoall") {
        return count * static_cast<uint64_t>(nranks) * elem_size;
    }
    return count * elem_size;
}

static double BusBwFactor(const std::string& op, int nranks) {
    if (nranks <= 1) {
        return 1.0;
    }
    if (op == "allreduce") {
        return 2.0 * static_cast<double>(nranks - 1) / static_cast<double>(nranks);
    }
    return static_cast<double>(nranks - 1) / static_cast<double>(nranks);
}

typedef ncclResult_t (*NcclAllToAllFn)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);

static ncclResult_t CallNcclAllToAll(const void* sendbuf, void* recvbuf, size_t count,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    static NcclAllToAllFn fn = []() {
        NcclAllToAllFn f = reinterpret_cast<NcclAllToAllFn>(dlsym(RTLD_DEFAULT, "ncclAlltoAll"));
        if (!f) {
            f = reinterpret_cast<NcclAllToAllFn>(dlsym(RTLD_DEFAULT, "ncclAllToAll"));
        }
        return f;
    }();
    if (!fn) {
        std::fprintf(stderr, "ncclAlltoAll/ncclAllToAll symbol not found. Run with LD_PRELOAD=libampccl_nccl.so or a NCCL version exposing ncclAllToAll.\n");
        return ncclSystemError;
    }
    return fn(sendbuf, recvbuf, count, datatype, comm, stream);
}

struct Options {
    std::string op = "allreduce";
    uint64_t count = 1024;
    int iters = 5;
    int warmup = 3;
    int device = -1;
};

static void PrintUsage(const char* prog) {
    std::printf("Usage: %s --op=<allreduce|allgather|reducescatter|alltoall> [options]\n", prog);
    std::printf("Options:\n");
    std::printf("  --count=N       float elements per rank/peer (default 1024)\n");
    std::printf("  --iters=N       measured iterations (default 5)\n");
    std::printf("  --warmup=N      warmup iterations (default 3)\n");
    std::printf("  --device=N      override CUDA device for every process\n");
}

static bool ParseArgs(int argc, char** argv, Options* opt) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--op=", 5) == 0) {
            opt->op = argv[i] + 5;
        } else if (std::strncmp(argv[i], "--count=", 8) == 0) {
            opt->count = static_cast<uint64_t>(std::strtoull(argv[i] + 8, nullptr, 10));
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            opt->iters = std::atoi(argv[i] + 8);
        } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
            opt->warmup = std::atoi(argv[i] + 9);
        } else if (std::strncmp(argv[i], "--device=", 9) == 0) {
            opt->device = std::atoi(argv[i] + 9);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            PrintUsage(argv[0]);
            return false;
        }
    }
    if (opt->count == 0 || opt->iters <= 0 || opt->warmup < 0) {
        std::fprintf(stderr, "Invalid count/iters/warmup\n");
        return false;
    }
    if (opt->op != "allreduce" && opt->op != "allgather" &&
        opt->op != "reducescatter" && opt->op != "alltoall") {
        std::fprintf(stderr, "Unsupported op: %s\n", opt->op.c_str());
        return false;
    }
    return true;
}

static int RunOneIteration(const Options& opt, int rank, int nranks, ncclComm_t comm,
                           cudaStream_t stream, cudaEvent_t start_event, cudaEvent_t end_event,
                           float* send_dev, float* recv_dev,
                           float* send_host, float* recv_host, uint32_t iter,
                           bool verify, double* elapsed_ms) {
    const uint64_t count = opt.count;
    uint64_t send_elems = count;
    uint64_t recv_elems = count;

    if (opt.op == "allgather") {
        recv_elems = count * static_cast<uint64_t>(nranks);
        for (uint64_t i = 0; i < count; ++i) {
            send_host[i] = BaseValue(rank, iter, i);
        }
        CUDACHECK(cudaMemcpy(send_dev, send_host, send_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(start_event, stream));
        NCCLCHECK(ncclAllGather(send_dev, recv_dev, count, ncclFloat32, comm, stream));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(end_event, stream));
    } else if (opt.op == "allreduce") {
        for (uint64_t i = 0; i < count; ++i) {
            send_host[i] = BaseValue(rank, iter, i);
        }
        CUDACHECK(cudaMemcpy(send_dev, send_host, send_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(start_event, stream));
        NCCLCHECK(ncclAllReduce(send_dev, recv_dev, count, ncclFloat32, ncclSum, comm, stream));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(end_event, stream));
    } else if (opt.op == "reducescatter") {
        send_elems = count * static_cast<uint64_t>(nranks);
        for (uint64_t i = 0; i < send_elems; ++i) {
            send_host[i] = BaseValue(rank, iter, i);
        }
        CUDACHECK(cudaMemcpy(send_dev, send_host, send_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(start_event, stream));
        NCCLCHECK(ncclReduceScatter(send_dev, recv_dev, count, ncclFloat32, ncclSum, comm, stream));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(end_event, stream));
    } else {
        send_elems = count * static_cast<uint64_t>(nranks);
        recv_elems = count * static_cast<uint64_t>(nranks);
        for (int dst = 0; dst < nranks; ++dst) {
            for (uint64_t i = 0; i < count; ++i) {
                send_host[static_cast<uint64_t>(dst) * count + i] = AllToAllValue(rank, dst, iter, i);
            }
        }
        CUDACHECK(cudaMemcpy(send_dev, send_host, send_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(start_event, stream));
        NCCLCHECK(CallNcclAllToAll(send_dev, recv_dev, count, ncclFloat32, comm, stream));
        if (elapsed_ms) CUDACHECK(cudaEventRecord(end_event, stream));
    }

    CUDACHECK(cudaStreamSynchronize(stream));
    if (elapsed_ms) {
        float ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&ms, start_event, end_event));
        *elapsed_ms = static_cast<double>(ms);
    }
    if (!verify) {
        return 0;
    }

    CUDACHECK(cudaMemcpy(recv_host, recv_dev, recv_elems * sizeof(float), cudaMemcpyDeviceToHost));
    int failed = 0;
    if (opt.op == "allgather") {
        for (int src = 0; src < nranks; ++src) {
            for (uint64_t i = 0; i < count; ++i) {
                float expected = BaseValue(src, iter, i);
                float actual = recv_host[static_cast<uint64_t>(src) * count + i];
                if (!NearlyEqual(actual, expected)) failed++;
            }
        }
    } else if (opt.op == "allreduce") {
        for (uint64_t i = 0; i < count; ++i) {
            if (!NearlyEqual(recv_host[i], ReducedValue(nranks, iter, i))) failed++;
        }
    } else if (opt.op == "reducescatter") {
        const uint64_t base = static_cast<uint64_t>(rank) * count;
        for (uint64_t i = 0; i < count; ++i) {
            if (!NearlyEqual(recv_host[i], ReducedValue(nranks, iter, base + i))) failed++;
        }
    } else {
        for (int src = 0; src < nranks; ++src) {
            for (uint64_t i = 0; i < count; ++i) {
                float expected = AllToAllValue(src, rank, iter, i);
                float actual = recv_host[static_cast<uint64_t>(src) * count + i];
                if (!NearlyEqual(actual, expected)) failed++;
            }
        }
    }
    return failed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Options opt;
    bool ok = ParseArgs(argc, argv, &opt);
    int parse_ok = ok ? 1 : 0;
    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok) {
        MPI_Finalize();
        return 1;
    }

    int dev_count = 0;
    CUDACHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        std::fprintf(stderr, "rank %d: no CUDA devices\n", rank);
        MPI_Finalize();
        return 1;
    }
    int device = opt.device >= 0 ? opt.device : (GetLocalRankFallback(rank) % dev_count);
    CUDACHECK(cudaSetDevice(device));

    ncclUniqueId id;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm = nullptr;
    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

    cudaStream_t stream = nullptr;
    CUDACHECK(cudaStreamCreate(&stream));
    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    CUDACHECK(cudaEventCreate(&start_event));
    CUDACHECK(cudaEventCreate(&end_event));

    uint64_t max_elems = opt.count;
    if (opt.op == "allgather" || opt.op == "reducescatter" || opt.op == "alltoall") {
        max_elems = opt.count * static_cast<uint64_t>(nranks);
    }
    const size_t max_bytes = max_elems * sizeof(float);

    float* send_dev = nullptr;
    float* recv_dev = nullptr;
    float* send_host = nullptr;
    float* recv_host = nullptr;
    CUDACHECK(cudaMalloc(&send_dev, max_bytes));
    CUDACHECK(cudaMalloc(&recv_dev, max_bytes));
    CUDACHECK(cudaMallocHost(&send_host, max_bytes));
    CUDACHECK(cudaMallocHost(&recv_host, max_bytes));

    const uint64_t op_bytes = opt.count * sizeof(float);
    const uint64_t payload_bytes = PayloadBytes(opt.op, opt.count, nranks);

    if (rank == 0) {
        std::printf("NCCL MPI primitive test: op=%s nranks=%d count=%lu op_bytes=%lu payload_bytes=%lu iters=%d warmup=%d\n",
                    opt.op.c_str(), nranks, static_cast<unsigned long>(opt.count),
                    static_cast<unsigned long>(op_bytes), static_cast<unsigned long>(payload_bytes),
                    opt.iters, opt.warmup);
        std::fflush(stdout);
    }
    std::printf("rank %d/%d using cuda device %d\n", rank, nranks, device);
    std::fflush(stdout);

    for (int w = 0; w < opt.warmup; ++w) {
        int failed = RunOneIteration(opt, rank, nranks, comm, stream, start_event, end_event,
                                     send_dev, recv_dev, send_host, recv_host,
                                     static_cast<uint32_t>(w), false, nullptr);
        if (failed != 0) {
            std::fprintf(stderr, "rank %d: warmup failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        // Same marker as the single-process tests; process_compare_logs.py uses it
        // to separate warmup samples from measured AMPCCL aggregate lines.
        std::printf("Warmup done (%d iters)\n", opt.warmup);
        std::fflush(stdout);
    }

    int local_failed = 0;
    double total_event_ms = 0.0;
    double total_wall_sec = 0.0;
    for (int iter = 0; iter < opt.iters; ++iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        double wall_t0 = MPI_Wtime();
        double local_event_ms = 0.0;
        int failed = RunOneIteration(opt, rank, nranks, comm, stream, start_event, end_event,
                                     send_dev, recv_dev, send_host, recv_host,
                                     static_cast<uint32_t>(iter), true, &local_event_ms);
        double wall_t1 = MPI_Wtime();
        double max_event_ms = 0.0;
        MPI_Allreduce(&local_event_ms, &max_event_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        total_event_ms += max_event_ms;
        total_wall_sec += wall_t1 - wall_t0;
        local_failed += failed;
    }

    int global_failed = 0;
    MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double max_total_wall_sec = 0.0;
    MPI_Reduce(&total_wall_sec, &max_total_wall_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_ms = total_event_ms / static_cast<double>(opt.iters);
        double avg_sec = avg_ms / 1000.0;
        double avg_wall_ms = (max_total_wall_sec / static_cast<double>(opt.iters)) * 1000.0;
        double algbw = avg_sec > 0.0 ? static_cast<double>(payload_bytes) / 1.0e9 / avg_sec : 0.0;
        double bus_factor = BusBwFactor(opt.op, nranks);
        double busbw = algbw * bus_factor;
        std::printf("result: %s timer=cuda_event_collective op=%s nranks=%d count=%lu op_bytes=%lu payload_bytes=%lu avg_time=%.6f s avg_time=%.3f ms algbw=%.6f GB/s busbw=%.6f GB/s bus_factor=%.6f wall_avg_time=%.3f ms global_mismatches=%d\n",
                    global_failed == 0 ? "PASSED" : "FAILED", opt.op.c_str(), nranks,
                    static_cast<unsigned long>(opt.count), static_cast<unsigned long>(op_bytes),
                    static_cast<unsigned long>(payload_bytes), avg_sec, avg_ms, algbw, busbw,
                    bus_factor, avg_wall_ms, global_failed);
    }

    ncclCommDestroy(comm);
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    cudaStreamDestroy(stream);
    cudaFree(send_dev);
    cudaFree(recv_dev);
    cudaFreeHost(send_host);
    cudaFreeHost(recv_host);

    MPI_Finalize();
    return global_failed == 0 ? 0 : 1;
}
