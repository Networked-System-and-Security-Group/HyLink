// Minimal MPI multi-process NCCL ncclCommSplit + AllGather test.
//
// Default run shape:
//   mpirun -np 4 ./nccl_commsplit_allgather_mpi 4 1024 1 4
//
// Each MPI rank owns one CUDA GPU. The test first creates a 4-rank NCCL
// communicator and validates an AllGather on it. It then calls ncclCommSplit
// with color = world_rank / child_size. The default child_size is 4, so all
// ranks enter one 4-rank child communicator, which then runs and validates a
// second AllGather.

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd)                                                            \
    do {                                                                          \
        cudaError_t e = (cmd);                                                    \
        if (e != cudaSuccess) {                                                   \
            std::printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__,            \
                        cudaGetErrorString(e));                                   \
            return -1;                                                            \
        }                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                            \
    do {                                                                          \
        ncclResult_t r = (cmd);                                                   \
        if (r != ncclSuccess) {                                                   \
            std::printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
                        ncclGetErrorString(r));                                   \
            return -1;                                                            \
        }                                                                         \
    } while (0)

static int EnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::atoi(v) : fallback;
}

static int LocalRank(int world_rank) {
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

static float ExpectedValue(int global_rank, uint32_t iter, uint64_t i) {
    return 1000000.0f * static_cast<float>(global_rank) +
           1000.0f * static_cast<float>(iter) + static_cast<float>(i);
}

static void PrintUsage(const char* prog) {
    std::printf("Usage: %s [world_size] [send_count] [iters] [child_size]\n", prog);
    std::printf("  world_size  MPI ranks / GPUs, default 4\n");
    std::printf("  send_count  float elements per rank, default 1024\n");
    std::printf("  iters       measured iterations, default 1\n");
    std::printf("  child_size  ranks per ncclCommSplit child group, default 4\n");
}

static int FillSend(int global_rank, uint32_t iter, uint64_t send_count, void* host_send) {
    float* p = static_cast<float*>(host_send);
    for (uint64_t i = 0; i < send_count; ++i) {
        p[i] = ExpectedValue(global_rank, iter, i);
    }
    return 0;
}

static int CheckGather(const char* label, int world_rank, uint32_t iter, uint64_t send_count,
                       const std::vector<int>& expected_global_ranks, void* host_recv) {
    const float* p = static_cast<const float*>(host_recv);
    int failed = 0;
    for (size_t slot = 0; slot < expected_global_ranks.size(); ++slot) {
        int src_global_rank = expected_global_ranks[slot];
        for (uint64_t i = 0; i < send_count; ++i) {
            float expected = ExpectedValue(src_global_rank, iter, i);
            float actual = p[slot * send_count + i];
            if (std::fabs(actual - expected) > 1e-3f) {
                if (failed < 8) {
                    std::printf("rank %d: %s mismatch iter=%u slot=%zu src_global=%d i=%lu actual=%f expected=%f\n",
                                world_rank, label, iter, slot, src_global_rank,
                                static_cast<unsigned long>(i), actual, expected);
                }
                ++failed;
            }
        }
    }
    return failed;
}

static int RunAllGatherCase(const char* label, MPI_Comm mpi_comm, int world_rank, int global_group_start,
                            int group_size, int local_rank_in_group, uint64_t send_count,
                            uint32_t iters, ncclComm_t nccl_comm, cudaStream_t stream,
                            void* send_buf, void* recv_buf, void* host_send, void* host_recv) {
    const size_t send_size = send_count * sizeof(float);
    const size_t recv_size = static_cast<size_t>(group_size) * send_size;
    int local_failed = 0;

    std::vector<int> expected_global_ranks;
    expected_global_ranks.reserve(static_cast<size_t>(group_size));
    for (int r = 0; r < group_size; ++r) {
        expected_global_ranks.push_back(global_group_start + r);
    }

    for (uint32_t iter = 0; iter < iters; ++iter) {
        FillSend(world_rank, iter, send_count, host_send);
        CUDACHECK(cudaMemcpy(send_buf, host_send, send_size, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemsetAsync(recv_buf, 0, recv_size, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        MPI_Barrier(mpi_comm);
        if (local_rank_in_group == 0) {
            std::printf("%s group_start=%d size=%d iter=%u begin\n",
                        label, global_group_start, group_size, iter);
        }

        NCCLCHECK(ncclAllGather(send_buf, recv_buf, send_count, ncclFloat32, nccl_comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaMemcpy(host_recv, recv_buf, recv_size, cudaMemcpyDeviceToHost));

        int failed = CheckGather(label, world_rank, iter, send_count, expected_global_ranks, host_recv);
        if (failed) {
            std::printf("rank %d: %s iter=%u FAILED mismatches=%d\n", world_rank, label, iter, failed);
            local_failed += failed;
        } else {
            std::printf("rank %d: %s iter=%u PASSED\n", world_rank, label, iter);
        }
    }
    return local_failed ? -1 : 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int expected_world_size = 4;
    uint64_t send_count = 1024;
    uint32_t iters = 1;
    int child_size = 4;

    if (argc > 1 && std::strcmp(argv[1], "--help") == 0) {
        if (world_rank == 0) PrintUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argc >= 2) expected_world_size = std::atoi(argv[1]);
    if (argc >= 3) send_count = static_cast<uint64_t>(std::atoll(argv[2]));
    if (argc >= 4) iters = static_cast<uint32_t>(std::atoi(argv[3]));
    if (argc >= 5) child_size = std::atoi(argv[4]);

    int invalid = 0;
    if (expected_world_size <= 0 || expected_world_size != world_size || send_count == 0 ||
        iters == 0 || child_size <= 0 || (world_size % child_size) != 0) {
        invalid = 1;
    }
    if (invalid) {
        if (world_rank == 0) {
            std::printf("Invalid args: expected_world_size=%d mpi_world_size=%d send_count=%lu iters=%u child_size=%d\n",
                        expected_world_size, world_size, static_cast<unsigned long>(send_count), iters, child_size);
            PrintUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int dev_count = 0;
    CUDACHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        std::printf("rank %d: no CUDA device available\n", world_rank);
        MPI_Finalize();
        return 1;
    }
    int device = LocalRank(world_rank) % dev_count;
    CUDACHECK(cudaSetDevice(device));

    if (world_rank == 0) {
        std::printf("ncclCommSplit AllGather MPI test: world=%d child_size=%d send_count=%lu iters=%u\n",
                    world_size, child_size, static_cast<unsigned long>(send_count), iters);
    }
    std::printf("rank %d: using cuda device %d\n", world_rank, device);

    ncclUniqueId unique_id;
    if (world_rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&unique_id));
    }
    MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t world_comm = nullptr;
    NCCLCHECK(ncclCommInitRank(&world_comm, world_size, unique_id, world_rank));

    cudaStream_t stream = nullptr;
    CUDACHECK(cudaStreamCreate(&stream));

    const size_t send_size = send_count * sizeof(float);
    const size_t max_recv_size = static_cast<size_t>(world_size) * send_size;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    void* host_send = nullptr;
    void* host_recv = nullptr;
    CUDACHECK(cudaMalloc(&send_buf, send_size));
    CUDACHECK(cudaMalloc(&recv_buf, max_recv_size));
    CUDACHECK(cudaMallocHost(&host_send, send_size));
    CUDACHECK(cudaMallocHost(&host_recv, max_recv_size));

    int local_failed = 0;

    local_failed |= RunAllGatherCase("world", MPI_COMM_WORLD, world_rank, 0, world_size, world_rank,
                                     send_count, iters, world_comm, stream, send_buf, recv_buf,
                                     host_send, host_recv);

    MPI_Barrier(MPI_COMM_WORLD);

    const int color = world_rank / child_size;
    const int key = world_rank;
    const int child_rank = world_rank % child_size;
    const int child_group_start = color * child_size;
    ncclComm_t child_comm = nullptr;
    if (world_rank == 0) {
        std::printf("calling ncclCommSplit: color=rank/%d, key=rank\n", child_size);
    }
    NCCLCHECK(ncclCommSplit(world_comm, color, key, &child_comm, nullptr));
    MPI_Comm mpi_child = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &mpi_child);

    local_failed |= RunAllGatherCase("child", mpi_child, world_rank, child_group_start, child_size,
                                     child_rank, send_count, iters, child_comm, stream, send_buf,
                                     recv_buf, host_send, host_recv);

    MPI_Comm_free(&mpi_child);
    NCCLCHECK(ncclCommDestroy(child_comm));
    NCCLCHECK(ncclCommDestroy(world_comm));

    CUDACHECK(cudaFreeHost(host_recv));
    CUDACHECK(cudaFreeHost(host_send));
    CUDACHECK(cudaFree(recv_buf));
    CUDACHECK(cudaFree(send_buf));
    CUDACHECK(cudaStreamDestroy(stream));

    int local_failed_flag = local_failed ? 1 : 0;
    int total_failed = 0;
    MPI_Allreduce(&local_failed_flag, &total_failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::printf("ncclCommSplit AllGather MPI test %s\n", total_failed ? "FAILED" : "PASSED");
    }
    MPI_Finalize();
    return total_failed ? 1 : 0;
}
