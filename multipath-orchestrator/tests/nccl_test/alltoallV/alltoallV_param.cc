#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>
#include <vector>

#define CUDACHECK(cmd) do { cudaError_t e = (cmd); if (e != cudaSuccess) { \
    printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); return -1; }} while (0)
#define NCCLCHECK(cmd) do { ncclResult_t r = (cmd); if (r != ncclSuccess) { \
    printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); return -1; }} while (0)

using NcclAllToAllVFn = int (*)(const void*, const size_t*, const size_t*, void*,
                                const size_t*, const size_t*, ncclDataType_t,
                                ncclComm_t, cudaStream_t);

struct ThreadContext {
    int rank;
    int nranks;
    size_t base_count;
    ncclUniqueId unique_id;
};

static size_t Count(size_t base, int src, int dst) {
    return base + static_cast<size_t>(src * 3 + dst);
}

static float Value(int src, int dst, size_t i) {
    return 1000000.0f * static_cast<float>(src) + 1000.0f * static_cast<float>(dst) +
           static_cast<float>(i);
}

static int Worker(ThreadContext ctx) {
    CUDACHECK(cudaSetDevice(ctx.rank));
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, ctx.nranks, ctx.unique_id, ctx.rank));

    auto fn = reinterpret_cast<NcclAllToAllVFn>(dlsym(RTLD_DEFAULT, "ncclAllToAllV"));
    if (!fn) {
        printf("rank %d: ncclAllToAllV symbol not found\n", ctx.rank);
        return -1;
    }

    std::vector<size_t> sendcounts(ctx.nranks), sdispls(ctx.nranks);
    std::vector<size_t> recvcounts(ctx.nranks), rdispls(ctx.nranks);
    size_t send_total = 0;
    size_t recv_total = 0;
    for (int peer = 0; peer < ctx.nranks; ++peer) {
        sdispls[peer] = send_total;
        sendcounts[peer] = Count(ctx.base_count, ctx.rank, peer);
        send_total += sendcounts[peer];
        rdispls[peer] = recv_total;
        recvcounts[peer] = Count(ctx.base_count, peer, ctx.rank);
        recv_total += recvcounts[peer];
    }

    std::vector<float> h_send(send_total);
    std::vector<float> h_recv(recv_total, -1.0f);
    for (int peer = 0; peer < ctx.nranks; ++peer) {
        for (size_t i = 0; i < sendcounts[peer]; ++i) {
            h_send[sdispls[peer] + i] = Value(ctx.rank, peer, i);
        }
    }

    float* d_send = nullptr;
    float* d_recv = nullptr;
    cudaStream_t stream;
    CUDACHECK(cudaMalloc(&d_send, send_total * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_recv, recv_total * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaMemcpyAsync(d_send, h_send.data(), send_total * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
    int rc = fn(d_send, sendcounts.data(), sdispls.data(), d_recv, recvcounts.data(),
                rdispls.data(), ncclFloat32, comm, stream);
    if (rc != 0) {
        printf("rank %d: ncclAllToAllV returned %d\n", ctx.rank, rc);
        return -1;
    }
    CUDACHECK(cudaMemcpyAsync(h_recv.data(), d_recv, recv_total * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    int failed = 0;
    for (int peer = 0; peer < ctx.nranks; ++peer) {
        for (size_t i = 0; i < recvcounts[peer]; ++i) {
            float expected = Value(peer, ctx.rank, i);
            float actual = h_recv[rdispls[peer] + i];
            if (std::fabs(expected - actual) > 1e-3f) {
                ++failed;
            }
        }
    }

    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_send));
    CUDACHECK(cudaFree(d_recv));
    NCCLCHECK(ncclCommDestroy(comm));
    printf("rank %d: %s\n", ctx.rank, failed == 0 ? "PASSED" : "FAILED");
    return failed == 0 ? 0 : -1;
}

int main(int argc, char** argv) {
    int nranks = argc > 1 ? std::atoi(argv[1]) : 2;
    size_t base_count = argc > 2 ? static_cast<size_t>(std::atoll(argv[2])) : 256;
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    std::vector<std::thread> threads;
    std::vector<int> results(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        threads.emplace_back([&, r] {
            results[r] = Worker(ThreadContext{r, nranks, base_count, id});
        });
    }
    int failed = 0;
    for (auto& t : threads) t.join();
    for (int r : results) failed += (r != 0);
    return failed == 0 ? 0 : 1;
}
