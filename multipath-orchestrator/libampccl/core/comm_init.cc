#include "comm_init.h"
#include "core/domain.h"
#include "common/config.h"
#include "backend/memory/device_mem.h"

#ifdef AMPCCL_ENABLE_PCIE
#include <atomic>
#include <cstdint>
#include <cstring>
#include <pccl/pccl.h>
#if defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#elif defined(AMPCCL_USE_CUDA_TIMER)
#include <cuda_runtime.h>
#endif

namespace {

std::atomic<int> g_pccl_comm_live{0};

bool EnvIsTruthy(const char* val) {
    if (val == nullptr || val[0] == '\0') {
        return false;
    }
    return std::strcmp(val, "1") == 0 || std::strcmp(val, "true") == 0 ||
           std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
           std::strcmp(val, "YES") == 0;
}

// Deterministic digest as32 hex chars (ASCII, no embedded NUL) for pccl strnlen-based id parsing.
void HashBootstrapToPcclUniqueId(const void* data, size_t len, pcclUniqueId* out) {
    std::memset(out, 0, sizeof(*out));
    uint64_t h1 = 1469598103934665603ULL;
    uint64_t h2 = 982451653ULL;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; ++i) {
        h1 ^= p[i];
        h1 *= 1099511628211ULL;
        h2 ^= static_cast<uint64_t>(p[i]) * (0xD6E8FEB866B47D05ULL + i);
        h2 = h2 * 0x9E3779B97F4A7C15ULL + h1;
    }
    h1 ^= static_cast<uint64_t>(len);
    h2 ^= static_cast<uint64_t>(len) << 32;
    (void)std::snprintf(out->internal, sizeof(out->internal), "%016llx%016llx",
                        static_cast<unsigned long long>(h1),
                        static_cast<unsigned long long>(h2));
}

void FillPcclUniqueId(const void* bootstrap, size_t bootstrap_len, pcclUniqueId* out) {
    std::memset(out, 0, sizeof(*out));
    if (EnvIsTruthy(std::getenv("AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID"))) {
        return;
    }
    const char* env_override = std::getenv("AMPCCL_PCCL_UNIQUE_ID");
    if (env_override != nullptr && env_override[0] != '\0') {
        size_t n = std::strlen(env_override);
        if (n > static_cast<size_t>(PCCL_UNIQUE_ID_BYTES)) {
            n = PCCL_UNIQUE_ID_BYTES;
        }
        std::memcpy(out->internal, env_override, n);
        return;
    }
    if (bootstrap == nullptr || bootstrap_len == 0) {
        return;
    }
    HashBootstrapToPcclUniqueId(bootstrap, bootstrap_len, out);
}

}  // namespace

#endif  // AMPCCL_ENABLE_PCIE

namespace ampccl {

void InitPCIeForDomain(CommDomain* domain, int rank, int nranks, const void* pccl_bootstrap,
                       size_t pccl_bootstrap_len) {
    AMPCCL_LOG(INFO, "[Rank %d] InitPCIeForDomain begin", rank);
    if (!domain || nranks < 2 || rank < 0 || rank >= nranks) {
        return;
    }
#ifdef AMPCCL_ENABLE_PCIE
    domain->set_pcie_rank(rank);
    domain->set_pcie_nranks(nranks);
    if (Config::IsPCIeEnabled()) {
        pcclUniqueId pccl_id{};
        FillPcclUniqueId(pccl_bootstrap, pccl_bootstrap_len, &pccl_id);
        pcclComm_t pcie_comm = nullptr;
        pcclResult_t ret = pcclCommInitRank(&pcie_comm, nranks, pccl_id, rank);

        if (ret != pcclSuccess || !pcie_comm) {
            AMPCCL_LOG(WARN, "[Rank %d] pcclCommInitRank failed: %d", rank, static_cast<int>(ret));
            return;
        }
        domain->set_pcie_comm(pcie_comm);
        g_pccl_comm_live.fetch_add(1, std::memory_order_relaxed);

#if defined(AMPCCL_USE_ACL_TIMER)
        aclrtStream pcie_stream = nullptr;
        aclError stream_ret = aclrtCreateStream(&pcie_stream);

        if (stream_ret == ACL_SUCCESS && pcie_stream) {
            domain->set_pcie_stream(pcie_stream);
            int32_t dev_id = -1;
            if (aclrtGetDevice(&dev_id) == ACL_SUCCESS) {
                domain->set_pcie_device_id(dev_id);
            }
            AMPCCL_LOG(INFO, "[Rank %d] InitPCIeForDomain end: pcie_comm %p, pcie_stream %p device=%d", rank,
                       static_cast<void*>(pcie_comm), static_cast<void*>(pcie_stream),
                       static_cast<int>(domain->pcie_device_id()));
        }
#elif defined(AMPCCL_USE_CUDA_TIMER)
        cudaStream_t pcie_stream = nullptr;
        if (cudaStreamCreate(&pcie_stream) == cudaSuccess && pcie_stream) {
            domain->set_pcie_stream(pcie_stream);
            int dev_id = -1;
            if (cudaGetDevice(&dev_id) == cudaSuccess) {
                domain->set_pcie_device_id(static_cast<int32_t>(dev_id));
            }
            AMPCCL_LOG(INFO, "[Rank %d] InitPCIeForDomain end: pcie_comm %p, pcie_stream %p device=%d", rank,
                       static_cast<void*>(pcie_comm), static_cast<void*>(pcie_stream),
                       static_cast<int>(domain->pcie_device_id()));
        }
#endif
#if defined(AMPCCL_BUILD_NCCL)
        // NCCL hybrid AllGather/ReduceScatter still use a staging buffer when V APIs are unused.
        size_t recvbuf_size = Config::GetMaxPCIeRecvBufBytes();
        if (recvbuf_size > 0) {
            void* recvbuf = AllocDeviceBuffer(recvbuf_size);
            if (recvbuf) {
                domain->set_pcie_recvbuf(recvbuf);
                domain->set_pcie_recvbuf_size(recvbuf_size);
                AMPCCL_LOG(INFO, "[Rank %d] InitPCIeForDomain: pcie_recvbuf %p size=%zu", rank, recvbuf,
                           recvbuf_size);
            }
        }
#endif
    }
#endif
}

void EnsurePCIeForDomain(CommDomain* domain) {
    if (!domain || domain->pcie_comm() != nullptr) {
        return;
    }
    int rank = -1;
    int nranks = 0;
    std::vector<uint8_t> bootstrap;
    if (!domain->BeginPendingPCIeInit(&rank, &nranks, &bootstrap)) {
        return;
    }
    AMPCCL_LOG(INFO, "[Rank %d] EnsurePCIeForDomain lazy init begin: nranks=%d bootstrap_bytes=%zu",
               rank, nranks, bootstrap.size());
    const void* bootstrap_ptr = bootstrap.empty() ? nullptr : bootstrap.data();
    InitPCIeForDomain(domain, rank, nranks, bootstrap_ptr, bootstrap.size());
    AMPCCL_LOG(INFO, "[Rank %d] EnsurePCIeForDomain lazy init end: pcie_comm=%p pcie_stream=%p",
               rank, domain->pcie_comm(), domain->pcie_stream());
    domain->FinishPendingPCIeInit();
}

void DestroyPCIeForDomain(CommDomain* domain) {
    if (!domain) {
        return;
    }
#ifdef AMPCCL_ENABLE_PCIE
    int rank = domain->pcie_rank();
    pcclComm_t pcie_comm = static_cast<pcclComm_t>(domain->pcie_comm());
    void* pcie_stream = domain->pcie_stream();
    AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain begin: pcie_comm=%p pcie_stream=%p",
               rank, static_cast<void*>(pcie_comm), pcie_stream);
    if (pcie_stream) {
#if defined(AMPCCL_USE_ACL_TIMER)
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcie stream synchronize begin: stream=%p",
                   rank, pcie_stream);
        aclError sync_ret = aclrtSynchronizeStream(static_cast<aclrtStream>(pcie_stream));
        AMPCCL_LOG(INFO,
                   "[Rank %d] DestroyPCIeForDomain pcie stream synchronize end: ret=%d stream=%p",
                   rank, static_cast<int>(sync_ret), pcie_stream);
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcie stream destroy begin: stream=%p",
                   rank, pcie_stream);
        aclError destroy_ret = aclrtDestroyStream(static_cast<aclrtStream>(pcie_stream));
        AMPCCL_LOG(INFO,
                   "[Rank %d] DestroyPCIeForDomain pcie stream destroy end: ret=%d stream=%p",
                   rank, static_cast<int>(destroy_ret), pcie_stream);
#elif defined(AMPCCL_USE_CUDA_TIMER)
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcie stream synchronize begin: stream=%p",
                   rank, pcie_stream);
        cudaError_t sync_ret = cudaStreamSynchronize(static_cast<cudaStream_t>(pcie_stream));
        AMPCCL_LOG(INFO,
                   "[Rank %d] DestroyPCIeForDomain pcie stream synchronize end: ret=%d stream=%p",
                   rank, static_cast<int>(sync_ret), pcie_stream);
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcie stream destroy begin: stream=%p",
                   rank, pcie_stream);
        cudaError_t destroy_ret = cudaStreamDestroy(static_cast<cudaStream_t>(pcie_stream));
        AMPCCL_LOG(INFO,
                   "[Rank %d] DestroyPCIeForDomain pcie stream destroy end: ret=%d stream=%p",
                   rank, static_cast<int>(destroy_ret), pcie_stream);
#endif
        domain->set_pcie_stream(nullptr);
    }
    void* pcie_recvbuf = domain->pcie_recvbuf();
    if (pcie_recvbuf) {
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcie recvbuf free begin: recvbuf=%p size=%zu",
                   rank, pcie_recvbuf, domain->pcie_recvbuf_size());
        FreeDeviceBuffer(pcie_recvbuf);
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcie recvbuf free end", rank);
        domain->set_pcie_recvbuf(nullptr);
        domain->set_pcie_recvbuf_size(0);
    }
    if (pcie_comm) {
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcclCommDestroy begin: pcie_comm=%p",
                   rank, static_cast<void*>(pcie_comm));
        pcclResult_t destroy_ret = pcclCommDestroy(pcie_comm);
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcclCommDestroy end: ret=%d pcie_comm=%p",
                   rank, static_cast<int>(destroy_ret), static_cast<void*>(pcie_comm));
        domain->set_pcie_comm(nullptr);
        int left = g_pccl_comm_live.fetch_sub(1, std::memory_order_acq_rel) - 1;
        AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pccl live comms after destroy=%d",
                   rank, left);
        if (left == 0) {
            AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcclFinalize begin", rank);
            pcclFinalize();
            AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain pcclFinalize end", rank);
        }
    }
    domain->set_pcie_rank(-1);
    domain->set_pcie_nranks(0);
    AMPCCL_LOG(INFO, "[Rank %d] DestroyPCIeForDomain end", rank);
#endif
}

}  // namespace ampccl
