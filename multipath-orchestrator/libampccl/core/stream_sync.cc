#include "stream_sync.h"
#include "domain_manager.h"
#include "domain.h"
#include "telemetry/stats.h"
#include "common/log.h"
#include "common/op_key.h"
#include "backend/memory/device_mem.h"
#include <optional>
#include <cstdlib>
#include <cstdio>

#ifdef AMPCCL_ENABLE_PCIE
#include "comm.hpp"
#endif

namespace ampccl {

void OnStreamSynchronized(void* stream) {
    std::optional<PendingCollective> pending =
        DomainManager::GetInstance().TakeStreamPending(stream);
    if (!pending) {
        return;
    }
    CommDomain* domain = pending->domain;
    if (!domain || !domain->controller) {
        return;
    }

#ifdef AMPCCL_ENABLE_PCIE
    if (domain->pcie_comm() && domain->pcie_stream() && pending->plan.use_pcie) {
        SyncDeviceStreamRaw(domain->pcie_stream());
        pcclSynchronizeInternalStreams(static_cast<pcclComm_t>(domain->pcie_comm()));
        AMPCCL_LOG(INFO,"[Rank %d] PCIe stream sync finished", domain->pcie_rank());
    }
#endif

    domain->timer_fast().Synchronize();
    if (pending->plan.use_pcie) {
        domain->timer_pcie().Synchronize();
    }

    if (pending->tmp_recvbuf && pending->user_recvbuf && stream) {
        const int nranks = pending->nranks;
        const size_t fast_bytes = pending->fast_bytes;
        const size_t pcie_bytes = pending->pcie_bytes;
        char* user = static_cast<char*>(pending->user_recvbuf);
        const char* tmp = static_cast<const char*>(pending->tmp_recvbuf);

        if (std::getenv("AMPCCL_DEBUG_MERGE") && pending->op_key.op == CollectiveType::AllGather &&
            domain->pcie_rank() ==1) {
            const size_t total_bytes = static_cast<size_t>(nranks) * (fast_bytes + pcie_bytes);
            const size_t total_floats = total_bytes / 4u;
            if (total_floats > 0 && total_floats <= 1024u) {
                float host_buf[1024];
                // printf("111111111");
                DeviceMemcpyD2H(host_buf, tmp, total_bytes);
                // printf("222222222");
                // SyncDeviceRaw();
                std::fprintf(stderr, "[AMPCCL_DEBUG_MERGE] Rank0 BEFORE merge, tmp all [0..%zu]: ", total_floats - 1);
                for (size_t i = 0; i < total_floats; ++i) {
                    std::fprintf(stderr, "%.0f ", static_cast<double>(host_buf[i]));
                }
                std::fprintf(stderr, "\n");
            }
        }

        if (pending->op_key.op == CollectiveType::AllReduce) {
            if (fast_bytes > 0) {
                DeviceMemcpyD2DAsync(user, tmp, fast_bytes, stream);
            }
            if (pcie_bytes > 0) {
                DeviceMemcpyD2DAsync(user + fast_bytes, tmp + fast_bytes, pcie_bytes, stream);
            }
        } else if (pending->op_key.op == CollectiveType::AllGather) {
            const size_t per_rank = fast_bytes + pcie_bytes;
            const size_t tmp_fast_total = static_cast<size_t>(nranks) * fast_bytes;
            for (int r = 0; r < nranks; ++r) {
                if (fast_bytes > 0) {
                    DeviceMemcpyD2DAsync(user + r * per_rank, tmp + r * fast_bytes, fast_bytes, stream);
                }
                if (pcie_bytes > 0) {
                    DeviceMemcpyD2DAsync(user + r * per_rank + fast_bytes,
                                        tmp + tmp_fast_total + r * pcie_bytes, pcie_bytes, stream);
                }
            }
        }
        SyncDeviceStreamRaw(stream);

        if (std::getenv("AMPCCL_DEBUG_MERGE") && pending->op_key.op == CollectiveType::AllGather &&
            domain->pcie_rank() ==1) {
            const size_t total_bytes = static_cast<size_t>(nranks) * (fast_bytes + pcie_bytes);
            const size_t total_floats = total_bytes / 4u;
            if (total_floats > 0 && total_floats <= 1024u) {
                float host_buf[1024];
                DeviceMemcpyD2H(host_buf, user, total_bytes);
                // SyncDeviceRaw();
                std::fprintf(stderr, "[AMPCCL_DEBUG_MERGE] Rank0 AFTER merge, user all [0..%zu]: ", total_floats - 1);
                for (size_t i = 0; i < total_floats; ++i) {
                    std::fprintf(stderr, "%.0f ", static_cast<double>(host_buf[i]));
                }
                std::fprintf(stderr, "\n");
            }
        }

        if (pending->owns_tmp_recvbuf) {
            FreeDeviceBuffer(pending->tmp_recvbuf);
        }
    } else if (pending->tmp_recvbuf && pending->user_recvbuf) {
        const int nranks = pending->nranks;
        const size_t fast_bytes = pending->fast_bytes;
        const size_t pcie_bytes = pending->pcie_bytes;
        char* user = static_cast<char*>(pending->user_recvbuf);
        const char* tmp = static_cast<const char*>(pending->tmp_recvbuf);
        if (pending->op_key.op == CollectiveType::AllReduce) {
            if (fast_bytes > 0) DeviceMemcpyD2D(user, tmp, fast_bytes);
            if (pcie_bytes > 0) DeviceMemcpyD2D(user + fast_bytes, tmp + fast_bytes, pcie_bytes);
        } else if (pending->op_key.op == CollectiveType::AllGather) {
            const size_t per_rank = fast_bytes + pcie_bytes;
            const size_t tmp_fast_total = static_cast<size_t>(nranks) * fast_bytes;
            for (int r = 0; r < nranks; ++r) {
                if (fast_bytes > 0) DeviceMemcpyD2D(user + r * per_rank, tmp + r * fast_bytes, fast_bytes);
                if (pcie_bytes > 0) DeviceMemcpyD2D(user + r * per_rank + fast_bytes,
                                                   tmp + tmp_fast_total + r * pcie_bytes, pcie_bytes);
            }
        }
        if (pending->owns_tmp_recvbuf) {
            FreeDeviceBuffer(pending->tmp_recvbuf);
        }
    }

    ExecStat stat;
    
    stat.fast_time = domain->timer_fast().ElapsedSeconds();
    stat.pcie_time = pending->plan.use_pcie ? domain->timer_pcie().ElapsedSeconds() : 0.0;
    stat.fast_bytes = pending->plan.fast_bytes;
    stat.pcie_bytes = pending->plan.pcie_bytes;
    stat.fast_success = pending->fast_success;
    stat.pcie_success = pending->pcie_success;

    domain->EnsureShmAttached();
    int nranks = domain->pcie_nranks();
    ShmParamStore* shm = domain->shm_store();
    if (nranks > 1 && shm->IsAttached()) {
        shm->WriteMyStat(domain->pcie_rank(), pending->op_key, stat);
        AMPCCL_LOG(INFO, "[Rank %d] StreamSync: wrote stat to shm op_key.bytes=%zu fast_time=%.6fs pcie_time=%.6fs",
                   domain->pcie_rank(), pending->op_key.bytes, stat.fast_time, stat.pcie_time);
    } else {
        domain->controller->Update(pending->op_key, stat, domain->param_cache);
        AMPCCL_LOG(INFO, "[Rank %d] StreamSync: op_key.bytes=%zu fast_time=%.6fs pcie_time=%.6fs fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), pending->op_key.bytes, stat.fast_time, stat.pcie_time,
                   stat.fast_bytes, stat.pcie_bytes);
    }
}

}  // namespace ampccl
