#include "stream_sync.h"
#include "domain_manager.h"
#include "domain.h"
#include "timer_query_pool.h"
#include "telemetry/stats.h"
#include "common/config.h"
#include "common/log.h"
#include "common/op_key.h"
#include "backend/memory/device_mem.h"
#include <optional>
#include <cstdint>
#include <thread>

#if defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#elif defined(AMPCCL_USE_CUDA_TIMER)
#include <cuda_runtime.h>
#endif

namespace ampccl {

namespace {

// Host callback threads may not inherit the app's current device; bind before stream/timer APIs.
int32_t EffectiveDeviceId(const CommDomain* d) {
    if (!d) {
        return -1;
    }
    if (d->pcie_device_id() >= 0) {
        return d->pcie_device_id();
    }
    if (!d->key.ranks.empty()) {
        return static_cast<int32_t>(d->key.ranks[0]);
    }
    return -1;
}

#if defined(AMPCCL_USE_ACL_TIMER)
struct RuntimeDeviceGuard {
    int32_t prev_ = -1;
    bool prev_ok_ = false;
    bool switched_ = false;

    explicit RuntimeDeviceGuard(int32_t target_id) {
        if (target_id < 0) {
            return;
        }
        int32_t cur = -1;
        aclError gr = aclrtGetDevice(&cur);
        prev_ok_ = (gr == ACL_SUCCESS);
        prev_ = cur;
        if (prev_ok_ && cur == target_id) {
            return;
        }
        if (aclrtSetDevice(target_id) != ACL_SUCCESS) {
            return;
        }
        switched_ = true;
    }
    ~RuntimeDeviceGuard() {
        if (!switched_) {
            return;
        }
        if (prev_ok_) {
            (void)aclrtSetDevice(prev_);
        }
    }
    RuntimeDeviceGuard(const RuntimeDeviceGuard&) = delete;
    RuntimeDeviceGuard& operator=(const RuntimeDeviceGuard&) = delete;
};
#elif defined(AMPCCL_USE_CUDA_TIMER)
struct RuntimeDeviceGuard {
    int prev_ = -1;
    bool prev_ok_ = false;
    bool switched_ = false;

    explicit RuntimeDeviceGuard(int32_t target_id) {
        if (target_id < 0) {
            return;
        }
        int cur = -1;
        cudaError_t gr = cudaGetDevice(&cur);
        prev_ok_ = (gr == cudaSuccess);
        prev_ = cur;
        if (prev_ok_ && cur == target_id) {
            return;
        }
        if (cudaSetDevice(target_id) != cudaSuccess) {
            return;
        }
        switched_ = true;
    }
    ~RuntimeDeviceGuard() {
        if (!switched_) {
            return;
        }
        if (prev_ok_) {
            (void)cudaSetDevice(prev_);
        }
    }
    RuntimeDeviceGuard(const RuntimeDeviceGuard&) = delete;
    RuntimeDeviceGuard& operator=(const RuntimeDeviceGuard&) = delete;
};
#endif

}  // namespace

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

    AMPCCL_LOG(INFO,
               "[Rank %d] OnStreamSynchronized begin: stream=%p fast_bytes=%zu pcie_bytes=%zu use_pcie=%d",
               domain->pcie_rank(), stream, pending->plan.fast_bytes, pending->plan.pcie_bytes,
               pending->plan.use_pcie);

#if defined(AMPCCL_USE_ACL_TIMER) || defined(AMPCCL_USE_CUDA_TIMER)
    const int32_t eff_dev = EffectiveDeviceId(domain);
    RuntimeDeviceGuard dev_guard(eff_dev);
#endif

    // Do not synchronize the user stream here: this callback is enqueued on that
    // same stream. Timer events are ordered before this callback; when PCIe is
    // used, the user stream also waits on the PCIe timer end event before the
    // callback is registered.

    if (pending->tmp_recvbuf && pending->user_recvbuf && stream) {
        const int nranks = pending->nranks;
        const size_t fast_bytes = pending->fast_bytes;
        const size_t pcie_bytes = pending->pcie_bytes;
        char* user = static_cast<char*>(pending->user_recvbuf);
        const char* tmp = static_cast<const char*>(pending->tmp_recvbuf);
        AMPCCL_LOG(INFO,
                   "[Rank %d] OnStreamSynchronized merge begin: op=%d nranks=%d fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), static_cast<int>(pending->op_key.op), nranks, fast_bytes,
                   pcie_bytes);

        // Do not call aclrtSynchronizeStream/cudaStreamSynchronize on `stream` here: this callback is
        // enqueued ON that stream via LaunchHostFunc; syncing the same stream from inside the callback
        // deadlocks.
        if (pending->op_key.op == CollectiveType::ReduceScatter) {
            if (pending->owns_tmp_recvbuf) {
                FreeDeviceBuffer(pending->tmp_recvbuf);
            }
        } else if ((pending->op_key.op == CollectiveType::AllGather ||
                pending->op_key.op == CollectiveType::AllToAll) && fast_bytes > 0) {
            const size_t per_rank = fast_bytes + pcie_bytes;
            for (int r = 0; r < nranks; ++r) {
                if (fast_bytes > 0) {
                    // Synchronous D2D (default/device memcpy path) — same as branch without `stream`;
                    // avoids async work + stream sync on the LaunchHostFunc stream.
                    DeviceMemcpyD2D(user + r * per_rank, tmp + r * fast_bytes, fast_bytes);
                }
            }
            if (pending->owns_tmp_recvbuf) {
                FreeDeviceBuffer(pending->tmp_recvbuf);
            }
        } else if (pending->owns_tmp_recvbuf) {
            FreeDeviceBuffer(pending->tmp_recvbuf);
        }
        AMPCCL_LOG(INFO, "[Rank %d] OnStreamSynchronized merge end", domain->pcie_rank());
    } else if (pending->tmp_recvbuf && pending->user_recvbuf) {
        const int nranks = pending->nranks;
        const size_t fast_bytes = pending->fast_bytes;
        const size_t pcie_bytes = pending->pcie_bytes;
        char* user = static_cast<char*>(pending->user_recvbuf);
        const char* tmp = static_cast<const char*>(pending->tmp_recvbuf);
        AMPCCL_LOG(INFO,
                   "[Rank %d] OnStreamSynchronized merge begin: op=%d nranks=%d fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), static_cast<int>(pending->op_key.op), nranks, fast_bytes,
                   pcie_bytes);
        if (pending->op_key.op == CollectiveType::ReduceScatter) {
            if (pending->owns_tmp_recvbuf) {
                FreeDeviceBuffer(pending->tmp_recvbuf);
            }
        } else if ((pending->op_key.op == CollectiveType::AllGather ||
                    pending->op_key.op == CollectiveType::AllToAll) && fast_bytes > 0) {
            const size_t per_rank = fast_bytes + pcie_bytes;
            for (int r = 0; r < nranks; ++r) {
                if (fast_bytes > 0) {
                    DeviceMemcpyD2D(user + r * per_rank, tmp + r * fast_bytes, fast_bytes);
                }
            }
            if (pending->owns_tmp_recvbuf) {
                FreeDeviceBuffer(pending->tmp_recvbuf);
            }
        } else if (pending->owns_tmp_recvbuf) {
            FreeDeviceBuffer(pending->tmp_recvbuf);
        }
        AMPCCL_LOG(INFO, "[Rank %d] OnStreamSynchronized merge end", domain->pcie_rank());
    }

    if (!Config::IsFeedbackEnabled()) {
        AMPCCL_LOG(INFO, "[Rank %d] StreamSync: feedback disabled, skip timer query and param update",
                   domain->pcie_rank());
        return;
    }

    ExecStat stat;

    double fast_time = 0.0;
    double pcie_time = 0.0;
    bool fast_elapsed_ok = true;
    bool pcie_elapsed_ok = true;
    bool fast_used_device_timer = false;
    bool pcie_used_device_timer = false;

#if defined(AMPCCL_USE_ACL_TIMER) || defined(AMPCCL_USE_CUDA_TIMER)
    {
        const Plan plan = pending->plan;
        const int rank = domain->pcie_rank();
        auto fast_timer = pending->fast_timer;
        auto pcie_timer = pending->pcie_timer;
        AMPCCL_LOG(INFO, "[Rank %d] timer query worker launch: effective_device=%d",
                   rank, static_cast<int>(eff_dev));
        TimerQueryPool::GetInstance().RunAndWait([fast_timer, pcie_timer, plan, eff_dev, rank,
                                                  &fast_time, &pcie_time,
                                                  &fast_elapsed_ok, &pcie_elapsed_ok,
                                                  &fast_used_device_timer, &pcie_used_device_timer]() {
            RuntimeDeviceGuard worker_dev_guard(eff_dev);
            AMPCCL_LOG(INFO, "[Rank %d] timer query worker begin", rank);
            if (plan.fast_bytes > 0 && fast_timer) {
                AMPCCL_LOG(INFO, "[Rank %d] timer_fast query begin", rank);
                bool ready = fast_timer->WaitUntilCompleteByQuery();
                fast_elapsed_ok = ready &&
                    fast_timer->ElapsedSecondsChecked(&fast_time, &fast_used_device_timer);
                AMPCCL_LOG(INFO,
                           "[Rank %d] timer_fast query end ready=%d elapsed_ok=%d source=%s time=%.9f",
                           rank, ready ? 1 : 0, fast_elapsed_ok ? 1 : 0,
                           fast_used_device_timer ? "device" : "cpu", fast_time);
            }
            if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_timer) {
                AMPCCL_LOG(INFO, "[Rank %d] timer_pcie query begin", rank);
                bool ready = pcie_timer->WaitUntilCompleteByQuery();
                pcie_elapsed_ok = ready &&
                    pcie_timer->ElapsedSecondsChecked(&pcie_time, &pcie_used_device_timer);
                AMPCCL_LOG(INFO,
                           "[Rank %d] timer_pcie query end ready=%d elapsed_ok=%d source=%s time=%.9f",
                           rank, ready ? 1 : 0, pcie_elapsed_ok ? 1 : 0,
                           pcie_used_device_timer ? "device" : "cpu", pcie_time);
            }
            AMPCCL_LOG(INFO, "[Rank %d] timer query worker end", rank);
        });
        AMPCCL_LOG(INFO, "[Rank %d] timer query worker joined", rank);
    }
#else
    if (pending->plan.fast_bytes > 0 && pending->fast_timer) {
        pending->fast_timer->MarkCompletedOnHost();
        fast_elapsed_ok = pending->fast_timer->ElapsedSecondsChecked(&fast_time, &fast_used_device_timer);
    }
    if (pending->plan.use_pcie && pending->plan.pcie_bytes > 0 && pending->pcie_timer) {
        pending->pcie_timer->MarkCompletedOnHost();
        pcie_elapsed_ok = pending->pcie_timer->ElapsedSecondsChecked(&pcie_time, &pcie_used_device_timer);
    }
#endif
    if (!fast_elapsed_ok || (pending->plan.fast_bytes > 0 && fast_time <= 0.0)) {
        AMPCCL_LOG(WARN, "[Rank %d] timer_fast elapsed invalid: ok=%d bytes=%zu time=%.9f",
                   domain->pcie_rank(), fast_elapsed_ok ? 1 : 0, pending->plan.fast_bytes, fast_time);
    }
    if (!pcie_elapsed_ok || (pending->plan.use_pcie && pending->plan.pcie_bytes > 0 && pcie_time <= 0.0)) {
        AMPCCL_LOG(WARN, "[Rank %d] timer_pcie elapsed invalid: ok=%d bytes=%zu time=%.9f",
                   domain->pcie_rank(), pcie_elapsed_ok ? 1 : 0, pending->plan.pcie_bytes, pcie_time);
    }

    stat.fast_time = fast_time;
    stat.pcie_time = pending->plan.use_pcie ? pcie_time : 0.0;
    stat.fast_bytes = pending->plan.fast_bytes;
    stat.pcie_bytes = pending->plan.pcie_bytes;
    stat.fast_success = pending->fast_success;
    stat.pcie_success = pending->pcie_success;

    domain->EnsureShmAttached();
    int nranks = domain->pcie_nranks();
    ShmParamStore* shm = domain->shm_store();
    if (nranks > 1 && shm->IsAttached()) {
        AMPCCL_LOG(INFO, "[Rank %d] WriteMyStat begin: op_key.bytes=%zu", domain->pcie_rank(),
                   pending->op_key.bytes);
        shm->WriteMyStat(domain->pcie_rank(), pending->op_key, stat);
        AMPCCL_LOG(INFO, "[Rank %d] StreamSync: wrote stat to shm op_key.bytes=%zu fast_time=%.6fs pcie_time=%.6fs",
                   domain->pcie_rank(), pending->op_key.bytes, stat.fast_time, stat.pcie_time);
    } else if (nranks > 1 && !Config::IsShmParamSyncEnabled()) {
        AMPCCL_LOG(OFF, "[Rank %d] StreamSync: SHM params off, skip Update (multi-thread / single-process)",
                   domain->pcie_rank());
    } else {
        domain->controller->Update(pending->op_key, stat, domain->param_cache);
        AMPCCL_LOG(INFO, "[Rank %d] StreamSync: op_key.bytes=%zu fast_time=%.6fs pcie_time=%.6fs fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), pending->op_key.bytes, stat.fast_time, stat.pcie_time,
                   stat.fast_bytes, stat.pcie_bytes);
    }
    AMPCCL_LOG(INFO, "[Rank %d] OnStreamSynchronized end", domain->pcie_rank());
}

}  // namespace ampccl
