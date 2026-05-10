#ifndef AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_
#define AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_

#include "domain.h"
#include "domain_manager.h"
#include "comm_init.h"
#include "planner.h"
#include "common/op_key.h"
#include "common/size_bucket.h"
#include "backend/fast_backend.h"
#include "backend/pcie_backend.h"
#include "backend/memory/device_mem.h"
#include "backend/original_collective_api.h"
#include "backend/dtype/native_dtype.h"
#include <cstdint>
#include <vector>
#include "telemetry/stats.h"
#include "common/log.h"
#include "stream_host_callback.h"
#include <cstddef>
#include <cstring>
#include <memory>

namespace ampccl {

namespace {

inline OpKey MakeOpKey(CommDomain* domain, CollectiveType op, size_t bytes, int datatype) {
    OpKey k;
    k.op = op;
    k.bytes = bytes;
    k.bucket_bytes = SizeBucket::BucketBytes(bytes);
    k.datatype = datatype;
    k.nranks = domain->pcie_nranks();
    k.topology_hash = domain->logical_key.comm_hash;
    return k;
}

inline void ShmRoundBegin(CommDomain* domain) {
    domain->EnsureShmAttached();
    ShmParamStore* shm = domain->shm_store();
    if (shm->IsAttached() && shm->IsRank0()) {
        shm->WaitUntilParamConsumed();
        ExecStat global_stat;
        OpKey agg_op_key;
        double algbw = 0.0, busbw = 0.0, bus_factor = 0.0;
        if (shm->ReadAllStatsAndAggregate(&global_stat, &agg_op_key,
                                          &algbw, &busbw, &bus_factor) && domain->controller) {
            domain->RecordBw(algbw, busbw, bus_factor);
            domain->controller->Update(agg_op_key, global_stat, domain->param_cache);
            shm->WriteParams(domain->param_cache, agg_op_key);
        }
        shm->SignalParamWritten();
    }
    if (shm->IsAttached()) {
        shm->ReadParams(&domain->param_cache);
    }
}

struct CollectiveLaunchPlan {
    double alpha = 0.0;
    Plan plan;
    bool cache_hit = false;
};

inline CollectiveLaunchPlan PrepareCollectiveLaunch(CommDomain* domain, const OpKey& op_key) {
    CollectiveLaunchPlan launch;
    if (!domain || !domain->controller) {
        launch.alpha = Config::GetStaticAlpha();
        launch.plan = Planner::CreatePlan(op_key.bytes, launch.alpha, Config::IsPCIeEnabled());
        return launch;
    }

    launch.cache_hit = domain->param_cache.Contains(op_key);

    if (!Config::IsFeedbackEnabled()) {
        launch.alpha = Config::GetStaticAlpha();
        launch.plan = Planner::CreatePlan(op_key.bytes, launch.alpha, Config::IsPCIeEnabled());
        return launch;
    }

    ShmRoundBegin(domain);

    if (domain->shm_store()->IsAttached()) {
        domain->shm_store()->BarrierSync();
    }

    ParamValue param = domain->param_cache.Lookup(op_key);
    launch.alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);
    launch.plan = Planner::CreatePlan(op_key.bytes, launch.alpha, param.use_pcie);
    return launch;
}

inline void ForceFastOnly(Plan* plan, size_t fast_bytes) {
    if (!plan) {
        return;
    }
    plan->fast_bytes = fast_bytes;
    plan->pcie_bytes = 0;
    plan->use_pcie = false;
}

}  // namespace

// The PCIe stream consumes the user's send buffer independently from the fast
// backend. Publish input readiness before either backend is launched so both
// paths can start after prior user-stream work without waiting on each other.
inline bool PcieWaitForUserInputIfNeeded(CommDomain* domain, void* user_stream,
                                         ValueSyncSubmission* value_sync) {
    if (!domain || !user_stream || !domain->pcie_stream()) {
        return false;
    }
    if (value_sync && value_sync->EnqueueInputReady(user_stream) &&
        value_sync->EnqueueInputWait(domain->pcie_stream())) {
        AMPCCL_LOG(INFO, "[Rank %d] Pcie input dependency: value wait", domain->pcie_rank());
        return true;
    }

    // Allocation/runtime failures are exceptional. A host-side sync preserves
    // correctness, while the normal value-token path remains fully concurrent.
    AMPCCL_LOG(WARN, "[Rank %d] Pcie input value sync unavailable; synchronizing user stream",
               domain->pcie_rank());
    return OriginalCollectiveAPI::Synchronize(user_stream).IsOk();
}

// After PCIe work, use a device token to make the user stream observe PCIe
// completion. Timer events remain a correctness fallback when value sync is
// unavailable or rejected by the runtime.
inline void PcieCompleteOnUserStreamIfNeeded(CommDomain* domain, void* user_stream, const Plan& plan,
                                             ValueSyncSubmission* value_sync, Timer* pcie_timer) {
    if (plan.use_pcie && plan.pcie_bytes > 0 && domain->pcie_stream() && user_stream && pcie_timer) {
        AMPCCL_LOG(INFO,
                   "[Rank %d] Pcie completion dependency begin: user_stream=%p pcie_stream=%p fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), user_stream, domain->pcie_stream(), plan.fast_bytes,
                   plan.pcie_bytes);
        if (value_sync && value_sync->EnqueueWrite(domain->pcie_stream()) &&
            value_sync->EnqueueWait(user_stream)) {
            AMPCCL_LOG(INFO, "[Rank %d] Pcie completion dependency end: value wait", domain->pcie_rank());
            return;
        }
        AMPCCL_LOG(WARN, "[Rank %d] Pcie value sync unavailable; falling back to event wait",
                   domain->pcie_rank());
        pcie_timer->WaitEventOnStream(user_stream, domain->pcie_rank());
        AMPCCL_LOG(INFO, "[Rank %d] Pcie completion dependency end: event wait", domain->pcie_rank());
    }
}

inline void SyncPCIeStreamBeforeCallbackIfEnabled(CommDomain* domain, void* pcie_stream,
                                                  const Plan& plan, const char* op_name) {
    if (!Config::IsPCIeStreamSyncBeforeCallbackEnabled() || !plan.use_pcie ||
        plan.pcie_bytes == 0 || !pcie_stream) {
        return;
    }
    AMPCCL_LOG(INFO,
               "[Rank %d] %s: explicit PCIe stream synchronize begin: pcie_stream=%p pcie_bytes=%zu",
               domain->pcie_rank(), op_name, pcie_stream, plan.pcie_bytes);
    BackendResult r = ampccl::OriginalCollectiveAPI::Synchronize(pcie_stream);
    AMPCCL_LOG(INFO, "[Rank %d] %s: explicit PCIe stream synchronize end result=%d",
               domain->pcie_rank(), op_name, static_cast<int>(r.code));
}

inline void EnqueueSegmentedMergeOnStream(void* stream, void* user_recvbuf, void* tmp_recvbuf,
                                          int nranks, size_t fast_bytes, size_t pcie_bytes,
                                          bool owns_tmp_recvbuf) {
    if (!stream || !user_recvbuf || !tmp_recvbuf || nranks <= 0 || fast_bytes == 0) {
        if (owns_tmp_recvbuf) {
            FreeDeviceBufferAsync(tmp_recvbuf, stream);
        }
        return;
    }

    char* user = static_cast<char*>(user_recvbuf);
    const char* tmp = static_cast<const char*>(tmp_recvbuf);
    const size_t per_rank = fast_bytes + pcie_bytes;
    for (int r = 0; r < nranks; ++r) {
        DeviceMemcpyD2DAsync(user + static_cast<size_t>(r) * per_rank,
                             tmp + static_cast<size_t>(r) * fast_bytes,
                             fast_bytes, stream);
    }
    if (owns_tmp_recvbuf) {
        FreeDeviceBufferAsync(tmp_recvbuf, stream);
    }
}

class VirtualCollective {
public:
    static BackendResult AllReduce(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* comm,
        void* stream) {
        size_t es = NativeDataTypeElemSize(datatype);
        if (es == 0) {
            return BackendResult::UnhandledError;
        }
        OpKey op_key = MakeOpKey(domain, CollectiveType::AllReduce, count * es, datatype);

        CollectiveLaunchPlan launch = PrepareCollectiveLaunch(domain, op_key);
        double alpha = launch.alpha;
        Plan plan = launch.plan;
        if (plan.use_pcie && !PCIeBackendImpl::SupportsAllReduce(datatype, op)) {
            AMPCCL_LOG(WARN,
                       "[Rank %d] AllReduce PCIe disabled: public PCCL API supports only fp32/sum",
                       domain->pcie_rank());
            ForceFastOnly(&plan, op_key.bytes);
        }

        BackendResult fast_r;
        BackendResult pcie_r;
        void* pcie_stream = domain->pcie_stream();
        auto fast_timer = std::make_shared<Timer>();
        auto pcie_timer = std::make_shared<Timer>();

        AMPCCL_LOG(INFO, "[Rank %d] Before AllReduce, alpha=%.4f, plan.use_pcie=%d, plan.pcie_bytes=%lu, plan.fast_bytes=%lu, "
            "domain_hash=%016llx",
            domain->pcie_rank(), alpha, plan.use_pcie, plan.pcie_bytes,  plan.fast_bytes,
            (unsigned long long)domain->logical_key.comm_hash);

        if (!plan.use_pcie || plan.pcie_bytes == 0 || !pcie_stream) {
            if (!Config::IsFastOnlyBwLogEnabled()) {
                return FastBackendImpl::AllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
            }
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::AllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                               fast_r.IsOk(), true);
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
            EnqueueCollectiveCompletionCallback(stream);
            if (!fast_r.IsOk()) {
                return fast_r;
            }
            return BackendResult::Success;
        }

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream) {
            size_t pcie_off = plan.fast_bytes;
            auto value_sync = domain->BeginValueSyncSubmission(stream);
            if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                return BackendResult::UnhandledError;
            }
            if (plan.fast_bytes > 0) {
                fast_timer->Start(stream);
                fast_r = FastBackendImpl::AllReduce(
                               const_cast<void*>(sendbuff), recvbuff, plan.fast_bytes / es, datatype, op,
                               comm, stream);
                fast_timer->Stop(stream);
            }
            if (plan.pcie_bytes > 0) {
                pcie_timer->Start(pcie_stream);
                const char* ps = static_cast<const char*>(sendbuff) + pcie_off;
                char* pr = static_cast<char*>(recvbuff) + pcie_off;
                pcie_r = PCIeBackendImpl::AllReduce(
                               domain, ps, pr, plan.pcie_bytes / es, datatype, op, pcie_stream);
                pcie_timer->Stop(pcie_stream);
                if (!pcie_r.IsOk()) {
                    AMPCCL_LOG(ERROR, "[Rank %d] AllReduce PCIe FAILED: domain_hash=%016llx pcie_bytes=%zu",
                               domain->pcie_rank(),
                               (unsigned long long)domain->logical_key.comm_hash,
                               plan.pcie_bytes);
                }
            }
            PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
            SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllReduce");
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                               fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        } else {
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::AllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                                                         stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                               fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        }
        if (!fast_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] AllReduce fast path failed: code=%d original=%d",
                       domain->pcie_rank(), static_cast<int>(fast_r.code), fast_r.original);
            return fast_r;
        }
        if (!pcie_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] AllReduce pcie path failed: code=%d original=%d",
                       domain->pcie_rank(), static_cast<int>(pcie_r.code), pcie_r.original);
            return pcie_r;
        }
        return BackendResult::Success;
    }

    static BackendResult AllGather(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* comm,
        void* stream) {
        size_t es = NativeDataTypeElemSize(datatype);
        if (es == 0) {
            return BackendResult::UnhandledError;
        }
        OpKey op_key = MakeOpKey(domain, CollectiveType::AllGather, sendcount * es, datatype);

        CollectiveLaunchPlan launch = PrepareCollectiveLaunch(domain, op_key);
        double alpha = launch.alpha;
        Plan plan = launch.plan;
        if (plan.use_pcie && !PCIeBackendImpl::SupportsAllGather(datatype)) {
            AMPCCL_LOG(WARN,
                       "[Rank %d] AllGather PCIe disabled: no PCCL typed mapping for HCCL datatype=%d",
                       domain->pcie_rank(), datatype);
            ForceFastOnly(&plan, op_key.bytes);
        }

        BackendResult fast_r;
        BackendResult pcie_r;
        void* pcie_stream = domain->pcie_stream();
        auto fast_timer = std::make_shared<Timer>();
        auto pcie_timer = std::make_shared<Timer>();
        const int nranks = domain->pcie_nranks();

        AMPCCL_LOG(INFO, "[Rank %d] Before AllGather, op=%d bytes=%zu bucket_bytes=%zu dtype=%d nranks=%d cache_hit=%d "
            "alpha=%.4f, plan.use_pcie=%d, plan.pcie_bytes=%lu, plan.fast_bytes=%lu, domain_hash=%016llx",
            domain->pcie_rank(), static_cast<int>(op_key.op), op_key.bytes, op_key.bucket_bytes,
            op_key.datatype, op_key.nranks, launch.cache_hit ? 1 : 0, alpha, plan.use_pcie,
            plan.pcie_bytes, plan.fast_bytes,
            (unsigned long long)domain->logical_key.comm_hash);

        if (!plan.use_pcie || plan.pcie_bytes == 0 || !pcie_stream) {
            if (!Config::IsFastOnlyBwLogEnabled()) {
                return FastBackendImpl::AllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
            }
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::AllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                               fast_r.IsOk(), true);
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
            EnqueueCollectiveCompletionCallback(stream);
            if (!fast_r.IsOk()) {
                return fast_r;
            }
            return BackendResult::Success;
        }

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream && plan.fast_bytes > 0) {
            const size_t fast_elems = plan.fast_bytes / es;
            const size_t per_rank_elems = sendcount;
            bool used_fast_v = false;
            if (OriginalCollectiveAPI::HasFastAllGatherV()) {
                // Fast AllGatherV expects recvCounts/recvDispls as HOST pointers.
                // Build them on the stack.
                std::vector<size_t> recv_counts_host(static_cast<size_t>(nranks));
                std::vector<size_t> recv_displs_host(static_cast<size_t>(nranks));
                for (int j = 0; j < nranks; ++j) {
                    recv_counts_host[static_cast<size_t>(j)] = fast_elems;
                    recv_displs_host[static_cast<size_t>(j)] =
                        static_cast<size_t>(j) * per_rank_elems;
                }

                auto value_sync = domain->BeginValueSyncSubmission(stream);
                if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                    return BackendResult::UnhandledError;
                }
                fast_timer->Start(stream);
                fast_r = OriginalCollectiveAPI::FastAllGatherV(
                         sendbuff, fast_elems, recvbuff, recv_counts_host.data(),
                         recv_displs_host.data(), datatype, comm, stream);
                fast_timer->Stop(stream);

                pcie_timer->Start(pcie_stream);
                pcie_r = PCIeBackendImpl::AllGatherStrided(
                               domain, sendbuff, recvbuff, sendcount, fast_elems, datatype, pcie_stream);
                pcie_timer->Stop(pcie_stream);

                PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
                SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllGather");

                DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(),
                                                                   pcie_r.IsOk());
                DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                EnqueueCollectiveCompletionCallback(stream);
                used_fast_v = true;
            }
            if (!used_fast_v) {
                const size_t line_bytes = static_cast<size_t>(nranks) * plan.fast_bytes;
                void* line_fast = nullptr;
                bool owns_line = true;
                if (domain->pcie_recvbuf() && line_bytes <= domain->pcie_recvbuf_size()) {
                    line_fast = domain->pcie_recvbuf();
                    owns_line = false;
                } else {
                    line_fast = AllocDeviceBuffer(line_bytes);
                }
                if (!line_fast) {
                    AMPCCL_LOG(WARN, "[Rank %d] AllGather: line buffer alloc failed, fast path only",
                               domain->pcie_rank());
                    fast_timer->Start(stream);
                    fast_r = FastBackendImpl::AllGather(sendbuff, recvbuff, sendcount, datatype, comm,
                                                         stream);
                    fast_timer->Stop(stream);
                    DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(),
                                                                       pcie_r.IsOk());
                    DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
                    if (!fast_r.IsOk()) return fast_r;
                    return BackendResult::Success;
                }

                auto value_sync = domain->BeginValueSyncSubmission(stream);
                if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                    if (owns_line) {
                        FreeDeviceBuffer(line_fast);
                    }
                    return BackendResult::UnhandledError;
                }
                fast_timer->Start(stream);
                fast_r = FastBackendImpl::AllGather(sendbuff, line_fast, fast_elems, datatype, comm, stream);
                fast_timer->Stop(stream);

                pcie_timer->Start(pcie_stream);
                pcie_r = PCIeBackendImpl::AllGatherStrided(domain, sendbuff, recvbuff, sendcount,
                                                             fast_elems, datatype, pcie_stream);
                pcie_timer->Stop(pcie_stream);

                PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
                SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllGather");
                // Keep the fallback layout fix outside AMPCCL fast/PCIe timers, but inside
                // the user stream dependency chain so correctness checks observe merged data.
                EnqueueSegmentedMergeOnStream(stream, recvbuff, line_fast, nranks,
                                              plan.fast_bytes, plan.pcie_bytes, owns_line);
                DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                                   fast_r.IsOk(), pcie_r.IsOk());
                DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                EnqueueCollectiveCompletionCallback(stream);
            }
        } else if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream && plan.fast_bytes == 0) {
            auto value_sync = domain->BeginValueSyncSubmission(stream);
            if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                return BackendResult::UnhandledError;
            }
            pcie_timer->Start(pcie_stream);
            pcie_r = PCIeBackendImpl::AllGather(domain, sendbuff, recvbuff, sendcount, datatype,
                                                  pcie_stream);
            pcie_timer->Stop(pcie_stream);
            PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());  // pure PCIe, no fast
            SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllGather");
            // Explicitly synchronize PCIe stream with user stream
            // (void)ampccl::OriginalCollectiveAPI::Synchronize(pcie_stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        } else {
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::AllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        }
        if (!fast_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] AllGather fast path failed: code=%d original=%d", domain->pcie_rank(), static_cast<int>(fast_r.code), fast_r.original);
            return fast_r;
        }
        if (!pcie_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] AllGather pcie path failed: code=%d original=%d", domain->pcie_rank(), static_cast<int>(pcie_r.code), pcie_r.original);
            return pcie_r;
        }
        return BackendResult::Success;
    }

    static BackendResult ReduceScatter(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t recvcount,
        int datatype,
        int op,
        void* comm,
        void* stream) {
        size_t es = NativeDataTypeElemSize(datatype);
        if (es == 0) {
            return BackendResult::UnhandledError;
        }
        const int nranks = domain->pcie_nranks();
        OpKey op_key = MakeOpKey(domain, CollectiveType::ReduceScatter, recvcount * es, datatype);

        CollectiveLaunchPlan launch = PrepareCollectiveLaunch(domain, op_key);
        double alpha = launch.alpha;
        Plan plan = launch.plan;
        if (plan.use_pcie && !PCIeBackendImpl::SupportsReduceScatter(datatype, op)) {
            AMPCCL_LOG(WARN,
                       "[Rank %d] ReduceScatter PCIe disabled: public PCCL API supports only fp32/sum",
                       domain->pcie_rank());
            ForceFastOnly(&plan, op_key.bytes);
        }

        BackendResult fast_r;
        BackendResult pcie_r;
        void* pcie_stream = domain->pcie_stream();
        auto fast_timer = std::make_shared<Timer>();
        auto pcie_timer = std::make_shared<Timer>();

        AMPCCL_LOG(INFO, "[Rank %d] Before ReduceScatter, op=%d bytes=%zu bucket_bytes=%zu dtype=%d nranks=%d cache_hit=%d "
            "alpha=%.4f, plan.use_pcie=%d, plan.pcie_bytes=%lu, plan.fast_bytes=%lu, domain_hash=%016llx",
            domain->pcie_rank(), static_cast<int>(op_key.op), op_key.bytes, op_key.bucket_bytes,
            op_key.datatype, op_key.nranks, launch.cache_hit ? 1 : 0, alpha, plan.use_pcie,
            plan.pcie_bytes, plan.fast_bytes,
            (unsigned long long)domain->logical_key.comm_hash);

        if (!plan.use_pcie || plan.pcie_bytes == 0 || !pcie_stream) {
            if (!Config::IsFastOnlyBwLogEnabled()) {
                return FastBackendImpl::ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
            }
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                               fast_r.IsOk(), true);
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
            EnqueueCollectiveCompletionCallback(stream);
            if (!fast_r.IsOk()) {
                return fast_r;
            }
            return BackendResult::Success;
        }

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream && plan.fast_bytes > 0) {
            const size_t fast_elems = plan.fast_bytes / es;
            const size_t recvcount_elems = recvcount;
            bool used_fast_v = false;
            if (OriginalCollectiveAPI::HasFastReduceScatterV()) {
                // Fast ReduceScatterV expects sendCounts/sendDispls as HOST pointers.
                std::vector<size_t> send_counts_host(static_cast<size_t>(nranks));
                std::vector<size_t> send_displs_host(static_cast<size_t>(nranks));
                for (int j = 0; j < nranks; ++j) {
                    send_counts_host[static_cast<size_t>(j)] = fast_elems;
                    send_displs_host[static_cast<size_t>(j)] =
                        static_cast<size_t>(j) * recvcount_elems;
                }
                auto value_sync = domain->BeginValueSyncSubmission(stream);
                if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                    return BackendResult::UnhandledError;
                }
                fast_timer->Start(stream);
                fast_r = OriginalCollectiveAPI::FastReduceScatterV(
                               sendbuff, send_counts_host.data(), send_displs_host.data(), recvbuff,
                               fast_elems, datatype, op, comm, stream);
                fast_timer->Stop(stream);

                const size_t total_elems = recvcount * static_cast<size_t>(nranks);
                pcie_timer->Start(pcie_stream);
                pcie_r = PCIeBackendImpl::ReduceScatterStrided(
                         domain, sendbuff, recvbuff,
                         total_elems, fast_elems, datatype, op, pcie_stream);
                pcie_timer->Stop(pcie_stream);

                PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
                SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "ReduceScatter");
                DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(),
                                                                   pcie_r.IsOk());
                DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
                used_fast_v = true;
            }
            if (!used_fast_v) {
                const size_t pack_bytes = static_cast<size_t>(nranks) * plan.fast_bytes;
                void* tmp_pack = nullptr;
                bool owns_pack = true;
                if (domain->pcie_recvbuf() && pack_bytes <= domain->pcie_recvbuf_size()) {
                    tmp_pack = domain->pcie_recvbuf();
                    owns_pack = false;
                } else {
                    tmp_pack = AllocDeviceBuffer(pack_bytes);
                }
                if (!tmp_pack) {
                    fast_timer->Start(stream);
                    fast_r = FastBackendImpl::ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm,
                                                        stream);
                    fast_timer->Stop(stream);
                    DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(),
                                                                       pcie_r.IsOk());
                    DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
                    if (!fast_r.IsOk()) return fast_r;
                    return BackendResult::Success;
                }
                auto value_sync = domain->BeginValueSyncSubmission(stream);
                if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                    if (owns_pack) {
                        FreeDeviceBuffer(tmp_pack);
                    }
                    return BackendResult::UnhandledError;
                }
                for (int b = 0; b < nranks; ++b) {
                    DeviceMemcpyD2DAsync(
                        static_cast<char*>(tmp_pack) + static_cast<size_t>(b) * plan.fast_bytes,
                        static_cast<const char*>(sendbuff) +
                            (static_cast<size_t>(b) * recvcount * es),
                        plan.fast_bytes, stream);
                }

                fast_timer->Start(stream);
                fast_r = FastBackendImpl::ReduceScatter(tmp_pack, recvbuff, fast_elems, datatype, op, comm,
                                                          stream);
                fast_timer->Stop(stream);

                const size_t total_elems = recvcount * static_cast<size_t>(nranks);
                pcie_timer->Start(pcie_stream);
                pcie_r = PCIeBackendImpl::ReduceScatterStrided(
                         domain, sendbuff, recvbuff, total_elems,
                         fast_elems, datatype, op, pcie_stream);
                pcie_timer->Stop(pcie_stream);

                PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
                SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "ReduceScatter");
                DomainManager::GetInstance().RegisterStreamPending(
                    stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk(), recvbuff, tmp_pack, nranks,
                    domain->pcie_rank(), plan.fast_bytes, plan.pcie_bytes, owns_pack);
                DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
            }
        } else if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream && plan.fast_bytes == 0) {
            auto value_sync = domain->BeginValueSyncSubmission(stream);
            if (!PcieWaitForUserInputIfNeeded(domain, stream, value_sync.get())) {
                return BackendResult::UnhandledError;
            }
            pcie_timer->Start(pcie_stream);
            pcie_r = PCIeBackendImpl::ReduceScatter(domain, sendbuff, recvbuff, recvcount, datatype, op,
                                                      pcie_stream);
            pcie_timer->Stop(pcie_stream);
            PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
            SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "ReduceScatter");
            // Explicitly synchronize PCIe stream with user stream
            // (void)ampccl::OriginalCollectiveAPI::Synchronize(pcie_stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        } else {
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        }
        if (!fast_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] ReduceScatter fast path failed: code=%d original=%d", domain->pcie_rank(), static_cast<int>(fast_r.code), fast_r.original);
            return fast_r;
        }
        if (!pcie_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] ReduceScatter pcie path failed: code=%d original=%d", domain->pcie_rank(), static_cast<int>(pcie_r.code), pcie_r.original);
            return pcie_r;
        }
        return BackendResult::Success;
    }

    // AllToAll：混合拆分布局依赖各 vendor 缓冲区约定，当前实现为「纯 Fast」或「纯 PCIe」。
    static BackendResult AllToAll(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        void* comm,
        void* stream) {
        size_t es = NativeDataTypeElemSize(datatype);
        if (es == 0) {
            return BackendResult::UnhandledError;
        }
        const int nranks = domain->pcie_nranks();
        if (nranks <= 0) {
            return BackendResult::UnhandledError;
        }
        // Use per-peer bytes (consistent with AllGather/ReduceScatter using per-rank bytes).
        // plan.fast_bytes / plan.pcie_bytes are per-peer; downstream pccl APIs that expect
        // total counts multiply by nranks explicitly.
        size_t per_peer_bytes = count * es;
        OpKey op_key = MakeOpKey(domain, CollectiveType::AllToAll, per_peer_bytes, datatype);

        CollectiveLaunchPlan launch = PrepareCollectiveLaunch(domain, op_key);
        double alpha = launch.alpha;
        Plan plan = launch.plan;
        if (plan.use_pcie && !PCIeBackendImpl::SupportsAllToAll(datatype)) {
            AMPCCL_LOG(WARN,
                       "[Rank %d] AllToAll PCIe disabled: public PCCL API supports only 4-byte elements",
                       domain->pcie_rank());
            ForceFastOnly(&plan, op_key.bytes);
        }

        BackendResult fast_r;
        BackendResult pcie_r;
        void* pcie_stream = domain->pcie_stream();
        auto fast_timer = std::make_shared<Timer>();
        auto pcie_timer = std::make_shared<Timer>();

        AMPCCL_LOG(INFO, "[Rank %d] Before AllToAll, alpha=%.4f, plan.use_pcie=%d, plan.pcie_bytes=%lu, plan.fast_bytes=%lu",
            domain->pcie_rank(), alpha, plan.use_pcie, plan.pcie_bytes,  plan.fast_bytes);

        if (!plan.use_pcie || plan.pcie_bytes == 0 || !pcie_stream) {
            if (!Config::IsFastOnlyBwLogEnabled()) {
                return FastBackendImpl::AllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
            }
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::AllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan,
                                                               fast_r.IsOk(), true);
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
            EnqueueCollectiveCompletionCallback(stream);
            if (!fast_r.IsOk()) {
                return fast_r;
            }
            return BackendResult::Success;
        }

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream && plan.fast_bytes == 0) {
            auto value_sync = domain->BeginValueSyncSubmission(stream);
            pcie_timer->Start(pcie_stream);
            // pcclAllToAll expects total element count (internally divides by nranks)
            pcie_r = PCIeBackendImpl::AllToAll(domain, sendbuff, recvbuff,
                                                  count * static_cast<size_t>(nranks), datatype, pcie_stream);
            pcie_timer->Stop(pcie_stream);
            PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
            SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllToAll");
            // Explicitly synchronize PCIe stream with user stream
            // (void)ampccl::OriginalCollectiveAPI::Synchronize(pcie_stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        } else if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream && plan.fast_bytes > 0) {
            const size_t fast_elems = plan.fast_bytes / es;
            const size_t elems_per_peer = count;
            bool used_fast_v = false;
            if (OriginalCollectiveAPI::HasFastAllToAllV()) {
                // Fast AllToAllV expects counts/displs as HOST pointers.
                std::vector<size_t> s_cnt_host(static_cast<size_t>(nranks));
                std::vector<size_t> s_dpl_host(static_cast<size_t>(nranks));
                std::vector<size_t> r_cnt_host(static_cast<size_t>(nranks));
                std::vector<size_t> r_dpl_host(static_cast<size_t>(nranks));
                for (int j = 0; j < nranks; ++j) {
                    size_t uj = static_cast<size_t>(j);
                    s_cnt_host[uj] = fast_elems;
                    s_dpl_host[uj] = uj * elems_per_peer;
                    r_cnt_host[uj] = fast_elems;
                    r_dpl_host[uj] = uj * elems_per_peer;
                }
                fast_timer->Start(stream);
                fast_r = OriginalCollectiveAPI::FastAllToAllV(sendbuff, s_cnt_host.data(), s_dpl_host.data(),
                                                         recvbuff, r_cnt_host.data(), r_dpl_host.data(),
                                                         datatype, comm, stream);
                fast_timer->Stop(stream);
                const size_t total_elems = elems_per_peer * static_cast<size_t>(nranks);
                auto value_sync = domain->BeginValueSyncSubmission(stream);
                pcie_timer->Start(pcie_stream);
                pcie_r = PCIeBackendImpl::AllToAllStrided(
                               domain, sendbuff, recvbuff, total_elems, fast_elems, datatype, pcie_stream);
                pcie_timer->Stop(pcie_stream);

                PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
                SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllToAll");
                DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(),
                                                                   pcie_r.IsOk());
                DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
                used_fast_v = true;
            }
            if (!used_fast_v) {
                const size_t per_peer_bytes = count * es;
                const size_t pack_bytes = static_cast<size_t>(nranks) * plan.fast_bytes;
                void* tmp_pack = nullptr;
                bool owns_pack = true;
                if (domain->pcie_recvbuf() && pack_bytes <= domain->pcie_recvbuf_size()) {
                    tmp_pack = domain->pcie_recvbuf();
                    owns_pack = false;
                } else {
                    tmp_pack = AllocDeviceBuffer(pack_bytes);
                }
                if (!tmp_pack) {
                    AMPCCL_LOG(WARN,
                               "[Rank %d] AllToAll hybrid: pack buffer alloc (%zu) failed, fast only",
                               domain->pcie_rank(), pack_bytes);
                    fast_timer->Start(stream);
                    fast_r = FastBackendImpl::AllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
                    fast_timer->Stop(stream);
                    DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(),
                                                                       pcie_r.IsOk());
                    DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
                    if (!fast_r.IsOk()) return fast_r;
                    return BackendResult::Success;
                }

                // Pack: for each peer j, copy the first fast_elems from the user send buffer
                // (offset j * count * es) into tmp_pack (offset j * fast_bytes).
                for (int b = 0; b < nranks; ++b) {
                    DeviceMemcpyD2DAsync(
                        static_cast<char*>(tmp_pack) + static_cast<size_t>(b) * plan.fast_bytes,
                        static_cast<const char*>(sendbuff) + static_cast<size_t>(b) * per_peer_bytes,
                        plan.fast_bytes, stream);
                }

                // Fast AlltoAll on packed buffer (in-place: send and recv on tmp_pack).
                fast_timer->Start(stream);
                fast_r = FastBackendImpl::AllToAll(tmp_pack, tmp_pack, fast_elems,
                                                     datatype, comm, stream);
                fast_timer->Stop(stream);

                // PCIe handles the remaining pcie_elems per peer via strided AlltoAll.
                const size_t total_elems = count * static_cast<size_t>(nranks);
                auto value_sync = domain->BeginValueSyncSubmission(stream);
                pcie_timer->Start(pcie_stream);
                pcie_r = PCIeBackendImpl::AllToAllStrided(
                               domain, sendbuff, recvbuff, total_elems, fast_elems, datatype, pcie_stream);
                pcie_timer->Stop(pcie_stream);

                PcieCompleteOnUserStreamIfNeeded(domain, stream, plan, value_sync.get(), pcie_timer.get());
                SyncPCIeStreamBeforeCallbackIfEnabled(domain, pcie_stream, plan, "AllToAll");
                DomainManager::GetInstance().RegisterStreamPending(
                    stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk(),
                    recvbuff, tmp_pack, nranks,
                    domain->pcie_rank(), plan.fast_bytes, plan.pcie_bytes, owns_pack);
                DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
            }
        } else {
            fast_timer->Start(stream);
            fast_r = FastBackendImpl::AllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
            fast_timer->Stop(stream);
            DomainManager::GetInstance().RegisterStreamPending(stream, domain, op_key, plan, fast_r.IsOk(), pcie_r.IsOk());
            DomainManager::GetInstance().AttachTimersToLastStreamPending(stream, fast_timer, pcie_timer);
                    EnqueueCollectiveCompletionCallback(stream);
        }
        if (!fast_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] AllToAll fast path failed: code=%d original=%d", domain->pcie_rank(), static_cast<int>(fast_r.code), fast_r.original);
            return fast_r;
        }
        if (!pcie_r.IsOk()) {
            AMPCCL_LOG(WARN, "[Rank %d] AllToAll pcie path failed: code=%d original=%d", domain->pcie_rank(), static_cast<int>(pcie_r.code), pcie_r.original);
            return pcie_r;
        }
        return BackendResult::Success;
    }
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_
