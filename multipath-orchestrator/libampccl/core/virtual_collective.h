#ifndef AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_
#define AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_

#include "domain.h"
#include "domain_manager.h"
#include "planner.h"
#include "common/op_key.h"
#include "backend/fast_backend.h"
#include "backend/pcie_backend.h"
#include "backend/memory/device_mem.h"
#include "telemetry/stats.h"
#include "common/config.h"
#include "common/log.h"
#include <cstddef>
#include <cstring>

namespace ampccl {

// Virtual collective layer - the heart of the system
class VirtualCollective {
public:
    // AllReduce operation
    static BackendResult AllReduce(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t count,
        int datatype,
        int op,
        void* comm,
        void* stream
    ) {
        // 1. Build OpKey
        OpKey op_key;
        op_key.op = CollectiveType::AllReduce;
        op_key.bytes = count * GetDataTypeSize(datatype);
        op_key.datatype = datatype;

        domain->EnsureShmAttached();
        ShmParamStore* shm = domain->shm_store();
        if (shm->IsAttached() && shm->IsRank0()) {
            shm->WaitUntilParamConsumed();
            ExecStat global_stat;
            OpKey agg_op_key;
            if (shm->ReadAllStatsAndAggregate(&global_stat, &agg_op_key) && domain->controller) {
                domain->controller->Update(agg_op_key, global_stat, domain->param_cache);
                shm->WriteParams(domain->param_cache);
            }
            shm->SignalParamWritten();
        }
        if (shm->IsAttached()) {
            shm->ReadParams(&domain->param_cache);
        }

        // 2. ParamCache lookup
        ParamValue param = domain->param_cache.Lookup(op_key);

        // 3. Controller suggests alpha
        double alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);

        // 4. Planner builds split plan
        Plan plan = Planner::CreatePlan(op_key.bytes, alpha, param.use_pcie);

        if (shm->IsAttached()) {
            shm->BarrierSync();
        }

        // 5. Launch fast + PCIe backend: record events only, no sync; pending consumed at SynchronizeStream.
        AMPCCL_LOG(OFF, "[Rank %d] AllReduce before: op=AllReduce bytes=%zu datatype=%d alpha=%.5f use_pcie=%d fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), op_key.bytes, datatype, alpha, plan.use_pcie ? 1 : 0, plan.fast_bytes, plan.pcie_bytes);

        bool fast_ok = true;
        bool pcie_ok = true;
        void* pcie_stream = domain->pcie_stream();

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream) {
            size_t pcie_offset = plan.fast_bytes;
            size_t elem_size = GetDataTypeSize(datatype);
            void* tmp_recv = nullptr;
            bool owns_tmp = true;
            void* user_recvbuf = recvbuff;
            const bool use_tmp = (plan.fast_bytes > 0 && plan.pcie_bytes > 0);
            const size_t required_size = plan.fast_bytes + plan.pcie_bytes;
            if (use_tmp) {
                if (domain->pcie_recvbuf() && required_size <= domain->pcie_recvbuf_size()) {
                    tmp_recv = domain->pcie_recvbuf();
                    owns_tmp = false;
                } else {
                    tmp_recv = AllocDeviceBuffer(required_size);
                    if (!tmp_recv) {
                        AMPCCL_LOG(WARN, "[Rank %d] AllReduce tmp alloc failed, fallback to direct recvbuff",
                                   domain->pcie_rank());
                    }
                }
            }
            if (tmp_recv) {
                if (plan.fast_bytes > 0) {
                    domain->timer_fast().Start(stream);
                    BackendResult fast_result = FastBackendImpl::AllReduce(
                        const_cast<void*>(sendbuff), tmp_recv, plan.fast_bytes / elem_size,
                        datatype, op, comm, stream);
                    domain->timer_fast().Stop(stream);
                    fast_ok = (fast_result == BackendResult::Success);
                }
                if (plan.pcie_bytes > 0) {
                    domain->timer_pcie().Start(pcie_stream);
                    const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                    char* pcie_recv = static_cast<char*>(tmp_recv) + plan.fast_bytes;
                    BackendResult pcie_result = PCIeBackendImpl::AllReduce(
                        domain, pcie_send, pcie_recv, plan.pcie_bytes / elem_size,
                        datatype, op, pcie_stream);
                    domain->timer_pcie().Stop(pcie_stream);
                    pcie_ok = (pcie_result == BackendResult::Success);
                }
                DomainManager::GetInstance().RegisterStreamPending(
                    stream, domain, op_key, plan, fast_ok, pcie_ok,
                    user_recvbuf, tmp_recv, domain->pcie_nranks(), domain->pcie_rank(),
                    plan.fast_bytes, plan.pcie_bytes, owns_tmp);
            } else {
                if (plan.fast_bytes > 0) {
                    domain->timer_fast().Start(stream);
                    BackendResult fast_result = FastBackendImpl::AllReduce(
                        const_cast<void*>(sendbuff), recvbuff, plan.fast_bytes / elem_size,
                        datatype, op, comm, stream);
                    domain->timer_fast().Stop(stream);
                    fast_ok = (fast_result == BackendResult::Success);
                }
                if (plan.pcie_bytes > 0) {
                    domain->timer_pcie().Start(pcie_stream);
                    const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                    char* pcie_recv = static_cast<char*>(recvbuff) + pcie_offset;
                    BackendResult pcie_result = PCIeBackendImpl::AllReduce(
                        domain, pcie_send, pcie_recv, plan.pcie_bytes / elem_size,
                        datatype, op, pcie_stream);
                    domain->timer_pcie().Stop(pcie_stream);
                    pcie_ok = (pcie_result == BackendResult::Success);
                }
                // if (plan.pcie_bytes > 0) domain->timer_pcie().WaitEventOnStream(stream);
                DomainManager::GetInstance().RegisterStreamPending(
                    stream, domain, op_key, plan, fast_ok, pcie_ok);
            }
        } else {
            domain->timer_fast().Start(stream);
            BackendResult result = FastBackendImpl::AllReduce(
                sendbuff, recvbuff, count, datatype, op, comm, stream);
            domain->timer_fast().Stop(stream);
            fast_ok = (result == BackendResult::Success);
            DomainManager::GetInstance().RegisterStreamPending(
                stream, domain, op_key, plan, fast_ok, pcie_ok);
        }

        return (fast_ok && pcie_ok) ? BackendResult::Success : BackendResult::UnhandledError;
    }

    // Similar implementations for other collectives...
    static BackendResult AllGather(
        CommDomain* domain,
        const void* sendbuff,
        void* recvbuff,
        size_t sendcount,
        int datatype,
        void* comm,
        void* stream
    ) {
        // Similar to AllReduce but for AllGather
        AMPCCL_LOG(INFO,"[Rank %d] Begin virtual allgather!", domain->pcie_rank());
        OpKey op_key;
        op_key.op = CollectiveType::AllGather;
        op_key.bytes = sendcount * GetDataTypeSize(datatype);
        op_key.datatype = datatype;

        domain->EnsureShmAttached();
        ShmParamStore* shm = domain->shm_store();
        if (shm->IsAttached() && shm->IsRank0()) {
            AMPCCL_LOG(INFO, "[Rank %d] shm is attached and rank 0", domain->pcie_rank());
            shm->WaitUntilParamConsumed();
            ExecStat global_stat;
            OpKey agg_op_key;
            if (shm->ReadAllStatsAndAggregate(&global_stat, &agg_op_key) && domain->controller) {
                domain->controller->Update(agg_op_key, global_stat, domain->param_cache);
                shm->WriteParams(domain->param_cache);
            }
            shm->SignalParamWritten();
        }
        if (shm->IsAttached()) {
            shm->ReadParams(&domain->param_cache);
            AMPCCL_LOG(INFO, "[Rank %d] param_cache is read", domain->pcie_rank());
        }
        // AMPCCL_LOG(INFO, "[Rank %d] param_cache is %p", domain->pcie_rank(), domain->param_cache);

        ParamValue param = domain->param_cache.Lookup(op_key);
        AMPCCL_LOG(INFO, "[Rank %d] param is %p", domain->pcie_rank(), param);
        double alpha = domain->controller->SuggestAlpha(op_key, domain->param_cache);
        AMPCCL_LOG(INFO,"[Rank %d] before create plan, bytes=%zu, alpha=%.3f, use_pcie=%d",domain->pcie_rank(), op_key.bytes, alpha, param.use_pcie);
        Plan plan = Planner::CreatePlan(op_key.bytes, alpha, param.use_pcie);

        if (shm->IsAttached()) {
            shm->BarrierSync();
        }
        AMPCCL_LOG(OFF, "[Rank %d] AllGather before: send_bytes=%zu datatype=%d alpha=%.5f use_pcie=%d fast_bytes=%zu pcie_bytes=%zu",
                   domain->pcie_rank(), op_key.bytes, datatype, alpha, plan.use_pcie ? 1 : 0, plan.fast_bytes, plan.pcie_bytes);

        bool fast_ok = true;
        bool pcie_ok = true;
        void* pcie_stream = domain->pcie_stream();

        if (plan.use_pcie && plan.pcie_bytes > 0 && pcie_stream) {
            size_t pcie_offset = plan.fast_bytes;
            size_t elem_size = GetDataTypeSize(datatype);
            const int nranks = domain->pcie_nranks();
            const size_t tmp_size = static_cast<size_t>(nranks) * (plan.fast_bytes + plan.pcie_bytes);
            void* tmp_recv = nullptr;
            bool owns_tmp = true;
            void* user_recvbuf = recvbuff;
            const bool use_tmp = (plan.fast_bytes > 0 && plan.pcie_bytes > 0);
            if (use_tmp) {
                if (domain->pcie_recvbuf() && tmp_size <= domain->pcie_recvbuf_size()) {
                    tmp_recv = domain->pcie_recvbuf();
                    owns_tmp = false;
                } else {
                    tmp_recv = AllocDeviceBuffer(tmp_size);
                    if (!tmp_recv) {
                        AMPCCL_LOG(WARN, "[Rank %d] AllGather tmp alloc failed, fallback to direct recvbuff",
                                   domain->pcie_rank());
                    }
                }
            }
            if (tmp_recv) {
                const size_t tmp_fast_total = static_cast<size_t>(nranks) * plan.fast_bytes;

               
                if (plan.fast_bytes > 0) {
                    domain->timer_fast().Start(stream);
                    BackendResult fast_result = FastBackendImpl::AllGather(
                        sendbuff, tmp_recv, plan.fast_bytes / elem_size, datatype, comm, stream);
                    domain->timer_fast().Stop(stream);
                    fast_ok = (fast_result == BackendResult::Success);
                }
                if (plan.pcie_bytes > 0) {
                    domain->timer_pcie().Start(pcie_stream);
                    const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                    char* pcie_recv = static_cast<char*>(tmp_recv) + tmp_fast_total;
                    size_t pcie_chunk_elems = plan.pcie_bytes / elem_size;
                    BackendResult pcie_result = PCIeBackendImpl::AllGather(
                        domain, pcie_send, pcie_recv, pcie_chunk_elems, datatype, pcie_stream);
                    domain->timer_pcie().Stop(pcie_stream);
                    pcie_ok = (pcie_result == BackendResult::Success);
                }
                DomainManager::GetInstance().RegisterStreamPending(
                    stream, domain, op_key, plan, fast_ok, pcie_ok,
                    user_recvbuf, tmp_recv, nranks, domain->pcie_rank(),
                    plan.fast_bytes, plan.pcie_bytes, owns_tmp);
            } else {
                if (plan.fast_bytes > 0) {
                    domain->timer_fast().Start(stream);
                    BackendResult fast_result = FastBackendImpl::AllGather(
                        sendbuff, recvbuff, plan.fast_bytes / elem_size, datatype, comm, stream);
                    domain->timer_fast().Stop(stream);
                    fast_ok = (fast_result == BackendResult::Success);
                }
                if (plan.pcie_bytes > 0) {
                    domain->timer_pcie().Start(pcie_stream);
                    const char* pcie_send = static_cast<const char*>(sendbuff) + pcie_offset;
                    char* pcie_recv = static_cast<char*>(recvbuff) + pcie_offset;
                    size_t pcie_chunk_elems = plan.pcie_bytes / elem_size;
                    BackendResult pcie_result = PCIeBackendImpl::AllGather(
                        domain, pcie_send, pcie_recv, pcie_chunk_elems, datatype, pcie_stream);
                    domain->timer_pcie().Stop(pcie_stream);
                    pcie_ok = (pcie_result == BackendResult::Success);
                }
                DomainManager::GetInstance().RegisterStreamPending(
                    stream, domain, op_key, plan, fast_ok, pcie_ok);
            }
        } else {
            domain->timer_fast().Start(stream);
            BackendResult result = FastBackendImpl::AllGather(
                sendbuff, recvbuff, sendcount, datatype, comm, stream);
            domain->timer_fast().Stop(stream);
            fast_ok = (result == BackendResult::Success);
            DomainManager::GetInstance().RegisterStreamPending(
                stream, domain, op_key, plan, fast_ok, pcie_ok);
        }

        return fast_ok ? BackendResult::Success : BackendResult::UnhandledError;
    }

private:
    static size_t GetDataTypeSize(int datatype) {
        // Map NCCL/HCCL datatype to size
        // This is a simplified version - actual implementation should handle all types
        // HCCL_DATA_TYPE_INT8 = 0,     /* int8 */
        // HCCL_DATA_TYPE_INT16 = 1,    /* int16 */
        // HCCL_DATA_TYPE_INT32 = 2,    /* int32 */
        // HCCL_DATA_TYPE_FP16 = 3,     /* float16 */
        // HCCL_DATA_TYPE_FP32 = 4,     /* float32 */
        // HCCL_DATA_TYPE_INT64 = 5,    /* int64 */
        // HCCL_DATA_TYPE_UINT64 = 6,   /* uint64 */
        // HCCL_DATA_TYPE_UINT8 = 7,    /* uint8 */
        // HCCL_DATA_TYPE_UINT16 = 8,   /* uint16 */
        // HCCL_DATA_TYPE_UINT32 = 9,   /* uint32 */
        // HCCL_DATA_TYPE_FP64 = 10,    /* fp64 */
        // HCCL_DATA_TYPE_BFP16 = 11,   /* bfp16 */
        // HCCL_DATA_TYPE_INT128 = 12,  /* int128 */
        // HCCL_DATA_TYPE_BOOL = 13     /* bool */
        // no fit for nccl (the ncclFloat32 is 7, so set case 7 to 4)
        switch (datatype) {
            case 0: return 1;
            case 1: return 2;
            case 2: return 4;
            case 3: return 2;
            case 4: return 4;
            case 5: return 8;
            case 6: return 8;
            case 7: return 4;
            case 8: return 2;
            case 9: return 4;
            case 10: return 8;
            case 11: return 2;
            case 12: return 16;
            case 13: return 1;
            default: return 0;
        }
    }
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_VIRTUAL_COLLECTIVE_H_