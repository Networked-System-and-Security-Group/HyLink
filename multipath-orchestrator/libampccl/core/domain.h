#ifndef AMPCCL_CORE_DOMAIN_H_
#define AMPCCL_CORE_DOMAIN_H_

#include "core/domain_key.h"
#include "cache/param_cache.h"
#include "controller/controller.h"
#include "telemetry/timer.h"
#include "core/shm_store.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

namespace ampccl {

class CommDomain {
public:
    CommDomainKey key;
    std::unique_ptr<AdaptiveController> controller;
    ParamCache param_cache;

    // PCIe (PCCL) communicator state - set in CommInit via InitPCIeForDomain
    void* pcie_comm() const { return pcie_comm_; }
    void set_pcie_comm(void* c) { pcie_comm_ = c; }
    int pcie_rank() const { return pcie_rank_; }
    void set_pcie_rank(int r) { pcie_rank_ = r; }
    int pcie_nranks() const { return pcie_nranks_; }
    void set_pcie_nranks(int n) { pcie_nranks_ = n; }
    void* pcie_stream() const { return pcie_stream_; }
    void set_pcie_stream(void* s) { pcie_stream_ = s; }

    void* pcie_recvbuf() const { return pcie_recvbuf_; }
    void set_pcie_recvbuf(void* p) { pcie_recvbuf_ = p; }
    size_t pcie_recvbuf_size() const { return pcie_recvbuf_size_; }
    void set_pcie_recvbuf_size(size_t s) { pcie_recvbuf_size_ = s; }

    // Timers bound to this domain (fast stream + PCIe stream); used for deferred sync at SynchronizeStream.
    Timer& timer_fast() { return timer_fast_; }
    const Timer& timer_fast() const { return timer_fast_; }
    Timer& timer_pcie() { return timer_pcie_; }
    const Timer& timer_pcie() const { return timer_pcie_; }

    // Shared-memory param store for multi-rank: only used when nranks > 1. Lazy-attach on first use.
    ShmParamStore* shm_store() { return &shm_store_; }
    const ShmParamStore* shm_store() const { return &shm_store_; }
    void EnsureShmAttached() {
        if (pcie_nranks_ > 1 && !shm_store_.IsAttached()) {
            shm_store_.Attach(key, pcie_rank_, pcie_nranks_);
        }
    }

    CommDomain(const CommDomainKey& k, std::unique_ptr<AdaptiveController> ctrl)
        : key(k), controller(std::move(ctrl)),
          pcie_comm_(nullptr), pcie_rank_(-1), pcie_nranks_(0), pcie_stream_(nullptr),
          pcie_recvbuf_(nullptr), pcie_recvbuf_size_(0) {}

private:
    void* pcie_comm_;   // pcclComm_t (opaque)
    int pcie_rank_;
    int pcie_nranks_;
    void* pcie_stream_;  // pcclStream_t (opaque), created in InitPCIeForDomain
    void* pcie_recvbuf_ = nullptr;
    size_t pcie_recvbuf_size_ = 0;
    Timer timer_fast_;
    Timer timer_pcie_;
    ShmParamStore shm_store_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_H_
