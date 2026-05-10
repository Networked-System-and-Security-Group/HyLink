#ifndef AMPCCL_CORE_DOMAIN_H_
#define AMPCCL_CORE_DOMAIN_H_

#include "core/domain_key.h"
#include "core/logical_comm_key.h"
#include "common/config.h"
#include "common/log.h"
#include "cache/param_cache.h"
#include "controller/controller.h"
#include "telemetry/timer.h"
#include "core/shm_store.h"
#include "core/value_sync.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace ampccl {

class CommDomain {
public:
    CommDomainKey key;
    LogicalCommKey logical_key;
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

    // The PCIe stream belongs to the domain, while completion waits belong to
    // individual user streams. This creates a context keyed by that pair.
    std::unique_ptr<ValueSyncSubmission> BeginValueSyncSubmission(void* user_stream);

    // Device index active when pcie_stream was created (aclrtGetDevice / cudaGetDevice).
    // Host callbacks may run on threads without the app's TLS device; compare in debug logs (H6).
    int32_t pcie_device_id() const { return pcie_device_id_; }
    void set_pcie_device_id(int32_t d) { pcie_device_id_ = d; }

    void* pcie_recvbuf() const { return pcie_recvbuf_; }
    void set_pcie_recvbuf(void* p) { pcie_recvbuf_ = p; }
    size_t pcie_recvbuf_size() const { return pcie_recvbuf_size_; }
    void set_pcie_recvbuf_size(size_t s) { pcie_recvbuf_size_ = s; }

    void SetPendingPCIeInit(int rank, int nranks, const void* bootstrap, size_t bootstrap_len) {
        std::lock_guard<std::mutex> lock(pcie_init_mutex_);
        pending_pcie_rank_ = rank;
        pending_pcie_nranks_ = nranks;
        pending_pcie_bootstrap_.clear();
        if (bootstrap && bootstrap_len > 0) {
            const auto* begin = static_cast<const uint8_t*>(bootstrap);
            pending_pcie_bootstrap_.assign(begin, begin + bootstrap_len);
        }
        pending_pcie_init_ = true;
        pending_pcie_init_in_progress_ = false;
    }

    bool BeginPendingPCIeInit(int* rank, int* nranks, std::vector<uint8_t>* bootstrap) {
        std::lock_guard<std::mutex> lock(pcie_init_mutex_);
        if (!pending_pcie_init_ || pending_pcie_init_in_progress_ || pcie_comm_ != nullptr) {
            return false;
        }
        pending_pcie_init_in_progress_ = true;
        if (rank) *rank = pending_pcie_rank_;
        if (nranks) *nranks = pending_pcie_nranks_;
        if (bootstrap) *bootstrap = pending_pcie_bootstrap_;
        return true;
    }

    void FinishPendingPCIeInit() {
        std::lock_guard<std::mutex> lock(pcie_init_mutex_);
        pending_pcie_init_ = false;
        pending_pcie_init_in_progress_ = false;
        pending_pcie_bootstrap_.clear();
    }

    // Timers bound to this domain (fast stream + PCIe stream); used for deferred sync at SynchronizeStream.
    Timer& timer_fast() { return timer_fast_; }
    const Timer& timer_fast() const { return timer_fast_; }
    Timer& timer_pcie() { return timer_pcie_; }
    const Timer& timer_pcie() const { return timer_pcie_; }

    // Shared-memory param store for multi-rank: only used when nranks > 1. Lazy-attach on first use.
    bool is_intra_node() const { return is_intra_node_; }
    bool allow_ampccl() const { return allow_ampccl_; }
    void set_node_scope(bool intra_node, bool allow_ampccl) {
        is_intra_node_ = intra_node;
        allow_ampccl_ = allow_ampccl;
    }

    ShmParamStore* shm_store() { return &shm_store_; }
    const ShmParamStore* shm_store() const { return &shm_store_; }
    void EnsureShmAttached() {
        if (!Config::IsShmParamSyncEnabled()) {
            return;
        }
        if (pcie_nranks_ > 1 && !shm_store_.IsAttached()) {
            shm_store_.Attach(logical_key, pcie_rank_, pcie_nranks_);
        }
    }

    // Device-side uint64_t arrays for Hccl*V counts/displs (HCCL hybrid path).
    void EnsureVMetaDevCapacity(size_t bytes);
    bool UploadVMetaHost64(const uint64_t* host, size_t num_uint64);
    void* v_meta_dev() const { return v_meta_dev_; }

    // Bandwidth accumulation for average reporting at communicator destroy.
    // Only active when Config::IsAvgBwLogEnabled() is true.
    void RecordBw(double algbw, double busbw, double bus_factor) {
        if (!Config::IsAvgBwLogEnabled()) return;
        int warmup = Config::GetAvgBwWarmupIters();
        bw_total_iters_++;
        if (bw_total_iters_ <= warmup) return;
        sum_algbw_ += algbw;
        sum_busbw_ += busbw;
        sum_bus_factor_ += bus_factor;
        bw_record_count_++;
    }

    void PrintAvgBw() const {
        if (!Config::IsAvgBwLogEnabled() || bw_record_count_ == 0) return;
        double avg_algbw = sum_algbw_ / static_cast<double>(bw_record_count_);
        double avg_busbw = sum_busbw_ / static_cast<double>(bw_record_count_);
        double avg_bus_factor = sum_bus_factor_ / static_cast<double>(bw_record_count_);
        int warmup = Config::GetAvgBwWarmupIters();
        AMPCCL_LOG(OFF, "=== Average Bandwidth (recorded %d iters, skipped %d warmup) ===",
                   bw_record_count_, warmup);
        AMPCCL_LOG(OFF, "avg_algbw(GB/s,decimal)=%.6f avg_busbw(GB/s,decimal)=%.6f avg_bus_factor=%.6f",
                   avg_algbw, avg_busbw, avg_bus_factor);
    }

    CommDomain(const CommDomainKey& k, const LogicalCommKey& logical,
               std::unique_ptr<AdaptiveController> ctrl)
        : key(k), logical_key(logical), controller(std::move(ctrl)),
          pcie_comm_(nullptr), pcie_rank_(-1), pcie_nranks_(0), pcie_stream_(nullptr),
          pcie_device_id_(-1), pcie_recvbuf_(nullptr), pcie_recvbuf_size_(0),
          is_intra_node_(false), allow_ampccl_(false) {}

    CommDomain(const CommDomainKey& k, std::unique_ptr<AdaptiveController> ctrl)
        : CommDomain(k, LogicalCommKey{k.world_size, k.topology_hash}, std::move(ctrl)) {}

    ~CommDomain();

private:
    void* pcie_comm_;   // pcclComm_t (opaque)
    int pcie_rank_;
    int pcie_nranks_;
    void* pcie_stream_;  // pcclStream_t (opaque), aclrt/cuda stream created in InitPCIeForDomain
    int32_t pcie_device_id_ = -1;
    void* pcie_recvbuf_ = nullptr;
    size_t pcie_recvbuf_size_ = 0;
    mutable std::mutex pcie_init_mutex_;
    bool pending_pcie_init_ = false;
    bool pending_pcie_init_in_progress_ = false;
    int pending_pcie_rank_ = -1;
    int pending_pcie_nranks_ = 0;
    std::vector<uint8_t> pending_pcie_bootstrap_;
    mutable std::mutex value_sync_contexts_mutex_;
    std::unordered_map<void*, std::shared_ptr<ValueSyncContext>> value_sync_contexts_;
    void* v_meta_dev_ = nullptr;
    size_t v_meta_cap_ = 0;
    Timer timer_fast_;
    Timer timer_pcie_;
    ShmParamStore shm_store_;
    bool is_intra_node_;
    bool allow_ampccl_;

    // Bandwidth accumulation for average reporting at destroy time (rank 0 only).
    int bw_total_iters_ = 0;
    int bw_record_count_ = 0;
    double sum_algbw_ = 0.0;
    double sum_busbw_ = 0.0;
    double sum_bus_factor_ = 0.0;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_H_
