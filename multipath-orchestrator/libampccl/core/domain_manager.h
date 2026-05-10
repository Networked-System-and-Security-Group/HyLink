#ifndef AMPCCL_CORE_DOMAIN_MANAGER_H_
#define AMPCCL_CORE_DOMAIN_MANAGER_H_

#include "domain.h"
#include "core/logical_comm_key.h"
#include "controller/algo_factory.h"
#include "common/log.h"
#include "common/op_key.h"
#include "planner.h"
#include "telemetry/timer.h"
#include <unordered_map>
#include <deque>
#include <mutex>
#include <memory>
#include <optional>

namespace ampccl {

// Pending collective record: registered when a collective is launched, consumed in host
// callback (FIFO per stream — matches GPU order of LaunchHostFunc callbacks).
struct PendingCollective {
    CommDomain* domain = nullptr;
    OpKey op_key;
    Plan plan;
    bool fast_success = true;
    bool pcie_success = true;
    void* user_recvbuf = nullptr;
    void* tmp_recvbuf = nullptr;
    bool owns_tmp_recvbuf = true;
    int nranks = 0;
    int rank = 0;
    size_t fast_bytes = 0;
    size_t pcie_bytes = 0;
    std::shared_ptr<Timer> fast_timer;
    std::shared_ptr<Timer> pcie_timer;
};

// Manages per-raw-communicator runtime state.
//
// CommDomain contains rank-private resources such as PCCL comm, PCIe stream,
// timers, and temporary buffers, so it must not be shared by different raw
// communicators. Shared cross-rank state uses CommDomain::logical_key instead.
class DomainManager {
public:
    static DomainManager& GetInstance() {
        static DomainManager instance;
        return instance;
    }

    // Register raw communicator to a fresh local CommDomain. Call after backend CommInit.
    void RegisterRawComm(void* raw_comm, const CommDomainKey& key) {
        LogicalCommKey logical{key.world_size, key.topology_hash};
        RegisterRawComm(raw_comm, key, logical);
    }

    void RegisterRawComm(void* raw_comm, const CommDomainKey& key,
                         const LogicalCommKey& logical_key) {
        if (raw_comm == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        int log_rank = key.ranks.empty() ? -1 : key.ranks[0];
        AMPCCL_LOG(INFO,
                   "[Rank %d] Comm domain created: raw=%p world_size=%d domain_hash=%llu logical_hash=%llu",
                   log_rank, raw_comm, key.world_size,
                   static_cast<unsigned long long>(key.topology_hash),
                   static_cast<unsigned long long>(logical_key.comm_hash));
        auto controller = std::make_unique<AdaptiveController>(key.world_size);
        raw_to_domain_[raw_comm] = std::make_unique<CommDomain>(key, logical_key, std::move(controller));
    }

    // Get domain for a raw communicator (for collective hooks).
    CommDomain* GetDomainByRawComm(void* raw_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = raw_to_domain_.find(raw_comm);
        if (it == raw_to_domain_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    // Unregister raw communicator on CommDestroy. The hook must destroy PCCL
    // resources before calling this; this removes the local runtime state.
    void UnregisterRawComm(void* raw_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_to_domain_.erase(raw_comm);
    }

    // Stream -> FIFO queue of pending collectives (multiple ops may be queued without CPU sync).
    void RegisterStreamPending(void* stream, CommDomain* domain, const OpKey& op_key,
                               const Plan& plan, bool fast_success, bool pcie_success,
                               void* user_recvbuf = nullptr, void* tmp_recvbuf = nullptr,
                               int nranks = 0, int rank = 0, size_t fast_bytes = 0,
                               size_t pcie_bytes = 0, bool owns_tmp_recvbuf = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        PendingCollective pending;
        pending.domain = domain;
        pending.op_key = op_key;
        pending.plan = plan;
        pending.fast_success = fast_success;
        pending.pcie_success = pcie_success;
        pending.user_recvbuf = user_recvbuf;
        pending.tmp_recvbuf = tmp_recvbuf;
        pending.owns_tmp_recvbuf = owns_tmp_recvbuf;
        pending.nranks = nranks;
        pending.rank = rank;
        pending.fast_bytes = fast_bytes;
        pending.pcie_bytes = pcie_bytes;
        stream_to_pending_[stream].push_back(std::move(pending));
    }

    void AttachTimersToLastStreamPending(void* stream, std::shared_ptr<Timer> fast_timer,
                                         std::shared_ptr<Timer> pcie_timer) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stream_to_pending_.find(stream);
        if (it == stream_to_pending_.end() || it->second.empty()) {
            return;
        }
        PendingCollective& pending = it->second.back();
        pending.fast_timer = std::move(fast_timer);
        pending.pcie_timer = std::move(pcie_timer);
    }

    // Pop oldest pending for this stream (matches first LaunchHostFunc to complete).
    std::optional<PendingCollective> TakeStreamPending(void* stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stream_to_pending_.find(stream);
        if (it == stream_to_pending_.end() || it->second.empty()) {
            return std::nullopt;
        }
        PendingCollective p = std::move(it->second.front());
        it->second.pop_front();
        if (it->second.empty()) {
            stream_to_pending_.erase(it);
        }
        return p;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_to_domain_.clear();
        stream_to_pending_.clear();
    }

private:
    DomainManager() = default;
    ~DomainManager() = default;
    DomainManager(const DomainManager&) = delete;
    DomainManager& operator=(const DomainManager&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<void*, std::unique_ptr<CommDomain>> raw_to_domain_;
    std::unordered_map<void*, std::deque<PendingCollective>> stream_to_pending_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_MANAGER_H_
