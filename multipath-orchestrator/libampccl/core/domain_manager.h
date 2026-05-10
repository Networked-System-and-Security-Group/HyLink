#ifndef AMPCCL_CORE_DOMAIN_MANAGER_H_
#define AMPCCL_CORE_DOMAIN_MANAGER_H_

#include "domain.h"
#include "controller/algo_factory.h"
#include "common/log.h"
#include "common/op_key.h"
#include "planner.h"
#include <unordered_map>
#include <mutex>
#include <memory>
#include <optional>

namespace ampccl {

// Pending collective record: registered when a collective is launched, consumed at stream sync.
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
};

// Manages the global "dividing param" table keyed by *our* Comm (CommDomainKey),
// and the mapping from raw backend communicator to our Comm.
//
// - Table: our Comm (topology + ranks + size) -> CommDomain (controller + param cache).
//   Shared across ranks; reused when a communicator is destroyed and re-created
//   with the same topology/ranks/size.
// - Raw mapping: raw comm pointer -> our Comm key. Used only to find which
//   domain to use for a given collective call. Unregistering a raw comm does
//   not remove the domain.
class DomainManager {
public:
    static DomainManager& GetInstance() {
        static DomainManager instance;
        return instance;
    }

    // Get or create domain by *our* Comm key (used by CommInit after building key).
    CommDomain* GetOrCreateDomainByKey(const CommDomainKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return GetOrCreateDomainByKeyLocked(key);
    }

    // Register raw communicator to our Comm. Call after backend CommInit.
    void RegisterRawComm(void* raw_comm, const CommDomainKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        (void)GetOrCreateDomainByKeyLocked(key);
        raw_to_key_[raw_comm] = key;
    }

    // Get domain for a raw communicator (for collective hooks).
    CommDomain* GetDomainByRawComm(void* raw_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = raw_to_key_.find(raw_comm);
        if (it == raw_to_key_.end()) {
            return nullptr;
        }
        auto jt = key_to_domain_.find(it->second);
        if (jt == key_to_domain_.end()) {
            return nullptr;
        }
        return jt->second.get();
    }

    // Unregister raw communicator on CommDestroy. Does not remove the domain.
    void UnregisterRawComm(void* raw_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_to_key_.erase(raw_comm);
    }

    // Stream -> pending collective: register when launching a collective, take at SynchronizeStream.
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
        stream_to_pending_[stream] = std::move(pending);
    }

    // Take and remove pending for this stream. Returns nullopt if none.
    std::optional<PendingCollective> TakeStreamPending(void* stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stream_to_pending_.find(stream);
        if (it == stream_to_pending_.end()) {
            return std::nullopt;
        }
        PendingCollective p = std::move(it->second);
        stream_to_pending_.erase(it);
        return p;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_to_key_.clear();
        key_to_domain_.clear();
        stream_to_pending_.clear();
    }

private:
    DomainManager() = default;
    ~DomainManager() = default;
    DomainManager(const DomainManager&) = delete;
    DomainManager& operator=(const DomainManager&) = delete;

    CommDomain* GetOrCreateDomainByKeyLocked(const CommDomainKey& key) {
        auto it = key_to_domain_.find(key);
        if (it != key_to_domain_.end()) {
            AMPCCL_LOG(INFO, "Comm reused (existing): world_size=%d topology_hash=%llu",
                       key.world_size, static_cast<unsigned long long>(key.topology_hash));
            return it->second.get();
        }
        AMPCCL_LOG(INFO, "[Rank %d] Comm created (new): world_size=%d topology_hash=%llu",
                   key.ranks[0], key.world_size, static_cast<unsigned long long>(key.topology_hash));
        auto algo = AlgoFactory::Create(key.world_size);
        auto controller = std::make_unique<AdaptiveController>(std::move(algo));
        auto domain = std::make_unique<CommDomain>(key, std::move(controller));
        CommDomain* ptr = domain.get();
        key_to_domain_[key] = std::move(domain);
        return ptr;
    }

    mutable std::mutex mutex_;
    std::unordered_map<CommDomainKey, std::unique_ptr<CommDomain>> key_to_domain_;
    std::unordered_map<void*, CommDomainKey> raw_to_key_;
    std::unordered_map<void*, PendingCollective> stream_to_pending_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_DOMAIN_MANAGER_H_
