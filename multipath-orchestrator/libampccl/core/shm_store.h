#ifndef AMPCCL_CORE_SHM_STORE_H_
#define AMPCCL_CORE_SHM_STORE_H_

#include "core/domain_key.h"
#include "common/op_key.h"
#include "cache/param_cache.h"
#include "telemetry/stats.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace ampccl {

// Shared-memory store for multi-rank: per-rank stat slots + single param table.
// Only rank 0 writes params; all ranks read params. Each rank writes its own stat at SyncStream;
// rank 0 aggregates at next collective entry and writes params.
class ShmParamStore {
public:
    ShmParamStore() = default;
    ~ShmParamStore();

    // Non-copyable
    ShmParamStore(const ShmParamStore&) = delete;
    ShmParamStore& operator=(const ShmParamStore&) = delete;

    // Create or attach to shared segment for this key. my_rank/nranks from domain (e.g. pcie_rank/pcie_nranks).
    // Returns true on success.
    bool Attach(const CommDomainKey& key, int my_rank, int nranks);

    // Write this rank's stat (called at SyncStream). Only this rank's slot is written.
    void WriteMyStat(int my_rank, const OpKey& op_key, const ExecStat& stat);

    // Rank 0 only: read all slots, aggregate (max fast_time, max pcie_time), fill out_global_stat and out_op_key.
    // Returns true if at least one slot was valid.
    bool ReadAllStatsAndAggregate(ExecStat* out_global_stat, OpKey* out_op_key) const;

    // Read param table from shm into cache (all ranks).
    void ReadParams(ParamCache* cache) const;

    // Rank 0 only: write cache to shm.
    void WriteParams(const ParamCache& cache);

    void WaitUntilParamConsumed();
    void SignalParamWritten();
    void BarrierSync();

    bool IsAttached() const { return base_ != nullptr; }
    int Nranks() const { return nranks_; }
    bool IsRank0() const { return my_rank_ == 0; }

private:
    static constexpr uint64_t kMagic = 0x414d5043434c5f53u;  // "AMPCCL_S"
    static constexpr int kMaxRanks = 128;
    static constexpr int kMaxParamEntries = 512;

#pragma pack(push, 1)
    struct StatSlot {
        int op;           // CollectiveType as int
        uint64_t bytes;   // size_t as uint64
        int datatype;
        double fast_time;
        double pcie_time;
        uint64_t fast_bytes;
        uint64_t pcie_bytes;
        uint8_t fast_success;
        uint8_t pcie_success;
        uint8_t valid;    // 1 = written
        uint8_t padding[5];
    };
    static_assert(sizeof(StatSlot) == 56, "StatSlot size");

    struct ParamEntry {
        int op;
        uint64_t bytes;
        int datatype;
        double alpha;
        uint8_t use_pcie;
        uint8_t pad[4];
        double fast_bw;
        double pcie_bw;
        double best_alpha;
        double best_bandwidth;
    };

    struct Header {
        uint64_t magic;
        int nranks;
        uint32_t param_version;
        uint32_t param_read_count;
        uint32_t barrier_count;
        uint32_t barrier_sense;
    };
#pragma pack(pop)

    static std::string ShmNameForKey(const CommDomainKey& key);
    size_t ShmSize() const;

    void* base_ = nullptr;
    size_t shm_size_ = 0;
    int my_rank_ = -1;
    int nranks_ = 0;
    int shm_fd_ = -1;
    std::string shm_name_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_SHM_STORE_H_
