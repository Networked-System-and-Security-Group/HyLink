#ifndef AMPCCL_CORE_SHM_STORE_H_
#define AMPCCL_CORE_SHM_STORE_H_

#include "core/logical_comm_key.h"
#include "common/op_key.h"
#include "cache/param_cache.h"
#include "telemetry/stats.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace ampccl {

// Shared-memory store for multi-rank: per-rank stat slots + param entries.
// Only rank 0 writes params; all ranks read params. Each rank writes its own stat at SyncStream;
// rank 0 aggregates at next collective entry and writes params for the current OpKey only.
class ShmParamStore {
public:
    ShmParamStore() = default;
    ~ShmParamStore();

    ShmParamStore(const ShmParamStore&) = delete;
    ShmParamStore& operator=(const ShmParamStore&) = delete;

    bool Attach(const LogicalCommKey& key, int my_rank, int nranks);

    void WriteMyStat(int my_rank, const OpKey& op_key, const ExecStat& stat);

    // ReadAllStatsAndAggregate: rank 0 waits for all ranks to publish the current
    // stat epoch, aggregates max-times / success / bytes, clears the epoch slots,
    // then advances the epoch. Returns false when no rank has written this epoch yet.
    bool ReadAllStatsAndAggregate(ExecStat* out_global_stat, OpKey* out_op_key,
                                  double* out_algbw = nullptr,
                                  double* out_busbw = nullptr,
                                  double* out_bus_factor = nullptr) const;

    // Merge single entry from shm (rank 0 wrote one key); does not clear other cache keys.
    void ReadParams(ParamCache* cache) const;

    // Rank 0 only: write one ParamEntry for op_key (must exist in cache after Update).
    void WriteParams(const ParamCache& cache, const OpKey& op_key);

    void WaitUntilParamConsumed();
    void SignalParamWritten();
    void BarrierSync();

    bool IsAttached() const { return base_ != nullptr; }
    int Nranks() const { return nranks_; }
    bool IsRank0() const { return my_rank_ == 0; }

private:
    // Bumped when layout changes (invalidate old segments).
    static constexpr uint64_t kMagic = 0x414d5043434c5f56u;
    static constexpr int kMaxRanks = 128;
    static constexpr int kMaxParamEntries = 512;

#pragma pack(push, 1)
    struct StatSlot {
        uint64_t epoch;
        int op;
        uint64_t bytes;
        uint64_t bucket_bytes;
        int datatype;
        int nranks;
        uint64_t topology_hash;
        double fast_time;
        double pcie_time;
        uint64_t fast_bytes;
        uint64_t pcie_bytes;
        uint8_t fast_success;
        uint8_t pcie_success;
        uint8_t valid;
        uint8_t padding[5];
    };

    struct ParamEntry {
        int op;
        uint64_t bytes;
        uint64_t bucket_bytes;
        int datatype;
        int nranks;
        uint64_t topology_hash;
        double alpha;
        uint8_t use_pcie;
        uint8_t pad[7];
        double fast_bw;
        double pcie_bw;
        double best_alpha;
        double best_bandwidth;
    };

    struct Header {
        uint64_t magic;
        uint64_t stat_epoch;
        int nranks;
        uint32_t param_version;
        uint32_t param_read_count;
        uint32_t barrier_count;
        uint32_t barrier_sense;
    };
#pragma pack(pop)

    static std::string ShmNameForKey(const LogicalCommKey& key);
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
