#include "controller/controller.h"
#include "common/op_key.h"
#include "cache/param_cache.h"
#include "core/logical_comm_key.h"
#include "core/shm_store.h"
#include "telemetry/stats.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unistd.h>

namespace {

using ampccl::AdaptiveController;
using ampccl::AlgoFactory;
using ampccl::CollectiveType;
using ampccl::ExecStat;
using ampccl::OpKey;
using ampccl::ParamCache;

OpKey MakeKey(CollectiveType op, size_t bytes) {
    return OpKey{op, bytes, bytes, 1, 8, 0x1234};
}

ExecStat MakeStat(double fast_time, double pcie_time) {
    ExecStat stat;
    stat.fast_time = fast_time;
    stat.pcie_time = pcie_time;
    stat.fast_bytes = 199313408;
    stat.pcie_bytes = 2013184;
    stat.fast_success = true;
    stat.pcie_success = true;
    return stat;
}

std::unique_ptr<AdaptiveController> MakeController() {
    return std::make_unique<AdaptiveController>(8);
}

void TestNewOpKeyStartsFromIndependentDefault() {
    const OpKey rs = MakeKey(CollectiveType::ReduceScatter, 201326592);
    const OpKey ag = MakeKey(CollectiveType::AllGather, 201326592);
    assert(!(rs == ag));

    ParamCache cache;
    std::unique_ptr<AdaptiveController> controller = MakeController();

    const double rs_initial = controller->SuggestAlpha(rs, cache);
    assert(std::abs(rs_initial - 0.01) < 1e-12);

    controller->Update(rs, MakeStat(0.100, 0.010), cache);
    const double rs_updated = controller->SuggestAlpha(rs, cache);
    assert(std::abs(rs_updated - 0.01) > 1e-8);

    // AG is a different OpKey and must not inherit RS's updated alpha.
    const double ag_initial = controller->SuggestAlpha(ag, cache);
    assert(std::abs(ag_initial - 0.01) < 1e-12);
}

void TestInterleavedKeysDoNotContaminateHistory() {
    const OpKey rs = MakeKey(CollectiveType::ReduceScatter, 201326592);
    const OpKey ag = MakeKey(CollectiveType::AllGather, 201326592);
    const ExecStat rs_stat = MakeStat(0.100, 0.010);
    const ExecStat ag_stat = MakeStat(0.020, 0.020);

    ParamCache interleaved_cache;
    std::unique_ptr<AdaptiveController> interleaved = MakeController();
    interleaved->Update(rs, rs_stat, interleaved_cache);
    interleaved->SuggestAlpha(rs, interleaved_cache);
    interleaved->SuggestAlpha(ag, interleaved_cache);
    interleaved->Update(ag, ag_stat, interleaved_cache);
    interleaved->SuggestAlpha(ag, interleaved_cache);
    interleaved->Update(rs, rs_stat, interleaved_cache);
    const double interleaved_rs = interleaved->SuggestAlpha(rs, interleaved_cache);

    ParamCache rs_only_cache;
    std::unique_ptr<AdaptiveController> rs_only = MakeController();
    rs_only->Update(rs, rs_stat, rs_only_cache);
    rs_only->SuggestAlpha(rs, rs_only_cache);
    rs_only->Update(rs, rs_stat, rs_only_cache);
    const double rs_only_value = rs_only->SuggestAlpha(rs, rs_only_cache);

    assert(std::abs(interleaved_rs - rs_only_value) < 1e-12);
}

void AssertCachedAlpha(const ParamCache& cache, const OpKey& key, double expected) {
    assert(cache.Contains(key));
    assert(std::abs(cache.Lookup(key).alpha - expected) < 1e-12);
}

void TestIncrementalShmKeepsEarlierKeys() {
    const OpKey rs = MakeKey(CollectiveType::ReduceScatter, 201326592);
    const OpKey ag = MakeKey(CollectiveType::AllGather, 201326592);

    // The PID makes this test's POSIX SHM name independent from stale runs.
    const ampccl::LogicalCommKey logical_key{
        2, static_cast<uint64_t>(::getpid()) ^ 0xfeedbeefULL};
    ampccl::ShmParamStore rank0_store;
    ampccl::ShmParamStore rank1_store;
    assert(rank0_store.Attach(logical_key, 0, 2));
    assert(rank1_store.Attach(logical_key, 1, 2));

    ParamCache rank0_cache;
    ParamCache rank1_cache;
    rank0_cache.Update(rs, ampccl::ParamValue(0.11, true, 10.0, 2.0));
    rank0_store.WriteParams(rank0_cache, rs);
    rank0_store.SignalParamWritten();
    rank1_store.ReadParams(&rank1_cache);
    AssertCachedAlpha(rank1_cache, rs, 0.11);

    rank0_cache.Update(ag, ampccl::ParamValue(0.22, true, 11.0, 3.0));
    rank0_store.WriteParams(rank0_cache, ag);
    rank0_store.SignalParamWritten();
    rank1_store.ReadParams(&rank1_cache);
    AssertCachedAlpha(rank1_cache, rs, 0.11);
    AssertCachedAlpha(rank1_cache, ag, 0.22);

    rank0_cache.Update(rs, ampccl::ParamValue(0.13, true, 12.0, 2.5));
    rank0_store.WriteParams(rank0_cache, rs);
    rank0_store.SignalParamWritten();
    rank1_store.ReadParams(&rank1_cache);
    AssertCachedAlpha(rank1_cache, rs, 0.13);
    AssertCachedAlpha(rank1_cache, ag, 0.22);
}

}  // namespace

int main() {
    setenv("AMPCCL_ALGO", "split_ratio", 1);
    setenv("AMPCCL_ENABLE_PCIE", "1", 1);

    TestNewOpKeyStartsFromIndependentDefault();
    TestInterleavedKeysDoNotContaminateHistory();
    TestIncrementalShmKeepsEarlierKeys();

    std::cout << "per-OpKey adaptive controller tests: PASS\n";
    return 0;
}
