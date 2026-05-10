#include "shm_store.h"
#include "common/log.h"
#include <cstring>
#include <sstream>
#include <functional>
#include <algorithm>
#include <thread>
#include <climits>
#include <cerrno>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>
#include <sched.h>
#endif

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <errno.h>
#endif
#include <cinttypes>
#include <mutex>

namespace ampccl {

namespace {

#if defined(__linux__) || defined(__APPLE__)
constexpr bool kShmAvailable = true;
std::mutex g_shm_attach_mutex;

// A process-local mutex cannot serialize initialization by different ranks.
// Keep the file lock through sizing, mapping and header initialization/reset.
class ShmAttachFileLock {
public:
    explicit ShmAttachFileLock(int& fd) : fd_(fd) {
        int result;
        do {
            result = flock(fd_, LOCK_EX);
        } while (result != 0 && errno == EINTR);
        locked_ = (result == 0);
    }
    ~ShmAttachFileLock() {
        // Closing the descriptor on an error path already releases its lock.
        if (locked_ && fd_ >= 0) flock(fd_, LOCK_UN);
    }
    bool locked() const { return locked_; }
    ShmAttachFileLock(const ShmAttachFileLock&) = delete;
    ShmAttachFileLock& operator=(const ShmAttachFileLock&) = delete;

private:
    int& fd_;
    bool locked_ = false;
};
#else
constexpr bool kShmAvailable = false;
#endif

#if defined(__linux__)
// Futex helpers for Linux (shared across processes in /dev/shm)
static int futex_wait(uint32_t* addr, uint32_t val, const struct timespec* timeout = nullptr) {
    return static_cast<int>(syscall(SYS_futex, addr, FUTEX_WAIT, val, timeout, nullptr, 0));
}

static int futex_wake(uint32_t* addr, int n) {
    return static_cast<int>(syscall(SYS_futex, addr, FUTEX_WAKE, n, nullptr, nullptr, 0));
}

// Adaptive wait: short spin then futex sleep
template <typename Predicate>
static void AdaptiveWait(uint32_t* futex_addr, Predicate pred) {
    constexpr int kSpinIterations = 100;
    for (int spin = 0; spin < kSpinIterations; ++spin) {
        if (pred()) return;
        std::this_thread::yield();
    }
    // After spinning, use futex with timeout to handle spurious wakeups
    struct timespec timeout = {0, 1000000};  // 1ms timeout
    while (!pred()) {
        uint32_t current = __atomic_load_n(futex_addr, __ATOMIC_SEQ_CST);
        if (pred()) return;
        futex_wait(futex_addr, current, &timeout);
    }
}
#endif

static double AlgBytesFactor(CollectiveType op, int n) {
    if (n <= 1) {
        return 1.0;
    }
    switch (op) {
        case CollectiveType::AllGather:
        case CollectiveType::ReduceScatter:
        case CollectiveType::AllToAll:
            return static_cast<double>(n);
        case CollectiveType::AllReduce:
        default:
            return 1.0;
    }
}

static double BusBwFactor(CollectiveType op, int n) {
    if (n <= 1) {
        return 0.0;
    }
    const double f = static_cast<double>(n - 1) / static_cast<double>(n);
    if (op == CollectiveType::AllReduce) {
        return 2.0 * f;
    }
    return f;
}

static const char* CollectiveTypeLabel(CollectiveType op) {
    switch (op) {
        case CollectiveType::AllReduce:
            return "AllReduce";
        case CollectiveType::AllGather:
            return "AllGather";
        case CollectiveType::ReduceScatter:
            return "ReduceScatter";
        case CollectiveType::AllToAll:
            return "AllToAll";
        default:
            return "Unknown";
    }
}

}  // namespace

std::string ShmParamStore::ShmNameForKey(const LogicalCommKey& key) {
    size_t h = std::hash<LogicalCommKey>{}(key);
    std::ostringstream os;
    os << "/ampccl_" << std::hex << h;
    return os.str();
}

size_t ShmParamStore::ShmSize() const {
    size_t header = sizeof(Header);
    size_t stat_slots = static_cast<size_t>(kMaxRanks) * sizeof(StatSlot);
    size_t param_header = sizeof(uint64_t) + sizeof(uint32_t);  // version + num_entries
    size_t param_entries = static_cast<size_t>(kMaxParamEntries) * sizeof(ParamEntry);
    return header + stat_slots + param_header + param_entries;
}

bool ShmParamStore::Attach(const LogicalCommKey& key, int my_rank, int nranks) {
    if (!kShmAvailable || nranks <= 0 || my_rank < 0 || my_rank >= nranks) {
        return false;
    }
    if (nranks > kMaxRanks) {
        AMPCCL_LOG(WARN, "ShmStore: nranks %d > kMaxRanks %d, shm disabled", nranks, kMaxRanks);
        return false;
    }
    std::string name = ShmNameForKey(key);
    shm_size_ = ShmSize();
    my_rank_ = my_rank;
    nranks_ = nranks;

#if defined(__linux__) || defined(__APPLE__)
    std::lock_guard<std::mutex> attach_lock(g_shm_attach_mutex);
    mode_t mode = 0666;
    shm_fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, mode);
    if (shm_fd_ < 0) {
        AMPCCL_LOG(WARN, "ShmStore: shm_open %s failed", name.c_str());
        return false;
    }
    ShmAttachFileLock file_lock(shm_fd_);
    if (!file_lock.locked()) {
        AMPCCL_LOG(WARN, "ShmStore: flock %s failed: errno=%d", name.c_str(), errno);
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    struct stat st;
    if (fstat(shm_fd_, &st) != 0) {
        if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0) {
            AMPCCL_LOG(WARN, "ShmStore: ftruncate failed");
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    } else if (st.st_size == 0) {
        if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0) {
            AMPCCL_LOG(WARN, "ShmStore: ftruncate failed");
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    } else if (static_cast<size_t>(st.st_size) != shm_size_) {
        if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0) {
            AMPCCL_LOG(WARN, "ShmStore: resize after size mismatch failed");
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    }
    base_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        close(shm_fd_);
        shm_fd_ = -1;
        AMPCCL_LOG(WARN, "ShmStore: mmap failed");
        return false;
    }
    Header* hdr = static_cast<Header*>(base_);
    if (__atomic_load_n(&hdr->magic, __ATOMIC_ACQUIRE) != kMagic || hdr->nranks != nranks) {
        std::memset(base_, 0, shm_size_);
        hdr->stat_epoch = 0;
        hdr->nranks = nranks;
        hdr->param_version = 0;
        __atomic_store_n(&hdr->param_read_count, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&hdr->barrier_count, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&hdr->barrier_sense, 0u, __ATOMIC_SEQ_CST);
        // Publish readiness only after the complete header/table is initialized.
        __atomic_store_n(&hdr->magic, kMagic, __ATOMIC_RELEASE);
    } else {
        if (my_rank_ == 0) {
            char* p = static_cast<char*>(base_);
            p += sizeof(Header);
            StatSlot* slots = reinterpret_cast<StatSlot*>(p);
            for (int r = 0; r < nranks_; ++r) {
                __atomic_store_n(&slots[r].valid, 0, __ATOMIC_SEQ_CST);
            }
            __atomic_store_n(&hdr->stat_epoch, 0ULL, __ATOMIC_SEQ_CST);
            __atomic_store_n(&hdr->param_read_count, 0u, __ATOMIC_SEQ_CST);
            __atomic_store_n(&hdr->barrier_count, 0u, __ATOMIC_SEQ_CST);
            __atomic_store_n(&hdr->barrier_sense, 0u, __ATOMIC_SEQ_CST);
        }
    }
    shm_name_ = std::move(name);
    return true;
#else
    (void)key;
    return false;
#endif
}

ShmParamStore::~ShmParamStore() {
#if defined(__linux__) || defined(__APPLE__)
    if (base_ != nullptr && base_ != MAP_FAILED) {
        munmap(base_, shm_size_);
        base_ = nullptr;
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
    // Do not shm_unlink here. Multiple ranks and recreated communicators can
    // share the same logical-key segment; unlinking from one domain destructor
    // can break late attachers. Stale segments are validated/reset by header.
    shm_name_.clear();
#endif
}

void ShmParamStore::WriteMyStat(int my_rank, const OpKey& op_key, const ExecStat& stat) {
    if (base_ == nullptr || my_rank < 0 || my_rank >= nranks_) {
        return;
    }
    char* p = static_cast<char*>(base_);
    Header* hdr = static_cast<Header*>(base_);
    p += sizeof(Header);
    StatSlot* slot = reinterpret_cast<StatSlot*>(p + my_rank * sizeof(StatSlot));
    slot->epoch = __atomic_load_n(&hdr->stat_epoch, __ATOMIC_SEQ_CST);
    slot->op = static_cast<int>(op_key.op);
    slot->bytes = static_cast<uint64_t>(op_key.bytes);
    slot->bucket_bytes = static_cast<uint64_t>(op_key.bucket_bytes);
    slot->datatype = op_key.datatype;
    slot->nranks = op_key.nranks;
    slot->topology_hash = op_key.topology_hash;
    slot->fast_time = stat.fast_time;
    slot->pcie_time = stat.pcie_time;
    slot->fast_bytes = stat.fast_bytes;
    slot->pcie_bytes = stat.pcie_bytes;
    slot->fast_success = stat.fast_success ? 1 : 0;
    slot->pcie_success = stat.pcie_success ? 1 : 0;
    __atomic_store_n(&slot->valid, 1, __ATOMIC_SEQ_CST);
}

bool ShmParamStore::ReadAllStatsAndAggregate(ExecStat* out_global_stat, OpKey* out_op_key,
                                             double* out_algbw,
                                             double* out_busbw,
                                             double* out_bus_factor) const {
    if (base_ == nullptr || out_global_stat == nullptr || out_op_key == nullptr) {
        return false;
    }
    Header* hdr = static_cast<Header*>(base_);
    char* p = static_cast<char*>(base_);
    p += sizeof(Header);
    StatSlot* slots = reinterpret_cast<StatSlot*>(p);
    const uint64_t epoch = __atomic_load_n(&hdr->stat_epoch, __ATOMIC_SEQ_CST);

    bool any_current_epoch = false;
    for (int r = 0; r < nranks_; ++r) {
        const StatSlot* s = &slots[r];
        if (__atomic_load_n(&s->valid, __ATOMIC_SEQ_CST) != 0 && s->epoch == epoch) {
            any_current_epoch = true;
            break;
        }
    }
    if (!any_current_epoch) {
        return false;
    }

    for (;;) {
        bool all_current_epoch = true;
        for (int r = 0; r < nranks_; ++r) {
            const StatSlot* s = &slots[r];
            if (__atomic_load_n(&s->valid, __ATOMIC_SEQ_CST) == 0 || s->epoch != epoch) {
                all_current_epoch = false;
                break;
            }
        }
        if (all_current_epoch) {
            break;
        }
#if defined(__linux__) || defined(__APPLE__)
        sched_yield();
#endif
    }

    double max_fast_time = 0.0;
    double max_pcie_time = 0.0;
    uint64_t fast_bytes = 0;
    uint64_t pcie_bytes = 0;
    uint8_t fast_ok = 1;
    uint8_t pcie_ok = 1;
    bool op_key_set = false;

    for (int r = 0; r < nranks_; ++r) {
        const StatSlot* s = &slots[r];
        if (!op_key_set) {
            out_op_key->op = static_cast<CollectiveType>(s->op);
            out_op_key->bytes = static_cast<size_t>(s->bytes);
            out_op_key->bucket_bytes = static_cast<size_t>(s->bucket_bytes);
            out_op_key->datatype = s->datatype;
            out_op_key->nranks = s->nranks;
            out_op_key->topology_hash = s->topology_hash;
            fast_bytes = s->fast_bytes;
            pcie_bytes = s->pcie_bytes;
            op_key_set = true;
        } else if (static_cast<int>(out_op_key->op) != s->op ||
                   out_op_key->bytes != static_cast<size_t>(s->bytes) ||
                   out_op_key->bucket_bytes != static_cast<size_t>(s->bucket_bytes) ||
                   out_op_key->datatype != s->datatype || out_op_key->nranks != s->nranks ||
                   out_op_key->topology_hash != s->topology_hash ||
                   fast_bytes != s->fast_bytes || pcie_bytes != s->pcie_bytes) {
            AMPCCL_LOG(WARN,
                       "ShmParamStore: stat slot mismatch across ranks in epoch %" PRIu64
                       "; using first slot key/bytes",
                       epoch);
        }
        if (s->fast_time > max_fast_time) {
            max_fast_time = s->fast_time;
        }
        if (s->pcie_time > max_pcie_time) {
            max_pcie_time = s->pcie_time;
        }
        if (s->fast_success == 0) {
            fast_ok = 0;
        }
        if (s->pcie_success == 0) {
            pcie_ok = 0;
        }
    }
    out_global_stat->fast_time = max_fast_time;
    out_global_stat->pcie_time = max_pcie_time;
    out_global_stat->fast_bytes = static_cast<size_t>(fast_bytes);
    out_global_stat->pcie_bytes = static_cast<size_t>(pcie_bytes);
    AMPCCL_LOG(OFF, "(max) fast_time=%lfs", out_global_stat->fast_time);
    AMPCCL_LOG(OFF, "(max) pcie_time=%lfs", out_global_stat->pcie_time);
    double algbw = 0.0;
    double busbw = 0.0;
    double bus_factor = 0.0;
    {
        const size_t base_bytes = out_global_stat->fast_bytes + out_global_stat->pcie_bytes;
        const double t = out_global_stat->GetTotalTime();
        const CollectiveType op = out_op_key->op;
        const int n = out_op_key->nranks > 0 ? out_op_key->nranks : nranks_;
        // Decimal GB/s (1e9 B/s), same convention as tests/HCCL_TEST/common.h PrintRow.
        const double alg_bytes = static_cast<double>(base_bytes) * AlgBytesFactor(op, n);
        if (t > 1e-30 && alg_bytes > 0.0) {
            algbw = alg_bytes / t / 1e9;
            busbw = algbw * BusBwFactor(op, n);
        }
        bus_factor = BusBwFactor(op, n);
        const size_t fb = out_global_stat->fast_bytes;
        const size_t pb = out_global_stat->pcie_bytes;
        double fast_bw = 0.0;
        double pcie_bw = 0.0;
        if (out_global_stat->fast_time > 1e-30 && fb > 0) {
            fast_bw = static_cast<double>(fb) / out_global_stat->fast_time / 1e9;
        }
        if (out_global_stat->pcie_time > 1e-30 && pb > 0) {
            pcie_bw = static_cast<double>(pb) / out_global_stat->pcie_time / 1e9;
        }
        double fast_ratio = 0.0;
        double pcie_ratio = 0.0;
        if (base_bytes > 0) {
            fast_ratio = static_cast<double>(fb) / static_cast<double>(base_bytes);
            pcie_ratio = static_cast<double>(pb) / static_cast<double>(base_bytes);
        }
        AMPCCL_LOG(OFF,
                   "aggregate op=%s op_key_bytes=%zu base_bytes=%zu alg_bytes=%.0f n=%d | "
                   "fast_bytes=%zu pcie_bytes=%zu split_fast=%.4f split_pcie=%.4f | "
                   "fast_bw(GB/s,decimal)=%.6f pcie_bw(GB/s,decimal)=%.6f | "
                   "algbw(GB/s,decimal)=%.6f busbw(GB/s,decimal)=%.6f bus_factor=%.6f",
                   CollectiveTypeLabel(op), static_cast<size_t>(out_op_key->bytes), base_bytes, alg_bytes, n,
                   fb, pb, fast_ratio, pcie_ratio, fast_bw, pcie_bw, algbw, busbw, BusBwFactor(op, n));
        AMPCCL_LOG(OFF, "Total bw is %lf GB/s", algbw);
    }
    if (out_algbw) *out_algbw = algbw;
    if (out_busbw) *out_busbw = busbw;
    if (out_bus_factor) *out_bus_factor = bus_factor;
    out_global_stat->fast_success = (fast_ok != 0);
    out_global_stat->pcie_success = (pcie_ok != 0);
    for (int r = 0; r < nranks_; ++r) {
        __atomic_store_n(&slots[r].valid, 0, __ATOMIC_SEQ_CST);
    }
    __atomic_store_n(&hdr->stat_epoch, epoch + 1, __ATOMIC_SEQ_CST);
    return true;
}

namespace {
constexpr int kMaxConsistentReadRetries = 2000;
}  // namespace

void ShmParamStore::WaitUntilParamConsumed() {
    if (base_ == nullptr || !IsRank0()) {
        return;
    }
    Header* hdr = static_cast<Header*>(base_);
#if defined(__linux__)
    AdaptiveWait(&hdr->param_read_count, [hdr]() {
        return __atomic_load_n(&hdr->param_read_count, __ATOMIC_SEQ_CST) == 0u;
    });
#elif defined(__APPLE__)
    while (__atomic_load_n(&hdr->param_read_count, __ATOMIC_SEQ_CST) != 0u) {
        sched_yield();
    }
#else
    (void)hdr;
#endif
}

void ShmParamStore::SignalParamWritten() {
    if (base_ == nullptr || !IsRank0()) {
        return;
    }
    Header* hdr = static_cast<Header*>(base_);
    uint32_t count = static_cast<uint32_t>(nranks_ > 1 ? nranks_ - 1 : 0);
    __atomic_store_n(&hdr->param_read_count, count, __ATOMIC_SEQ_CST);
#if defined(__linux__)
    // Wake all non-rank0 processes waiting in ReadParams
    if (count > 0) {
        futex_wake(&hdr->param_read_count, INT_MAX);
    }
#endif
}

void ShmParamStore::BarrierSync() {
    if (base_ == nullptr || nranks_ <= 1) {
        return;
    }
    Header* hdr = static_cast<Header*>(base_);
#if defined(__linux__)
    uint32_t my_sense = 1u - __atomic_load_n(&hdr->barrier_sense, __ATOMIC_SEQ_CST);
    uint32_t count = __atomic_fetch_add(&hdr->barrier_count, 1u, __ATOMIC_SEQ_CST) + 1u;
    if (count == static_cast<uint32_t>(nranks_)) {
        // Last rank: reset barrier and wake all waiters
        __atomic_store_n(&hdr->barrier_count, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&hdr->barrier_sense, my_sense, __ATOMIC_SEQ_CST);
        futex_wake(&hdr->barrier_sense, INT_MAX);
    } else {
        // Wait for barrier_sense to flip
        AdaptiveWait(&hdr->barrier_sense, [hdr, my_sense]() {
            return __atomic_load_n(&hdr->barrier_sense, __ATOMIC_SEQ_CST) == my_sense;
        });
    }
#elif defined(__APPLE__)
    uint32_t my_sense = 1u - __atomic_load_n(&hdr->barrier_sense, __ATOMIC_SEQ_CST);
    uint32_t count = __atomic_fetch_add(&hdr->barrier_count, 1u, __ATOMIC_SEQ_CST) + 1u;
    if (count == static_cast<uint32_t>(nranks_)) {
        __atomic_store_n(&hdr->barrier_count, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&hdr->barrier_sense, my_sense, __ATOMIC_SEQ_CST);
    } else {
        while (__atomic_load_n(&hdr->barrier_sense, __ATOMIC_SEQ_CST) != my_sense) {
            sched_yield();
        }
    }
#endif
}

void ShmParamStore::ReadParams(ParamCache* cache) const {
    if (base_ == nullptr || cache == nullptr) {
        return;
    }
    char* p = static_cast<char*>(base_);
    Header* hdr = static_cast<Header*>(base_);
    size_t off = sizeof(Header) + static_cast<size_t>(kMaxRanks) * sizeof(StatSlot);
    uint64_t* pver = reinterpret_cast<uint64_t*>(p + off);
    uint32_t* pnum = reinterpret_cast<uint32_t*>(p + off + sizeof(uint64_t));
    ParamEntry* entries = reinterpret_cast<ParamEntry*>(p + off + sizeof(uint64_t) + sizeof(uint32_t));

#if defined(__linux__)
    if (!IsRank0()) {
        AdaptiveWait(&hdr->param_read_count, [hdr]() {
            return __atomic_load_n(&hdr->param_read_count, __ATOMIC_SEQ_CST) != 0u;
        });
    }
#elif defined(__APPLE__)
    if (!IsRank0()) {
        while (__atomic_load_n(&hdr->param_read_count, __ATOMIC_SEQ_CST) == 0u) {
            sched_yield();
        }
    }
#endif

    std::vector<std::pair<OpKey, ParamValue>> snapshot;
    for (int retry = 0; retry < kMaxConsistentReadRetries; ++retry) {
        uint64_t v1 = *pver;
        uint32_t n = *pnum;
        if (n > static_cast<uint32_t>(kMaxParamEntries)) {
            n = kMaxParamEntries;
        }
        snapshot.clear();
        snapshot.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            const ParamEntry& e = entries[i];
            OpKey key;
            key.op = static_cast<CollectiveType>(e.op);
            key.bytes = static_cast<size_t>(e.bytes);
            key.bucket_bytes = static_cast<size_t>(e.bucket_bytes);
            key.datatype = e.datatype;
            key.nranks = e.nranks;
            key.topology_hash = e.topology_hash;
            ParamValue val(e.alpha, e.use_pcie != 0, e.fast_bw, e.pcie_bw,
                           e.best_alpha, e.best_bandwidth);
            snapshot.emplace_back(key, val);
        }
        uint64_t v2 = *pver;
        if (v1 == v2) {
            break;
        }
#if defined(__linux__) || defined(__APPLE__)
        sched_yield();
#endif
    }
    for (const auto& kv : snapshot) {
        cache->MergeOne(kv.first, kv.second);
    }

#if defined(__linux__) || defined(__APPLE__)
    if (!IsRank0()) {
        __atomic_fetch_sub(&hdr->param_read_count, 1, __ATOMIC_SEQ_CST);
    }
#endif
    AMPCCL_LOG(INFO, "ShmParamStore: rank %d read params version=%" PRIu64 " entries=%zu",
               my_rank_, *pver, snapshot.size());
}

void ShmParamStore::WriteParams(const ParamCache& cache, const OpKey& op_key) {
    if (base_ == nullptr || !IsRank0()) {
        return;
    }
    ParamValue v = cache.Lookup(op_key);
    char* p = static_cast<char*>(base_);
    size_t off = sizeof(Header) + static_cast<size_t>(kMaxRanks) * sizeof(StatSlot);
    uint64_t* pver = reinterpret_cast<uint64_t*>(p + off);
    uint32_t* pnum = reinterpret_cast<uint32_t*>(p + off + sizeof(uint64_t));
    ParamEntry* entries = reinterpret_cast<ParamEntry*>(p + off + sizeof(uint64_t) + sizeof(uint32_t));

    ParamEntry& e = entries[0];
    e.op = static_cast<int>(op_key.op);
    e.bytes = static_cast<uint64_t>(op_key.bytes);
    e.bucket_bytes = static_cast<uint64_t>(op_key.bucket_bytes);
    e.datatype = op_key.datatype;
    e.nranks = op_key.nranks;
    e.topology_hash = op_key.topology_hash;
    e.alpha = v.alpha;
    e.use_pcie = v.use_pcie ? 1 : 0;
    std::memset(e.pad, 0, sizeof(e.pad));
    e.fast_bw = v.fast_bw;
    e.pcie_bw = v.pcie_bw;
    e.best_alpha = v.best_alpha;
    e.best_bandwidth = v.best_bandwidth;
    *pnum = 1u;
    (*pver)++;
}

}  // namespace ampccl
