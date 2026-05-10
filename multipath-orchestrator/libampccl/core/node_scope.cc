#include "core/node_scope.h"
#include "common/log.h"

#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ampccl {
namespace {

#if defined(__linux__) || defined(__APPLE__)
constexpr bool kNodeScopeAvailable = true;
#else
constexpr bool kNodeScopeAvailable = false;
#endif

constexpr uint64_t kMagic = 0x414d5043434c4e53ULL;  // AMPCCLNS
constexpr int kMaxRanks = 1024;
constexpr uint64_t kSlotMaxAgeNs = 10ULL * 60ULL * 1000ULL * 1000ULL * 1000ULL;

struct NodeScopeHeader {
    uint64_t magic;
    uint64_t comm_hash;
    uint64_t node_hash;
    int nranks;
    int reserved;
};

struct NodeScopeSlot {
    uint8_t valid;
    uint8_t pad[3];
    int comm_rank;
    int pid;
    int device_id;
    int pad2;
    uint64_t timestamp_ns;
};

static int GetLocalDeviceId() {
    const char* env_lr = std::getenv("LOCAL_RANK");
    if (env_lr != nullptr && env_lr[0] != '\0') {
        return std::atoi(env_lr);
    }
    const char* env_dev = std::getenv("ASCEND_DEVICE_ID");
    if (env_dev != nullptr && env_dev[0] != '\0') {
        return std::atoi(env_dev);
    }
    return -1;
}

uint64_t HashBytes(const void* data, size_t len, uint64_t seed = 1469598103934665603ULL) {
    uint64_t h = seed;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; p != nullptr && i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(len) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

uint64_t HashString(const std::string& s, uint64_t seed = 1469598103934665603ULL) {
    return HashBytes(s.data(), s.size(), seed);
}

uint64_t NowNs() {
    using Clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

uint64_t LocalNodeHash() {
    std::string identity;
#if defined(__linux__) || defined(__APPLE__)
    char host[256] = {};
    if (gethostname(host, sizeof(host)) == 0) {
        identity += host;
    }
    identity += "|uid=" + std::to_string(static_cast<unsigned long long>(getuid()));
#if defined(__linux__)
    int fd = open("/proc/sys/kernel/random/boot_id", O_RDONLY);
    if (fd >= 0) {
        char buf[128] = {};
        ssize_t n = read(fd, buf, sizeof(buf));
        close(fd);
        if (n > 0) {
            identity += "|boot=";
            identity.append(buf, static_cast<size_t>(n));
        }
    }
#endif
#else
    identity = "unsupported";
#endif
    return HashString(identity);
}

std::string ShmNameForNodeScope(const LogicalCommKey& key, uint64_t node_hash) {
    std::ostringstream os;
    os << "/ampccl_node_" << std::hex << key.comm_hash << "_" << node_hash;
    return os.str();
}

size_t ShmSize() {
    return sizeof(NodeScopeHeader) + static_cast<size_t>(kMaxRanks) * sizeof(NodeScopeSlot);
}

bool PidAlive(int pid) {
#if defined(__linux__) || defined(__APPLE__)
    if (pid <= 0) {
        return false;
    }
    if (kill(pid, 0) == 0) {
        return true;
    }
    return errno == EPERM;
#else
    (void)pid;
    return false;
#endif
}

struct Mapping {
    int fd = -1;
    void* base = nullptr;
    size_t size = 0;
};

void CloseMapping(Mapping* m) {
#if defined(__linux__) || defined(__APPLE__)
    if (m->base != nullptr && m->base != MAP_FAILED) {
        munmap(m->base, m->size);
    }
    if (m->fd >= 0) {
        close(m->fd);
    }
#endif
    m->fd = -1;
    m->base = nullptr;
    m->size = 0;
}

bool OpenMapping(const LogicalCommKey& key, int nranks, uint64_t node_hash,
                 Mapping* out, bool initialize) {
    if (!kNodeScopeAvailable || key.world_size <= 0 || nranks <= 0 || nranks > kMaxRanks) {
        return false;
    }
#if defined(__linux__) || defined(__APPLE__)
    const std::string name = ShmNameForNodeScope(key, node_hash);
    const size_t size = ShmSize();
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        AMPCCL_LOG(WARN, "NodeScope: shm_open %s failed", name.c_str());
        return false;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        AMPCCL_LOG(WARN, "NodeScope: ftruncate %s failed", name.c_str());
        close(fd);
        return false;
    }
    void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        AMPCCL_LOG(WARN, "NodeScope: mmap %s failed", name.c_str());
        close(fd);
        return false;
    }
    out->fd = fd;
    out->base = base;
    out->size = size;

    if (initialize) {
        flock(fd, LOCK_EX);
        auto* hdr = static_cast<NodeScopeHeader*>(base);
        if (hdr->magic != kMagic || hdr->comm_hash != key.comm_hash ||
            hdr->node_hash != node_hash || hdr->nranks != nranks) {
            std::memset(base, 0, size);
            hdr->magic = kMagic;
            hdr->comm_hash = key.comm_hash;
            hdr->node_hash = node_hash;
            hdr->nranks = nranks;
        }
        flock(fd, LOCK_UN);
    }
    return true;
#else
    (void)key;
    (void)nranks;
    (void)node_hash;
    (void)out;
    (void)initialize;
    return false;
#endif
}

NodeScopeSlot* Slots(void* base) {
    return reinterpret_cast<NodeScopeSlot*>(static_cast<char*>(base) + sizeof(NodeScopeHeader));
}

int CountLocalSeen(void* base, int nranks, bool* out_contiguous = nullptr) {
    auto* hdr = static_cast<NodeScopeHeader*>(base);
    if (hdr->magic != kMagic || hdr->nranks != nranks) {
        return 0;
    }
    NodeScopeSlot* slots = Slots(base);
    const uint64_t now = NowNs();
    int seen = 0;
    bool contiguous = true;
    int first_dev = -1;
    for (int r = 0; r < nranks; ++r) {
        NodeScopeSlot* slot = &slots[r];
        if (__atomic_load_n(&slot->valid, __ATOMIC_SEQ_CST) == 0) {
            continue;
        }
        const int slot_rank = __atomic_load_n(&slot->comm_rank, __ATOMIC_SEQ_CST);
        const int pid = __atomic_load_n(&slot->pid, __ATOMIC_SEQ_CST);
        const int dev = __atomic_load_n(&slot->device_id, __ATOMIC_SEQ_CST);
        const uint64_t ts = __atomic_load_n(&slot->timestamp_ns, __ATOMIC_SEQ_CST);
        if (slot_rank == r && PidAlive(pid) && ts <= now && now - ts <= kSlotMaxAgeNs) {
            ++seen;
            if (dev >= 0) {
                if (r == 0) {
                    first_dev = dev;
                } else if (first_dev >= 0 && dev != first_dev + r) {
                    contiguous = false;
                }
            }
        }
    }
    if (seen == nranks && nranks == 2 && first_dev >= 0) {
        int dev0 = __atomic_load_n(&slots[0].device_id, __ATOMIC_SEQ_CST);
        int dev1 = __atomic_load_n(&slots[1].device_id, __ATOMIC_SEQ_CST);
        if (dev0 / 2 != dev1 / 2) {
            contiguous = false;
        }
    }
    if (out_contiguous) {
        *out_contiguous = (seen == nranks) ? contiguous : false;
    }
    return seen;
}

}  // namespace

bool RegisterNodeScopeRank(const LogicalCommKey& key, int rank, int nranks) {
    if (!kNodeScopeAvailable || rank < 0 || rank >= nranks || nranks > kMaxRanks) {
        return false;
    }
    const uint64_t node_hash = LocalNodeHash();
    Mapping mapping;
    if (!OpenMapping(key, nranks, node_hash, &mapping, true)) {
        return false;
    }
#if defined(__linux__) || defined(__APPLE__)
    flock(mapping.fd, LOCK_EX);
    NodeScopeSlot* slots = Slots(mapping.base);
    NodeScopeSlot* slot = &slots[rank];
    __atomic_store_n(&slot->valid, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&slot->comm_rank, rank, __ATOMIC_SEQ_CST);
    __atomic_store_n(&slot->pid, static_cast<int>(getpid()), __ATOMIC_SEQ_CST);
    __atomic_store_n(&slot->device_id, GetLocalDeviceId(), __ATOMIC_SEQ_CST);
    __atomic_store_n(&slot->timestamp_ns, NowNs(), __ATOMIC_SEQ_CST);
    __atomic_store_n(&slot->valid, 1, __ATOMIC_SEQ_CST);
    flock(mapping.fd, LOCK_UN);
#endif
    CloseMapping(&mapping);
    return true;
}

NodeScopeResult ResolveNodeScope(const LogicalCommKey& key, int nranks, int wait_ms) {
    NodeScopeResult result;
    result.nranks = nranks;
    result.comm_hash = key.comm_hash;
    result.node_hash = LocalNodeHash();
    if (!kNodeScopeAvailable || nranks <= 0 || nranks > kMaxRanks) {
        return result;
    }
    Mapping mapping;
    if (!OpenMapping(key, nranks, result.node_hash, &mapping, true)) {
        return result;
    }
    const int bounded_wait_ms = wait_ms < 0 ? 0 : wait_ms;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(bounded_wait_ms);
    bool contiguous = true;
    for (;;) {
        result.local_seen = CountLocalSeen(mapping.base, nranks, &contiguous);
        if (result.local_seen == nranks || std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    result.intra_node = (result.local_seen == nranks);
    result.allow_ampccl = result.intra_node && contiguous;
    AMPCCL_LOG(INFO,
               "NodeScope: comm_hash=%llu node_hash=%llu nranks=%d local_seen=%d contiguous=%d allow=%d",
               static_cast<unsigned long long>(result.comm_hash),
               static_cast<unsigned long long>(result.node_hash), nranks, result.local_seen,
               contiguous ? 1 : 0,
               result.allow_ampccl ? 1 : 0);
    CloseMapping(&mapping);
    return result;
}

NodeScopeResult DetectNodeScope(const LogicalCommKey& key, int rank, int nranks, int wait_ms) {
    (void)RegisterNodeScopeRank(key, rank, nranks);
    return ResolveNodeScope(key, nranks, wait_ms);
}

}  // namespace ampccl
