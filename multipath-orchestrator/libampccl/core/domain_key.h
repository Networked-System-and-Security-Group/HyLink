#ifndef AMPCCL_CORE_DOMAIN_KEY_H_
#define AMPCCL_CORE_DOMAIN_KEY_H_

#include <vector>
#include <cstdint>
#include <functional>

namespace ampccl {

struct CommDomainKey {
    int world_size;
    std::vector<int> ranks;
    uint64_t topology_hash;

    bool operator==(const CommDomainKey& other) const {
        if (world_size != other.world_size ||
            topology_hash != other.topology_hash ||
            ranks.size() != other.ranks.size()) {
            return false;
        }
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (ranks[i] != other.ranks[i]) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace ampccl

// Hash function for CommDomainKey
namespace std {
template <>
struct hash<ampccl::CommDomainKey> {
    size_t operator()(const ampccl::CommDomainKey& key) const {
        size_t h = std::hash<int>{}(key.world_size);
        h ^= std::hash<uint64_t>{}(key.topology_hash) << 1;
        for (int rank : key.ranks) {
            h ^= std::hash<int>{}(rank) << 1;
        }
        return h;
    }
};
}  // namespace std

#endif  // AMPCCL_CORE_DOMAIN_KEY_H_
