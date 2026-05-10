#ifndef AMPCCL_CORE_LOGICAL_COMM_KEY_H_
#define AMPCCL_CORE_LOGICAL_COMM_KEY_H_

#include <cstdint>
#include <functional>

namespace ampccl {

struct LogicalCommKey {
    int world_size = 0;
    uint64_t comm_hash = 0;

    bool operator==(const LogicalCommKey& other) const {
        return world_size == other.world_size && comm_hash == other.comm_hash;
    }
};

}  // namespace ampccl

namespace std {
template <>
struct hash<ampccl::LogicalCommKey> {
    size_t operator()(const ampccl::LogicalCommKey& key) const {
        size_t h = std::hash<int>{}(key.world_size);
        h ^= std::hash<uint64_t>{}(key.comm_hash) << 1;
        return h;
    }
};
}  // namespace std

#endif  // AMPCCL_CORE_LOGICAL_COMM_KEY_H_
