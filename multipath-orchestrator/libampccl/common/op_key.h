#ifndef AMPCCL_COMMON_OP_KEY_H_
#define AMPCCL_COMMON_OP_KEY_H_

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ampccl {

enum class CollectiveType {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    Reduce,
    AllToAll
};

struct OpKey {
    CollectiveType op;
    size_t bytes;
    int datatype;  // NCCL/HCCL datatype enum value

    bool operator==(const OpKey& other) const {
        return op == other.op && bytes == other.bytes && datatype == other.datatype;
    }
};

}  // namespace ampccl

// Hash function for OpKey
namespace std {
template <>
struct hash<ampccl::OpKey> {
    size_t operator()(const ampccl::OpKey& key) const {
        size_t h1 = std::hash<int>{}(static_cast<int>(key.op));
        size_t h2 = std::hash<size_t>{}(key.bytes);
        size_t h3 = std::hash<int>{}(key.datatype);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
}  // namespace std

#endif  // AMPCCL_COMMON_OP_KEY_H_
