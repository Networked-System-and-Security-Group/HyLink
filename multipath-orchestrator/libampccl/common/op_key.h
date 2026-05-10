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
    AllToAll
};

struct OpKey {
    CollectiveType op;
    size_t bytes;        // Real message bytes for execution, stats, and logs.
    size_t bucket_bytes; // Reuse key for adaptive params; 0 means no reusable size bucket.
    int datatype;       // backend-native enum value (NCCL or HCCL, see dtype_* headers)
    int nranks;         // communicator world size for this adaptive domain
    uint64_t topology_hash;

    size_t ParamBytes() const { return bucket_bytes != 0 ? bucket_bytes : bytes; }

    bool operator==(const OpKey& other) const {
        return op == other.op && ParamBytes() == other.ParamBytes() && datatype == other.datatype &&
               nranks == other.nranks && topology_hash == other.topology_hash;
    }
};

}  // namespace ampccl

// Hash function for OpKey
namespace std {
template <>
struct hash<ampccl::OpKey> {
    size_t operator()(const ampccl::OpKey& key) const {
        size_t h1 = std::hash<int>{}(static_cast<int>(key.op));
        size_t h2 = std::hash<size_t>{}(key.ParamBytes());
        size_t h3 = std::hash<int>{}(key.datatype);
        size_t h4 = std::hash<int>{}(key.nranks);
        size_t h5 = std::hash<uint64_t>{}(key.topology_hash);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};
}  // namespace std

#endif  // AMPCCL_COMMON_OP_KEY_H_
