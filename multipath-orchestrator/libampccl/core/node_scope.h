#ifndef AMPCCL_CORE_NODE_SCOPE_H_
#define AMPCCL_CORE_NODE_SCOPE_H_

#include "core/logical_comm_key.h"
#include <cstdint>

namespace ampccl {

struct NodeScopeResult {
    bool intra_node = false;
    bool allow_ampccl = false;
    int local_seen = 0;
    int nranks = 0;
    uint64_t comm_hash = 0;
    uint64_t node_hash = 0;
};

bool RegisterNodeScopeRank(const LogicalCommKey& key, int rank, int nranks);
NodeScopeResult ResolveNodeScope(const LogicalCommKey& key, int nranks, int wait_ms);
NodeScopeResult DetectNodeScope(const LogicalCommKey& key, int rank, int nranks, int wait_ms);

}  // namespace ampccl

#endif  // AMPCCL_CORE_NODE_SCOPE_H_
