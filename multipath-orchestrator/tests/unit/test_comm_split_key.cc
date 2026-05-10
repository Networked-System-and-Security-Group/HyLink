#include "libampccl/core/comm_init.h"

#include <cassert>

int main() {
    auto parent = ampccl::BuildLogicalKeyFromBootstrap(4, "parent", 6, 0);

    auto child_a = ampccl::BuildLogicalKeyFromCommSplit(parent, 2, 0, 2);
    auto child_a_again = ampccl::BuildLogicalKeyFromCommSplit(parent, 2, 0, 2);
    auto child_b = ampccl::BuildLogicalKeyFromCommSplit(parent, 2, 1, 2);
    auto nested_child = ampccl::BuildLogicalKeyFromCommSplit(child_a, 1, 0, 1);

    assert(child_a == child_a_again);
    assert(!(child_a == child_b));
    assert(!(child_a == parent));
    assert(!(nested_child == child_a));
    assert(child_a.world_size == 2);
    assert(nested_child.world_size == 1);

    auto local_key = ampccl::BuildKeyFromNcclCommSplit(parent, 2, 0, 2, 1);
    assert(local_key.world_size == 2);
    assert(local_key.ranks.size() == 1);
    assert(local_key.ranks[0] == 1);
    assert(local_key.topology_hash == child_a.comm_hash);

    uint32_t rank_ids_a[] = {0, 2};
    uint32_t rank_ids_b[] = {1, 3};
    auto hccl_child_a = ampccl::BuildLogicalKeyFromHcclSubComm(parent, 2, rank_ids_a, 1234);
    auto hccl_child_a_again = ampccl::BuildLogicalKeyFromHcclSubComm(parent, 2, rank_ids_a, 1234);
    auto hccl_child_different_ranks = ampccl::BuildLogicalKeyFromHcclSubComm(parent, 2, rank_ids_b, 1234);
    auto hccl_child_different_id = ampccl::BuildLogicalKeyFromHcclSubComm(parent, 2, rank_ids_a, 5678);
    assert(hccl_child_a == hccl_child_a_again);
    assert(!(hccl_child_a == parent));
    assert(!(hccl_child_a == hccl_child_different_ranks));
    assert(!(hccl_child_a == hccl_child_different_id));
    assert(hccl_child_a.world_size == 2);

    auto hccl_local_key = ampccl::BuildKeyFromHcclSubComm(parent, 2, rank_ids_a, 1234, 1);
    assert(hccl_local_key.world_size == 2);
    assert(hccl_local_key.ranks.size() == 1);
    assert(hccl_local_key.ranks[0] == 1);
    assert(hccl_local_key.topology_hash == hccl_child_a.comm_hash);

    return 0;
}
