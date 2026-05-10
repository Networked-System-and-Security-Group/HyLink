#include <cstddef>
#include <limits>

#include "executor/dma_lane_policy.hpp"

int main() {
  using pccl::CollectiveDebugOp;
  constexpr size_t mib = 1024 * 1024;

  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 16 * mib, 12, -1) != 8)
    return 1;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 32 * mib, 12, -1) != 12)
    return 2;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 64 * mib, 12, -1) != 8)
    return 3;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 65 * mib, 12, -1) != 4)
    return 4;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 1 * mib, 12, 3) != 3)
    return 5;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 1 * mib, 4, 12) != 4)
    return 6;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 0, 12, -1) != 1)
    return 7;

  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllGather, 16 * mib, 12, -1) != 12)
    return 8;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::ReduceScatter, 32 * mib, 12, -1) != 8)
    return 9;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllReduce, 32 * mib, 12, -1) != 12)
    return 10;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::Unknown, 1 * mib, 12, -1) != 1)
    return 11;

  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 1 * mib, 12, -1, 4) != 8)
    return 15;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 4 * mib, 12, -1, 4) != 8)
    return 16;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 8 * mib, 12, -1, 4) != 12)
    return 17;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 16 * mib, 12, -1, 4) != 8)
    return 18;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 32 * mib, 12, -1, 4) != 1)
    return 19;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 64 * mib, 12, -1, 4) != 8)
    return 20;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 65 * mib, 12, -1, 4) != 1)
    return 21;

  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 1 * mib, 12, -1, 8) != 4)
    return 22;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 4 * mib, 12, -1, 8) != 12)
    return 23;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 8 * mib, 12, -1, 8) != 4)
    return 24;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 64 * mib, 12, -1, 8) != 12)
    return 25;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 65 * mib, 12, -1, 8) != 8)
    return 26;
  if (pccl::selectHostDmaLaneCount(CollectiveDebugOp::AllToAll, 1 * mib, 4, 12, 8) != 4)
    return 27;

  size_t total = 0;
  if (!pccl::allToAllLogicalElementCount(7, 4, &total) || total != 28)
    return 12;
  if (pccl::allToAllLogicalElementCount(std::numeric_limits<size_t>::max(), 2, &total))
    return 13;
  if (pccl::allToAllLogicalElementCount(7, 4, nullptr))
    return 14;
  return 0;
}
