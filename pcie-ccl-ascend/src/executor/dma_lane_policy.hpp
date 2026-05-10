#ifndef PCCL_DMA_LANE_POLICY_HPP
#define PCCL_DMA_LANE_POLICY_HPP

#include <algorithm>
#include <cstddef>
#include <limits>

#include "../memory/completion_flag.hpp"

namespace pccl {

inline int selectHostDmaLaneCount(CollectiveDebugOp op, std::size_t message_bytes,
                                  int max_lanes, int forced_lanes, int world_size = 0) {
  const int lane_limit = std::max(1, max_lanes);
  const auto clamp = [lane_limit](int lanes) {
    return std::max(1, std::min(lanes, lane_limit));
  };

  if (message_bytes == 0)
    return 1;
  if (forced_lanes > 0)
    return clamp(forced_lanes);

  constexpr std::size_t mib = 1024 * 1024;
  switch (op) {
    case CollectiveDebugOp::AllGather:
      return clamp(message_bytes <= 16 * mib ? 12 : 4);
    case CollectiveDebugOp::ReduceScatter:
      return clamp(message_bytes <= 32 * mib ? 8 : message_bytes <= 64 * mib ? 12 : 4);
    case CollectiveDebugOp::AllReduce:
      return clamp(message_bytes <= 16 * mib ? 8
                                              : message_bytes <= 32 * mib ? 12
                                                                          : message_bytes <= 64 * mib ? 8 : 4);
    case CollectiveDebugOp::AllToAll:
      if (world_size == 4) {
        if (message_bytes <= 4 * mib)
          return clamp(8);
        if (message_bytes <= 8 * mib)
          return clamp(12);
        if (message_bytes <= 16 * mib)
          return clamp(8);
        if (message_bytes <= 32 * mib)
          return clamp(1);
        if (message_bytes <= 64 * mib)
          return clamp(8);
        return clamp(1);
      }
      if (world_size == 8) {
        if (message_bytes <= 1 * mib)
          return clamp(4);
        if (message_bytes <= 4 * mib)
          return clamp(12);
        if (message_bytes <= 32 * mib)
          return clamp(4);
        if (message_bytes <= 64 * mib)
          return clamp(12);
        return clamp(8);
      }
      return clamp(message_bytes <= 16 * mib ? 8
                                              : message_bytes <= 32 * mib ? 12
                                                                          : message_bytes <= 64 * mib ? 8 : 4);
    case CollectiveDebugOp::Broadcast:
    case CollectiveDebugOp::Unknown:
      return 1;
  }
  return 1;
}

inline bool allToAllLogicalElementCount(std::size_t segment_count, int world_size,
                                        std::size_t* total_count) {
  if (total_count == nullptr || world_size <= 0)
    return false;

  const std::size_t ranks = static_cast<std::size_t>(world_size);
  if (segment_count > std::numeric_limits<std::size_t>::max() / ranks)
    return false;

  *total_count = segment_count * ranks;
  return true;
}

}  // namespace pccl

#endif /* PCCL_DMA_LANE_POLICY_HPP */
