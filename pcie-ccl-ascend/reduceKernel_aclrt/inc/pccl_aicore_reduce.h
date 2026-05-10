#ifndef PCCL_AICORE_REDUCE_H
#define PCCL_AICORE_REDUCE_H

#include <acl/acl_rt.h>

#include <cstddef>
#include <cstdint>

namespace pccl {
namespace aicore_reduce {

struct AicoreReducePartition {
  uint64_t begin_bytes;
  uint64_t end_bytes;
};

constexpr AicoreReducePartition GetAicoreReducePartition(uint64_t count_bytes, uint32_t block_index,
                                                          uint32_t block_count) {
  if (block_count == 0 || block_index >= block_count)
    return {0, 0};

  // Partition in 32-byte units.  The old proportional formula assigned the
  // rounding remainder to the last block.  That is the opposite of the
  // Ascend-C tiling convention (the first r blocks get one extra unit), and
  // makes a short trailing tile repeatedly land on the same core.  Computing
  // q/r explicitly also keeps the host contract identical to the device code.
  const uint64_t aligned_bytes = count_bytes & ~uint64_t{31};
  const uint64_t aligned_units = aligned_bytes / 32;
  const uint64_t units_per_block = aligned_units / block_count;
  const uint32_t extra_units = static_cast<uint32_t>(aligned_units % block_count);
  const uint64_t begin_units = units_per_block * block_index +
                               (block_index < extra_units ? block_index : extra_units);
  const uint64_t end_units = begin_units + units_per_block +
                             (block_index < extra_units ? 1 : 0);
  const uint64_t begin_bytes = begin_units * 32;
  const uint64_t end_bytes = end_units * 32;
  return {begin_bytes, end_bytes};
}

// Return the number of elements from a non-32-byte tail assigned to a block.
// This is deliberately independent from the aligned partition above: every
// block can therefore own a small padded tail without overlapping GM ranges.
constexpr uint32_t GetAicoreReduceTailElements(uint64_t count_bytes, size_t element_size,
                                                uint32_t block_index, uint32_t block_count,
                                                uint32_t tail_offset = 0) {
  if (block_count == 0 || block_index >= block_count || element_size == 0)
    return 0;
  const uint64_t aligned_bytes = count_bytes & ~uint64_t{31};
  const uint64_t tail_bytes = count_bytes - aligned_bytes;
  const uint32_t tail_elements = static_cast<uint32_t>(tail_bytes / element_size);
  const uint32_t base = tail_elements / block_count;
  const uint32_t extra = tail_elements % block_count;
  const uint32_t offset = tail_offset % block_count;
  const uint32_t distance = (block_index + block_count - offset) % block_count;
  return base + (distance < extra ? 1 : 0);
}

inline size_t AicoreReduceElementSize(aclDataType type) {
  switch (type) {
    case ACL_FLOAT16:
    case ACL_BF16:
      return sizeof(uint16_t);
    case ACL_FLOAT:
      return sizeof(float);
    default:
      return 0;
  }
}

inline bool IsAicoreReduceSupported(aclrtReduceKind kind, aclDataType type, uint64_t count_bytes) {
  const bool kind_supported = kind == ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM ||
                              kind == ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX ||
                              kind == ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN;
  const size_t element_size = AicoreReduceElementSize(type);
  return kind_supported && element_size != 0 && count_bytes != 0 && count_bytes % element_size == 0;
}

}  // namespace aicore_reduce
}  // namespace pccl

extern "C" __attribute__((visibility("default"))) aclError pcclAicoreReduceAsync(
    void* dst, const void* src, uint64_t count_bytes, aclrtReduceKind kind, aclDataType type,
    aclrtStream stream);

#endif  // PCCL_AICORE_REDUCE_H
