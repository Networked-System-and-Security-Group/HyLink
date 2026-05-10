#include "pccl_aicore_reduce.h"

#include <cassert>
#include <cstdint>

namespace {

void CheckAlignedPartitions(uint64_t count_bytes, uint32_t block_count) {
  using pccl::aicore_reduce::GetAicoreReducePartition;

  uint64_t previous_end = 0;
  for (uint32_t block_index = 0; block_index < block_count; ++block_index) {
    const auto partition = GetAicoreReducePartition(count_bytes, block_index, block_count);
    assert(partition.begin_bytes == previous_end);
    assert(partition.begin_bytes % 32 == 0);
    assert(partition.end_bytes % 32 == 0);
    assert(partition.end_bytes >= partition.begin_bytes);
    previous_end = partition.end_bytes;
  }
  assert(previous_end == (count_bytes & ~uint64_t{31}));
}

void CheckBalancedPartitions(uint64_t count_bytes, uint32_t block_count) {
  using pccl::aicore_reduce::GetAicoreReducePartition;

  const uint64_t units = (count_bytes & ~uint64_t{31}) / 32;
  const uint64_t base = units / block_count;
  const uint32_t extra = static_cast<uint32_t>(units % block_count);
  for (uint32_t block = 0; block < block_count; ++block) {
    const auto partition = GetAicoreReducePartition(count_bytes, block, block_count);
    const uint64_t block_units = (partition.end_bytes - partition.begin_bytes) / 32;
    assert(block_units == base + (block < extra ? 1 : 0));
  }
}

void CheckBalancedRawTail(uint64_t count_bytes, size_t element_size, uint32_t block_count) {
  using pccl::aicore_reduce::GetAicoreReduceTailElements;

  const uint64_t aligned_bytes = count_bytes & ~uint64_t{31};
  const uint32_t tail_elements = static_cast<uint32_t>((count_bytes - aligned_bytes) / element_size);
  const uint32_t base = tail_elements / block_count;
  const uint32_t extra = tail_elements % block_count;
  uint32_t total = 0;
  for (uint32_t block = 0; block < block_count; ++block) {
    const uint32_t assigned =
        GetAicoreReduceTailElements(count_bytes, element_size, block, block_count);
    assert(assigned == base + (block < extra ? 1 : 0));
    total += assigned;
  }
  assert(total == tail_elements);
}

void CheckRotatedRawTail(uint64_t count_bytes, size_t element_size, uint32_t block_count,
                         uint32_t tail_offset) {
  using pccl::aicore_reduce::GetAicoreReduceTailElements;

  const uint64_t aligned_bytes = count_bytes & ~uint64_t{31};
  const uint32_t tail_elements = static_cast<uint32_t>((count_bytes - aligned_bytes) / element_size);
  const uint32_t base = tail_elements / block_count;
  const uint32_t extra = tail_elements % block_count;
  uint32_t total = 0;
  for (uint32_t block = 0; block < block_count; ++block) {
    const uint32_t distance = (block + block_count - (tail_offset % block_count)) % block_count;
    const uint32_t expected = base + (distance < extra ? 1 : 0);
    const uint32_t assigned = GetAicoreReduceTailElements(
        count_bytes, element_size, block, block_count, tail_offset);
    assert(assigned == expected);
    total += assigned;
  }
  assert(total == tail_elements);
}

}  // namespace

int main()
{
  using pccl::aicore_reduce::AicoreReduceElementSize;
  using pccl::aicore_reduce::IsAicoreReduceSupported;

  assert(AicoreReduceElementSize(ACL_FLOAT16) == sizeof(uint16_t));
  assert(AicoreReduceElementSize(ACL_FLOAT) == sizeof(float));
  assert(AicoreReduceElementSize(ACL_BF16) == sizeof(uint16_t));
  assert(AicoreReduceElementSize(ACL_INT32) == 0);

  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_FLOAT16, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, ACL_FLOAT16, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, ACL_FLOAT16, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_FLOAT, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, ACL_FLOAT, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, ACL_FLOAT, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_BF16, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, ACL_BF16, 16));
  assert(IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, ACL_BF16, 16));

  assert(!IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_INT32, 16));
  assert(!IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, ACL_FLOAT16, 3));
  assert(!IsAicoreReduceSupported(ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL, ACL_FLOAT, 16));

  constexpr uint64_t kOneMiBWithFloatTail = 1024 * 1024 + 12;
  CheckAlignedPartitions(kOneMiBWithFloatTail, 8);
  CheckAlignedPartitions(kOneMiBWithFloatTail, 16);
  CheckAlignedPartitions(kOneMiBWithFloatTail, 24);
  CheckBalancedPartitions(1024 * 1024 + 12, 2);
  CheckBalancedPartitions(1024 * 1024 + 12, 3);
  CheckBalancedRawTail(1024 * 1024 + 12, sizeof(float), 2);
  CheckBalancedRawTail(1024 * 1024 + 14, sizeof(uint16_t), 2);
  CheckRotatedRawTail(1024 * 1024 + 12, sizeof(float), 2, 1);
  return 0;
}
