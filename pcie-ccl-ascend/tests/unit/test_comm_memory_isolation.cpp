#include <cstddef>

#include "core/comm_memory.hpp"

int main() {
  const pccl::CommMemoryKey world{0x11, 3, 0, 0};
  const pccl::CommMemoryKey child4{0x22, 7, 0, 0};
  const pccl::CommMemoryKey child2{0x33, 9, 0, 0};

  if (world == child4 || world == child2 || child4 == child2)
    return 1;
  if (!(world == pccl::CommMemoryKey{0x11, 3, 0, 0}))
    return 2;

  constexpr size_t requested = 3276292096ULL;
  const size_t allocated = pccl::InternalBufferAllocationBytes(requested);
  if (allocated < requested || allocated > pccl::PCCL_INTERNAL_BUFFER_BYTES)
    return 3;
  if (pccl::InternalBufferAllocationBytes(0) != 0)
    return 4;
  if (pccl::InternalBufferGrowthAllowed(2ULL * 1024 * 1024, 4ULL * 1024 * 1024, true))
    return 5;
  if (!pccl::InternalBufferGrowthAllowed(2ULL * 1024 * 1024, 4ULL * 1024 * 1024, false))
    return 6;

  return 0;
}
