#include <cstddef>

#include "core/limits.hpp"

int main() {
  return pccl::PCCL_INTERNAL_BUFFER_BYTES == (4ULL << 30) ? 0 : 1;
}
