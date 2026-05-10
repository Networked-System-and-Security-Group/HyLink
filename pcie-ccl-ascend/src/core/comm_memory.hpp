#ifndef PCCL_COMM_MEMORY_HPP
#define PCCL_COMM_MEMORY_HPP

#include <cstddef>
#include <cstdint>

#include "limits.hpp"

namespace pccl {

// Device/host scratch is private to a live logical communicator.  comm_id is
// the cross-process slot; comm_hash keeps single-process communicators with
// slot 0 distinct as well.
struct CommMemoryKey {
  uint64_t comm_hash = 0;
  int comm_id = 0;
  int rank = -1;
  int device_id = -1;

  bool operator==(const CommMemoryKey& other) const {
    return comm_hash == other.comm_hash && comm_id == other.comm_id && rank == other.rank &&
           device_id == other.device_id;
  }
};

struct CommMemoryKeyHash {
  size_t operator()(const CommMemoryKey& key) const {
    size_t h = static_cast<size_t>(key.comm_hash ^ (key.comm_hash >> 32));
    h ^= static_cast<size_t>(key.comm_id) * static_cast<size_t>(0x9e3779b9U);
    h ^= static_cast<size_t>(key.rank + 1) * static_cast<size_t>(0x85ebca6bU);
    h ^= static_cast<size_t>(key.device_id + 1) * static_cast<size_t>(0xc2b2ae35U);
    return h;
  }
};

inline size_t InternalBufferAllocationBytes(size_t requested) {
  if (requested == 0 || requested > PCCL_INTERNAL_BUFFER_BYTES)
    return 0;
  constexpr size_t kGranularity = 2ULL * 1024ULL * 1024ULL;
  const size_t rounded = ((requested + kGranularity - 1) / kGranularity) * kGranularity;
  return rounded > PCCL_INTERNAL_BUFFER_BYTES ? PCCL_INTERNAL_BUFFER_BYTES : rounded;
}

inline bool InternalBufferGrowthAllowed(size_t current, size_t requested, bool has_inflight_work) {
  const size_t allocation = InternalBufferAllocationBytes(requested);
  return allocation > current && !has_inflight_work;
}

}  // namespace pccl

#endif  // PCCL_COMM_MEMORY_HPP
