#ifndef PCCL_TRANSPORT_HPP
#define PCCL_TRANSPORT_HPP

#include <cstddef>
#include <cstdint>

namespace pccl {

class Transport {
 public:
  virtual ~Transport() = default;

  // Multi-node barrier
  virtual int barrier() = 0;

  // Read a slice of data from a remote node
  virtual int readSlice(int remote_node, int numa_id, int chunk_idx, void* local_dst, size_t offset, size_t size) = 0;

  // Write a slice of data to a remote node
  virtual int writeSlice(int remote_node, int numa_id, int chunk_idx, const void* local_src, size_t offset,
                         size_t size) = 0;

  // Check if transport is connected
  virtual bool isConnected() const = 0;
};

}  // namespace pccl

#endif /* PCCL_TRANSPORT_HPP */
