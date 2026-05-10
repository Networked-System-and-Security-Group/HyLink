#ifndef PCCL_HOST_POOL_HPP
#define PCCL_HOST_POOL_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pccl {

struct LocalTopology;
struct BootstrapShm;

class HostMemoryPool {
 public:
  // Single-process mode: allocates via devMallocHost (existing behavior)
  HostMemoryPool(int nranks, size_t buffer_size, const LocalTopology* topo);

  // Multi-process mode: shared hugepage host staging via memfd_create(MFD_HUGETLB)
  // + mbind + devHostRegister.
  HostMemoryPool(int nranks, size_t buffer_size, const LocalTopology* topo,
                 const std::string& unique_id, bool is_creator);
  HostMemoryPool(int nranks, size_t buffer_size, const LocalTopology* topo,
                 const std::string& unique_id, bool is_creator, const std::string& namespace_suffix);

  ~HostMemoryPool();

  HostMemoryPool(const HostMemoryPool&) = delete;
  HostMemoryPool& operator=(const HostMemoryPool&) = delete;

  // Get buffer for a specific chunk on a specific NUMA node.
  // chunk_size is determined by the caller (compiler) based on actual data size.
  void* getChunkBuffer(int numa_node, int chunk_idx, size_t chunk_size);

  // Get total buffer size per NUMA node
  size_t getBufferSize() const { return buffer_size_; }
  size_t getAllocatedSize(int numa_node) const;

  // Get the NUMA node for a given rank
  int getRankNUMANode(int rank) const;

  // Check if pool is valid
  bool isValid() const { return !numa_buffers_.empty(); }

  // Return the set of NUMA node IDs that have allocated buffers
  std::vector<int> numaNodes() const {
    std::vector<int> nodes;
    for (const auto& kv : numa_buffers_) nodes.push_back(kv.first);
    return nodes;
  }

 private:
  std::map<int, void*> numa_buffers_;    // numa_node -> base ptr
  std::map<int, size_t> numa_sizes_;     // numa_node -> total size
  std::map<int, int> numa_fds_;          // numa_node -> shared memfd (multi-process, creator only)
  size_t buffer_size_;                   // usable buffer size per NUMA node
  int nranks_;
  const LocalTopology* topo_;

  bool multi_process_ = false;
  bool is_creator_ = false;
  std::string unique_id_;

  // Data offset: 0 for hugepage mode (preserve 2MB alignment), 64 for devMallocHost
  size_t alignedDataOffset() const { return multi_process_ ? 0 : 64; }
};

}  // namespace pccl

#endif /* PCCL_HOST_POOL_HPP */
