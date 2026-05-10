#ifndef PCCL_COMM_HPP
#define PCCL_COMM_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "../hal/device_rt.hpp"
#include "../ir/compiled_graph.hpp"
#include "../ir/ir.hpp"
#include "pccl/types.h"

namespace pccl {
class GraphExecutor;
class SmallMessageExecutor;
class CompletionFlag;
class ProcessState;

// Cache key for IR + compiled graph reuse across repeated collective calls.
struct PlanCacheKey {
  std::string collective;
  size_t seg_count;
  size_t element_size;
  ScalarDataType scalar_type;
  ReduceOpKind reduce_op;
  size_t stride;
  int d2h_segments;
  int h2d_segments;
  bool is_in_place;

  bool operator==(const PlanCacheKey& o) const {
    return collective == o.collective &&
           seg_count == o.seg_count && element_size == o.element_size && scalar_type == o.scalar_type &&
           reduce_op == o.reduce_op && stride == o.stride && d2h_segments == o.d2h_segments &&
           h2d_segments == o.h2d_segments && is_in_place == o.is_in_place;
  }
};

struct PlanCacheKeyHash {
  size_t operator()(const PlanCacheKey& k) const {
    size_t h = std::hash<std::string>{}(k.collective);
    h ^= std::hash<size_t>{}(k.seg_count) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<size_t>{}(k.element_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(k.scalar_type)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(k.reduce_op)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<size_t>{}(k.stride) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.d2h_segments) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.h2d_segments) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(k.is_in_place) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct PlanCacheEntry {
  IRProgram program;
  CompiledGraph graph;
};

using PlanCache = std::unordered_map<PlanCacheKey, PlanCacheEntry, PlanCacheKeyHash>;

}  // namespace pccl

// pcclComm is defined in the global namespace to match the C API typedef in types.h
struct pcclComm {
  int rank;
  int world_size;
  int device_id;
  int comm_id;

  std::unique_ptr<pccl::CompletionFlag> completion_flag;
  std::unique_ptr<pccl::CompletionFlag> small_completion_flag;
  std::unique_ptr<pccl::GraphExecutor> graph_executor;
  std::unique_ptr<pccl::SmallMessageExecutor> small_message_executor;

  pccl::ProcessState* process_state;  // non-owning

  pccl::PlanCache plan_cache;

  ~pcclComm();
};

#endif /* PCCL_COMM_HPP */
