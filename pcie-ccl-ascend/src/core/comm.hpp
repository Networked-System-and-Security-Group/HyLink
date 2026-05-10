#ifndef PCCL_COMM_HPP
#define PCCL_COMM_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "../runtime/ascend_runtime.hpp"
#include "../ir/compiled_graph.hpp"
#include "../ir/ir.hpp"
#include "pccl/types.h"

namespace pccl {
class GraphExecutor;
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

  bool operator<(const PlanCacheKey& o) const {
    return std::tie(collective, seg_count, element_size, scalar_type, reduce_op, stride, d2h_segments,
                    h2d_segments, is_in_place) <
           std::tie(o.collective, o.seg_count, o.element_size, o.scalar_type, o.reduce_op, o.stride,
                    o.d2h_segments, o.h2d_segments, o.is_in_place);
  }
};

struct PlanCacheEntry {
  IRProgram program;
  CompiledGraph graph;
};

using PlanCache = std::map<PlanCacheKey, PlanCacheEntry>;

}  // namespace pccl

// pcclComm is defined in the global namespace to match the C API typedef in types.h
struct pcclComm {
  int rank;
  int world_size;
  int device_id;
  int comm_id;
  uint64_t comm_hash;

  std::unique_ptr<pccl::CompletionFlag> completion_flag;
  std::unique_ptr<pccl::GraphExecutor> graph_executor;

  pccl::ProcessState* process_state;  // non-owning

  pccl::PlanCache plan_cache;

  ~pcclComm();
};

#endif /* PCCL_COMM_HPP */
