#ifndef PCCL_COMPILER_HPP
#define PCCL_COMPILER_HPP

#include <cstddef>

#include "ir.hpp"

namespace pccl {

class HostMemoryPool;
class ProcessState;
struct CompiledGraph;
struct GraphCompileOptions;
struct RuntimeBinding;

enum class InPlaceAliasMode {
  SameBase,
  AllGather,
};

enum class ScalarDataType {
  Int8 = 0,
  Uint8 = 1,
  Int32 = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Float32 = 7,
  Float64 = 8,
  Bfloat16 = 9,
};

enum class ReduceOpKind {
  Sum,
  Max,
  Min,
};

// Controls how the Compiler maps IR chunk_idx to physical device addresses.
//
// When MPCCL splits a collective into fast_path (e.g. NVLink) + pcie_path,
// PCCL only handles the data *after* the fast-path region in each per-rank
// segment.  For segmented buffers (e.g. AllGather recvbuf), this produces a
// strided access pattern:
//
//   segment_size   = seg_count * elem_size
//   stride_offset  = stride * elem_size
//   pcie_per_seg   = segment_size - stride_offset
//   chunks_per_seg = device_chunks / num_segments
//   chunk_size     = pcie_per_seg / chunks_per_seg
//
//   addr = base + seg * segment_size + stride_offset + chunk_in_seg * chunk_size
//          where seg = chunk_idx / chunks_per_seg
//                chunk_in_seg = chunk_idx % chunks_per_seg
//
// When stride == 0 and segments == 1, the formula collapses to contiguous layout.
struct CompileOptions {
  size_t seg_count;     // Per-segment element count (fast + pcie combined)
  size_t element_size;  // Bytes per element.
  ScalarDataType scalar_type;
  ReduceOpKind reduce_op;
  size_t stride;        // Fast-path element count per segment; PCCL operates on
                        // the (seg_count - stride) elements after this region.
                        // 0 = no split, contiguous layout.
  int d2h_segments;     // Number of per-rank segments in sendbuf (1 or nranks)
  int h2d_segments;     // Number of per-rank segments in recvbuf (1 or nranks)
  bool is_in_place;     // sendbuff == recvbuff; final device result is staged through internal.
  InPlaceAliasMode in_place_alias_mode;
  ProcessState* process_state;
  int comm_id;
  int local_rank;
  int local_device_id;

  CompileOptions()
      : seg_count(0),
        element_size(sizeof(float)),
        scalar_type(ScalarDataType::Float32),
        reduce_op(ReduceOpKind::Sum),
        stride(0),
        d2h_segments(1),
        h2d_segments(1),
        is_in_place(false),
        in_place_alias_mode(InPlaceAliasMode::SameBase),
        process_state(nullptr),
        comm_id(0),
        local_rank(-1),
        local_device_id(-1) {}
};

class Compiler {
 public:
  // Compile address-free graph IR. Runtime buffers are supplied later via RuntimeBinding.
  static CompiledGraph compileGraph(const IRProgram& program, int rank, int world_size,
                                    const GraphCompileOptions& options);

  static RuntimeBinding makeRuntimeBinding(const CompiledGraph& graph, const CompileOptions& options,
                                           void* sendbuff, void* recvbuff, HostMemoryPool* host_pool);
};

}  // namespace pccl

#endif /* PCCL_COMPILER_HPP */
