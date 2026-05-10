#ifndef PCCL_COMPILED_GRAPH_HPP
#define PCCL_COMPILED_GRAPH_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "compiler.hpp"
#include "ir.hpp"

namespace pccl {

struct ChunkVersion {
  ProgressKey key;
  bool ring_buffered = false;
  int num_readers = 0;
};

struct NodeInput {
  ProgressKey key;
  int reader_id = -1;
  bool ring_buffered = false;
};

enum class D2DPlacement {
  Local,
};

struct InstructionNode {
  OpCode op = OpCode::NOP;
  int source_rank = -1;  // D2D source rank; -1 for non-D2D
  int destination_rank = -1;  // D2D destination rank; -1 for non-D2D
  D2DPlacement d2d_placement = D2DPlacement::Local;
  size_t work_bytes = 0;
  std::vector<NodeInput> inputs;
  ProgressKey output;
  bool output_ring_buffered = false;
  bool has_overwrite_guard = false;
  ProgressKey overwrite_guard;
  std::string resource_key;
  bool terminal = false;
};

struct ReaderAssignment {
  std::vector<ChunkVersion> chunks;
};

enum class StagingPolicy {
  Auto,
  Conservative,
  DirectOverwrite,
  ForceInternal,
};

struct LayoutDesc {
  size_t segment_size = 0;
  size_t stride_offset = 0;
  size_t chunk_size = 0;
  int chunks_per_segment = 1;
  std::vector<size_t> chunk_offsets;
  std::vector<size_t> chunk_sizes;
};

struct RuntimeBinding {
  char* host_base[MAX_NUMA_NODES] = {};
  size_t host_capacity[MAX_NUMA_NODES] = {};
  std::vector<int> host_chunk_remap[MAX_NUMA_NODES];
  char* device_base[ProgressSpaceCount][MAX_RANKS] = {};
  LayoutDesc layout[ProgressSpaceCount];
  ScalarDataType scalar_type = ScalarDataType::Float32;
  ReduceOpKind reduce_op = ReduceOpKind::Sum;
  size_t element_size = sizeof(float);
};

struct CompiledGraph {
  int rank = 0;
  int world_size = 0;
  int input_chunk_count = 0;
  int output_chunk_count = 0;
  size_t total_bytes = 0;
  size_t chunk_size = 0;
  size_t element_size = sizeof(float);
  size_t num_subchunks = 0;
  ScalarDataType scalar_type = ScalarDataType::Float32;
  ReduceOpKind reduce_op = ReduceOpKind::Sum;
  bool is_in_place = false;
  size_t internal_bytes = 0;
  LayoutDesc runtime_layout[ProgressSpaceCount];
  std::vector<InstructionNode> nodes;
  std::vector<ProgressKey> initial_ready_inputs;
  std::vector<ProgressKey> terminal_outputs;
  std::vector<ProgressKey> active_progress_keys;
  // Host addresses are shared across ranks, so compaction must use the
  // complete graph-wide logical host-key set rather than this rank's nodes.
  std::vector<ProgressKey> global_host_progress_keys;
  ReaderAssignment reader_assignment;
};

struct GraphCompileOptions {
  CompileOptions legacy_layout;
  StagingPolicy staging_policy = StagingPolicy::Conservative;
};

}  // namespace pccl

#endif /* PCCL_COMPILED_GRAPH_HPP */
