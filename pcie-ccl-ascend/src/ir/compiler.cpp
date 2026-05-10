#include "compiler.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "../memory/host_pool.hpp"
#include "../core/process_state.hpp"
#include "compiled_graph.hpp"
#include "../util/log.hpp"

namespace pccl {

static size_t divUp(size_t a, size_t b) {
  return b == 0 ? 0 : (a + b - 1) / b;
}

static LayoutDesc makeLayout(size_t segment_size, size_t stride_offset, size_t chunk_size, int chunks_per_segment) {
  LayoutDesc d;
  d.segment_size = segment_size;
  d.stride_offset = stride_offset;
  d.chunk_size = chunk_size;
  d.chunks_per_segment = std::max(1, chunks_per_segment);
  return d;
}

static LayoutDesc makeDeviceLayout(size_t segment_size, size_t stride_offset, size_t element_size,
                                   size_t active_elems, int total_chunks, int chunks_per_segment,
                                   size_t slot_size) {
  LayoutDesc d = makeLayout(segment_size, stride_offset, slot_size, chunks_per_segment);
  int cps = std::max(1, chunks_per_segment);
  int chunks = std::max(0, total_chunks);
  d.chunk_offsets.assign(static_cast<size_t>(chunks), 0);
  d.chunk_sizes.assign(static_cast<size_t>(chunks), 0);
  for (int c = 0; c < chunks; ++c) {
    int chunk_in_seg = c % cps;
    size_t begin_elem = (active_elems * static_cast<size_t>(chunk_in_seg)) / static_cast<size_t>(cps);
    size_t end_elem = (active_elems * static_cast<size_t>(chunk_in_seg + 1)) / static_cast<size_t>(cps);
    d.chunk_offsets[static_cast<size_t>(c)] = begin_elem * element_size;
    d.chunk_sizes[static_cast<size_t>(c)] = (end_elem - begin_elem) * element_size;
  }
  return d;
}

static size_t maxChunkBytes(size_t active_elems, size_t element_size, int chunks_per_segment) {
  int cps = std::max(1, chunks_per_segment);
  return divUp(active_elems, static_cast<size_t>(cps)) * element_size;
}

static size_t chunkBytesForKey(const ProgressKey& key, const CompiledGraph& graph) {
  if (key.space == ProgressSpace::HOST || key.space == ProgressSpace::INTERNAL)
    return graph.chunk_size;
  int idx = static_cast<int>(key.space);
  if (idx < 0 || idx >= ProgressSpaceCount)
    return graph.chunk_size;
  const auto& sizes = graph.runtime_layout[idx].chunk_sizes;
  if (key.chunk >= 0 && static_cast<size_t>(key.chunk) < sizes.size())
    return sizes[static_cast<size_t>(key.chunk)];
  return graph.chunk_size;
}

static bool sameBaseKey(const ProgressKey& a, const ProgressKey& b) {
  return a.space == b.space && a.owner == b.owner && a.chunk == b.chunk;
}

static void addUniqueKey(std::vector<ProgressKey>& keys, const ProgressKey& key) {
  auto it = std::find_if(keys.begin(), keys.end(), [&](const ProgressKey& existing) { return existing == key; });
  if (it == keys.end())
    keys.push_back(key);
}

namespace compiler_detail {

std::vector<int> BuildDenseHostChunkMap(const std::vector<ProgressKey>& active_keys, int numa_node) {
  std::vector<int> remap(MAX_CHUNKS, -1);
  std::vector<int> logical_chunks;
  for (const auto& key : active_keys) {
    if (key.space != ProgressSpace::HOST || key.owner != numa_node || key.chunk < 0 || key.chunk >= MAX_CHUNKS)
      continue;
    logical_chunks.push_back(key.chunk);
  }

  std::sort(logical_chunks.begin(), logical_chunks.end());
  logical_chunks.erase(std::unique(logical_chunks.begin(), logical_chunks.end()), logical_chunks.end());
  for (size_t physical_slot = 0; physical_slot < logical_chunks.size(); ++physical_slot)
    remap[static_cast<size_t>(logical_chunks[physical_slot])] = static_cast<int>(physical_slot);
  return remap;
}

}  // namespace compiler_detail

static ProgressKey hostKey(int numa, int chunk, int version) {
  return ProgressKey{ProgressSpace::HOST, numa, chunk, version};
}

static ProgressKey deviceKey(DeviceBufferKind kind, int owner, int chunk, int version) {
  return ProgressKey{progressSpaceForDeviceBuffer(kind), owner, chunk, version};
}

static int maxInputChunkCount(const IRProgram& program, int rank) {
  int count = program.input_chunk_count;
  for (const auto& inst : program.all_instructions) {
    if ((inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
        inst.src_buf == DeviceBufferKind::INPUT && inst.src_rank == rank) {
      count = std::max(count, inst.src.chunk_idx + 1);
    }
  }
  return count;
}

CompiledGraph Compiler::compileGraph(const IRProgram& program, int rank, int world_size,
                                     const GraphCompileOptions& graph_options) {
  const auto& options = graph_options.legacy_layout;

  CompiledGraph graph;
  graph.rank = rank;
  graph.world_size = world_size;
  graph.input_chunk_count = maxInputChunkCount(program, rank);
  graph.output_chunk_count = program.output_chunk_count;
  graph.element_size = options.element_size;
  graph.scalar_type = options.scalar_type;
  graph.reduce_op = options.reduce_op;
  graph.is_in_place = options.is_in_place;

  size_t segment_size = options.seg_count * options.element_size;
  size_t stride_offset = options.stride * options.element_size;
  size_t pcie_per_seg = segment_size - stride_offset;
  size_t active_elems = options.seg_count - options.stride;
  int d2h_cps = std::max(1, program.input_chunk_count / std::max(1, options.d2h_segments));
  int h2d_cps = std::max(1, program.output_chunk_count / std::max(1, options.h2d_segments));
  graph.chunk_size = std::max(maxChunkBytes(active_elems, options.element_size, d2h_cps),
                              maxChunkBytes(active_elems, options.element_size, h2d_cps));
  graph.total_bytes = pcie_per_seg * std::max(options.d2h_segments, options.h2d_segments);
  graph.num_subchunks = (graph.chunk_size + SUBCHUNK_SLICE_SIZE - 1) / SUBCHUNK_SLICE_SIZE;
  if (graph.num_subchunks == 0)
    graph.num_subchunks = 1;

  graph.runtime_layout[static_cast<int>(ProgressSpace::HOST)] =
      makeLayout(segment_size, 0, graph.chunk_size, 1);
  graph.runtime_layout[static_cast<int>(ProgressSpace::INPUT)] =
      makeDeviceLayout(segment_size, stride_offset, options.element_size, active_elems,
                       graph.input_chunk_count, d2h_cps, graph.chunk_size);
  graph.runtime_layout[static_cast<int>(ProgressSpace::OUTPUT)] =
      makeDeviceLayout(segment_size, stride_offset, options.element_size, active_elems,
                       graph.output_chunk_count, h2d_cps, graph.chunk_size);
  graph.runtime_layout[static_cast<int>(ProgressSpace::SCRATCH)] =
      makeDeviceLayout(segment_size, stride_offset, options.element_size, active_elems,
                       graph.output_chunk_count, h2d_cps, graph.chunk_size);
  graph.runtime_layout[static_cast<int>(ProgressSpace::INTERNAL)] =
      makeLayout(graph.chunk_size, 0, graph.chunk_size, 1);

  auto add_global_host_key = [&](int numa, int chunk) {
    if (numa >= 0 && numa < MAX_NUMA_NODES && chunk >= 0 && chunk < MAX_CHUNKS)
      addUniqueKey(graph.global_host_progress_keys, hostKey(numa, chunk, 0));
  };
  for (const auto& inst : program.all_instructions) {
    switch (inst.op) {
      case OpCode::D2H:
        add_global_host_key(inst.dst.numa_id, inst.dst.chunk_idx);
        break;
      case OpCode::H2D:
        add_global_host_key(inst.src.numa_id, inst.src.chunk_idx);
        break;
      case OpCode::H2H:
        add_global_host_key(inst.src.numa_id, inst.src.chunk_idx);
        add_global_host_key(inst.dst.numa_id, inst.dst.chunk_idx);
        break;
      case OpCode::D2D:
      case OpCode::D2DReduce:
      case OpCode::NOP:
        break;
    }
  }

  std::vector<std::vector<bool>> stage_output_chunk_by_rank(
      world_size, std::vector<bool>(program.output_chunk_count, false));
  if (options.is_in_place) {
    std::vector<std::vector<bool>> input_read_by_rank(world_size);
    for (int r = 0; r < world_size; ++r) {
      input_read_by_rank[r].assign(std::max(1, maxInputChunkCount(program, r)), false);
    }

    auto mark_input_read = [&](int owner, int chunk) {
      if (owner < 0 || owner >= world_size || chunk < 0)
        return;
      if (chunk >= static_cast<int>(input_read_by_rank[owner].size()))
        input_read_by_rank[owner].resize(chunk + 1, false);
      input_read_by_rank[owner][chunk] = true;
    };

    for (const auto& inst : program.all_instructions) {
      switch (inst.op) {
        case OpCode::D2H:
          mark_input_read(inst.rank, inst.src.chunk_idx);
          break;
        case OpCode::D2D: {
          int src_rank = (inst.src_rank >= 0) ? inst.src_rank : inst.rank;
          if (inst.src_buf == DeviceBufferKind::INPUT)
            mark_input_read(src_rank, inst.src.chunk_idx);
          break;
        }
        case OpCode::D2DReduce: {
          int src_rank = (inst.src_rank >= 0) ? inst.src_rank : inst.rank;
          if (inst.src_buf == DeviceBufferKind::INPUT)
            mark_input_read(src_rank, inst.src.chunk_idx);
          break;
        }
        default:
          break;
      }
    }

    auto output_overlaps_live_input = [&](int owner, int output_chunk) {
      if (owner < 0 || owner >= world_size || output_chunk < 0)
        return false;
      if (options.in_place_alias_mode == InPlaceAliasMode::AllGather) {
        int input_chunk = output_chunk - owner * std::max(1, program.input_chunk_count);
        return input_chunk >= 0 &&
               input_chunk < static_cast<int>(input_read_by_rank[owner].size()) &&
               input_read_by_rank[owner][input_chunk];
      }
      return output_chunk < static_cast<int>(input_read_by_rank[owner].size()) &&
             input_read_by_rank[owner][output_chunk];
    };

    for (const auto& inst : program.all_instructions) {
      if ((inst.op == OpCode::H2D || inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
          inst.dst_buf == DeviceBufferKind::OUTPUT &&
          inst.rank >= 0 && inst.rank < world_size &&
          inst.dst.chunk_idx >= 0 && inst.dst.chunk_idx < program.output_chunk_count &&
          output_overlaps_live_input(inst.rank, inst.dst.chunk_idx)) {
        if (graph_options.staging_policy == StagingPolicy::DirectOverwrite ||
            graph_options.staging_policy == StagingPolicy::Auto) {
          continue;
        }
        stage_output_chunk_by_rank[inst.rank][inst.dst.chunk_idx] = true;
      }
    }
  }
  const auto& stage_output_chunk = stage_output_chunk_by_rank[rank];
  int max_internal_chunk = -1;
  for (const auto& staged : stage_output_chunk) {
    (void)staged;
  }
  for (const auto& inst : program.all_instructions) {
    auto mark_internal = [&](DeviceBufferKind buf, int owner, int chunk) {
      if (owner == rank && buf == DeviceBufferKind::INTERNAL && chunk >= 0)
        max_internal_chunk = std::max(max_internal_chunk, chunk);
    };
    mark_internal(inst.src_buf, (inst.src_rank >= 0) ? inst.src_rank : inst.rank, inst.src.chunk_idx);
    mark_internal(inst.dst_buf, inst.rank, inst.dst.chunk_idx);
  }
  for (int c = 0; c < static_cast<int>(stage_output_chunk.size()); ++c) {
    if (stage_output_chunk[c])
      max_internal_chunk = std::max(max_internal_chunk, c);
  }
  graph.internal_bytes = max_internal_chunk >= 0 ? static_cast<size_t>(max_internal_chunk + 1) * graph.chunk_size : 0;

  struct ChunkMeta {
    bool is_d2h_dst = false;
    bool is_h2d_src = false;
    int readers = 0;
    bool ring = false;
  };
  ChunkMeta host_meta[MAX_NUMA_NODES][MAX_CHUNKS] = {};

  for (const auto& inst : program.all_instructions) {
    switch (inst.op) {
      case OpCode::D2H:
        host_meta[inst.dst.numa_id][inst.dst.chunk_idx].is_d2h_dst = true;
        break;
      case OpCode::H2D:
        host_meta[inst.src.numa_id][inst.src.chunk_idx].is_h2d_src = true;
        break;
      default:
        break;
    }
  }

  for (int n = 0; n < MAX_NUMA_NODES; ++n) {
    for (int c = 0; c < MAX_CHUNKS; ++c) {
      auto& meta = host_meta[n][c];
      meta.ring = false;
    }
  }

  auto visible_dst_buf = [&](const Instruction& inst) {
    if (options.is_in_place && inst.dst_buf == DeviceBufferKind::OUTPUT &&
        inst.rank >= 0 && inst.rank < world_size &&
        inst.dst.chunk_idx >= 0 && inst.dst.chunk_idx < program.output_chunk_count &&
        stage_output_chunk_by_rank[inst.rank][inst.dst.chunk_idx]) {
      return DeviceBufferKind::INTERNAL;
    }
    return inst.dst_buf;
  };

  auto visible_src_buf = [&](DeviceBufferKind buf, int owner, int chunk) {
    if (options.is_in_place && buf == DeviceBufferKind::OUTPUT &&
        owner >= 0 && owner < world_size &&
        chunk >= 0 && chunk < program.output_chunk_count &&
        stage_output_chunk_by_rank[owner][chunk]) {
      return DeviceBufferKind::INTERNAL;
    }
    return buf;
  };

  std::map<ProgressKey, ProgressKey> latest_by_base;
  std::map<ProgressKey, ProgressKey> final_produced_by_base;

  auto produced_base_for_inst = [&](const Instruction& inst, ProgressKey* out) -> bool {
    switch (inst.op) {
      case OpCode::D2H:
        *out = hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0);
        return true;
      case OpCode::H2D:
        *out = deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0);
        return true;
      case OpCode::H2H:
        *out = hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0);
        return true;
      case OpCode::D2D:
      case OpCode::D2DReduce:
        *out = deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0);
        return true;
      case OpCode::NOP:
        return false;
    }
    return false;
  };

  {
    std::map<ProgressKey, int> produced_versions;
    for (const auto& inst : program.all_instructions) {
      ProgressKey base;
      if (!produced_base_for_inst(inst, &base))
        continue;
      int version = ++produced_versions[base];
      ProgressKey produced = base;
      produced.version = version;
      final_produced_by_base[base] = produced;
    }
  }

  auto latest = [&](ProgressKey base) {
    base.version = 0;
    auto it = latest_by_base.find(base);
    if (it != latest_by_base.end())
      return it->second;
    auto final_it = final_produced_by_base.find(base);
    if (final_it != final_produced_by_base.end())
      return final_it->second;
    latest_by_base.emplace(base, base);
    return base;
  };

  auto latest_existing = [&](ProgressKey base) {
    base.version = 0;
    auto it = latest_by_base.find(base);
    if (it != latest_by_base.end())
      return it->second;
    return base;
  };

  auto produce = [&](ProgressKey key) {
    ProgressKey base = key;
    base.version = 0;
    auto it = latest_by_base.find(base);
    if (it == latest_by_base.end()) {
      key.version = 1;
    } else {
      key.version = it->second.version + 1;
    }
    latest_by_base[base] = key;
    return key;
  };

  for (int c = 0; c < graph.input_chunk_count; ++c) {
    ProgressKey key = deviceKey(DeviceBufferKind::INPUT, rank, c, 0);
    latest_by_base[key] = key;
    graph.initial_ready_inputs.push_back(key);
    addUniqueKey(graph.active_progress_keys, key);
  }

  // Initialize all ranks' input base keys so host-staged cross-rank graph
  // dependencies resolve to version 0 even on ranks that do not own that input.
  for (int r = 0; r < world_size; ++r) {
    int rank_input_count = maxInputChunkCount(program, r);
    for (int c = 0; c < rank_input_count; ++c) {
      ProgressKey key = deviceKey(DeviceBufferKind::INPUT, r, c, 0);
      latest_by_base.emplace(key, key);
    }
  }

  // Pass 1: assign versions globally and instantiate this rank's nodes.
  std::map<ProgressKey, int> reader_counts;
  for (const auto& inst : program.all_instructions) {
    std::vector<ProgressKey> inputs;
    std::vector<int> input_reader_ids;
    ProgressKey output;
    ProgressKey old_output;
    bool has_output = true;
    int src_rank = (inst.src_rank >= 0) ? inst.src_rank : inst.rank;

    switch (inst.op) {
      case OpCode::D2H:
        inputs.push_back(latest(deviceKey(visible_src_buf(inst.src_buf, inst.rank, inst.src.chunk_idx),
                                          inst.rank, inst.src.chunk_idx, 0)));
        old_output = latest_existing(hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0));
        output = produce(hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0));
        break;
      case OpCode::H2D:
        inputs.push_back(latest(hostKey(inst.src.numa_id, inst.src.chunk_idx, 0)));
        old_output = latest_existing(deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0));
        output = produce(deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0));
        break;
      case OpCode::H2H:
        inputs.push_back(latest(hostKey(inst.src.numa_id, inst.src.chunk_idx, 0)));
        old_output = latest_existing(hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0));
        output = produce(hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0));
        break;
      case OpCode::D2D:
        if (src_rank != inst.rank) {
          throw std::runtime_error("Compiler: cross-rank D2D is not supported in the Ascend build");
        }
        inputs.push_back(latest(deviceKey(visible_src_buf(inst.src_buf, src_rank, inst.src.chunk_idx),
                                          src_rank, inst.src.chunk_idx, 0)));
        old_output = latest_existing(deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0));
        output = produce(deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0));
        break;
      case OpCode::D2DReduce: {
        if (src_rank != inst.rank) {
          throw std::runtime_error("Compiler: cross-rank D2DReduce is not supported in the Ascend build");
        }
        auto dst_key = deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0);
        inputs.push_back(latest(deviceKey(visible_src_buf(inst.src_buf, src_rank, inst.src.chunk_idx),
                                          src_rank, inst.src.chunk_idx, 0)));
        inputs.push_back(latest(dst_key));
        old_output = latest_existing(dst_key);
        output = produce(dst_key);
        break;
      }
      case OpCode::NOP:
        has_output = false;
        break;
    }

    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
      const auto& input = inputs[input_idx];
      bool self_rmw_input = inst.op == OpCode::D2DReduce && input_idx == 1 && sameBaseKey(input, output);
      if (self_rmw_input) {
        input_reader_ids.push_back(-1);
        continue;
      }
      int reader_id = reader_counts[input]++;
      if (reader_id >= MAX_PROGRESS_READERS) {
        throw std::runtime_error("Compiler: too many readers for a chunk version");
      }
      input_reader_ids.push_back(reader_id);
      if (input.space == ProgressSpace::HOST) {
        host_meta[input.owner][input.chunk].readers++;
      }
    }

    D2DPlacement d2d_placement = D2DPlacement::Local;

    if (inst.rank != rank || !has_output)
      continue;

    InstructionNode node;
    node.op = inst.op;
    node.source_rank = src_rank;
    node.destination_rank = inst.rank;
    node.d2d_placement = d2d_placement;
    for (const auto& input : inputs) {
      NodeInput ni;
      ni.key = input;
      ni.reader_id = input_reader_ids[node.inputs.size()];
      ni.ring_buffered = input.space == ProgressSpace::HOST && host_meta[input.owner][input.chunk].ring;
      node.inputs.push_back(ni);
      addUniqueKey(graph.active_progress_keys, input);
    }
    node.output = output;
    node.output_ring_buffered = output.space == ProgressSpace::HOST && host_meta[output.owner][output.chunk].ring;
    node.work_bytes = chunkBytesForKey(output, graph);

    if (sameBaseKey(old_output, output) && old_output.version != output.version) {
      node.has_overwrite_guard = true;
      node.overwrite_guard = old_output;
      addUniqueKey(graph.active_progress_keys, old_output);
    }

    switch (inst.op) {
      case OpCode::D2H:
        node.resource_key = "d2h";
        break;
      case OpCode::H2D:
        node.resource_key = "h2d";
        break;
      case OpCode::H2H:
        node.resource_key = "h2h";
        break;
      case OpCode::D2D:
        node.resource_key = "d2d:same";
        break;
      case OpCode::D2DReduce:
        node.resource_key = "d2d_reduce";
        break;
      case OpCode::NOP:
        node.resource_key = "nop";
        break;
    }

    addUniqueKey(graph.active_progress_keys, output);
    graph.nodes.push_back(std::move(node));
  }

  for (const auto& kv : reader_counts) {
    bool ring = kv.first.space == ProgressSpace::HOST && host_meta[kv.first.owner][kv.first.chunk].ring;
    graph.reader_assignment.chunks.push_back(ChunkVersion{kv.first, ring, kv.second});
  }

  // Terminal outputs are this rank's user-visible OUTPUT chunks. In conservative
  // in-place mode, staged INTERNAL results are copied back to OUTPUT and the
  // copyback nodes become terminal.
  for (int c = 0; c < graph.output_chunk_count; ++c) {
    ProgressKey visible = latest(deviceKey(DeviceBufferKind::OUTPUT, rank, c, 0));
    if (stage_output_chunk[c]) {
      ProgressKey staged = latest(deviceKey(DeviceBufferKind::INTERNAL, rank, c, 0));
      ProgressKey old_visible = visible;
      if (options.in_place_alias_mode == InPlaceAliasMode::SameBase &&
          c < graph.input_chunk_count) {
        old_visible = latest(deviceKey(DeviceBufferKind::INPUT, rank, c, 0));
      }
      ProgressKey copyback = produce(deviceKey(DeviceBufferKind::OUTPUT, rank, c, 0));

      InstructionNode node;
      node.op = OpCode::D2D;
      node.source_rank = rank;
      node.inputs.push_back(NodeInput{staged, -1, false});
      node.output = copyback;
      node.has_overwrite_guard = true;
      node.overwrite_guard = old_visible;
      node.resource_key = "d2d:copyback";
      node.terminal = true;
      node.work_bytes = chunkBytesForKey(copyback, graph);
      graph.nodes.push_back(node);
      graph.terminal_outputs.push_back(copyback);
      addUniqueKey(graph.active_progress_keys, staged);
      addUniqueKey(graph.active_progress_keys, old_visible);
      addUniqueKey(graph.active_progress_keys, copyback);
    } else {
      graph.terminal_outputs.push_back(visible);
      addUniqueKey(graph.active_progress_keys, visible);
    }
  }

  for (auto& node : graph.nodes) {
    for (const auto& terminal : graph.terminal_outputs) {
      if (node.output == terminal) {
        node.terminal = true;
        break;
      }
    }
  }

  for (const auto& key : graph.terminal_outputs)
    addUniqueKey(graph.active_progress_keys, key);

  LOG_INFO("Compiler: Compiled graph for rank %d: %zu nodes, %zu terminal outputs, chunk_size=%zu, subchunks=%zu",
           rank, graph.nodes.size(), graph.terminal_outputs.size(), graph.chunk_size, graph.num_subchunks);
  return graph;
}

RuntimeBinding Compiler::makeRuntimeBinding(const CompiledGraph& graph, const CompileOptions& options,
                                            void* sendbuff, void* recvbuff, HostMemoryPool* host_pool) {
  RuntimeBinding binding;
  binding.scalar_type = options.scalar_type;
  binding.reduce_op = options.reduce_op;
  binding.element_size = options.element_size;

  for (int i = 0; i < ProgressSpaceCount; ++i)
    binding.layout[i] = graph.runtime_layout[i];

  for (int numa = 0; numa < MAX_NUMA_NODES; ++numa)
    binding.host_chunk_remap[numa] =
        compiler_detail::BuildDenseHostChunkMap(graph.global_host_progress_keys, numa);

  if (host_pool) {
    for (int numa = 0; numa < MAX_NUMA_NODES; ++numa) {
      const auto& remap = binding.host_chunk_remap[numa];
      size_t used_chunks = static_cast<size_t>(std::count_if(remap.begin(), remap.end(), [](int slot) {
        return slot >= 0;
      }));
      if (used_chunks == 0)
        continue;
      size_t required = used_chunks * graph.chunk_size;
      size_t allocated = host_pool->getAllocatedSize(numa);
      LOG_DEBUG("Compiler: host staging map comm=%d rank=%d numa=%d used_chunks=%zu chunk_size=%zu "
                "required=%zu allocated=%zu",
                options.comm_id,
                graph.rank,
                numa,
                used_chunks,
                graph.chunk_size,
                required,
                allocated);
      if (required > allocated) {
        LOG_ERROR("Compiler: host staging buffer too small numa=%d required=%zu allocated=%zu chunk_size=%zu "
                  "used_chunks=%zu",
                  numa,
                  required,
                  allocated,
                  graph.chunk_size,
                  used_chunks);
        throw std::runtime_error("Compiler: host staging buffer too small for graph binding");
      }
      binding.host_base[numa] = static_cast<char*>(host_pool->getChunkBuffer(numa, 0, graph.chunk_size));
      binding.host_capacity[numa] = allocated;
    }
  }

  char* send_base = static_cast<char*>(sendbuff);
  char* recv_base = static_cast<char*>(recvbuff);
  char* internal_base = nullptr;
  if (graph.internal_bytes > 0) {
    if (!options.process_state || options.process_state->ensureInternalBuffer(
                                      options.comm_hash, options.comm_id, graph.rank,
                                      options.local_device_id, graph.internal_bytes) != 0) {
      throw std::runtime_error("Compiler: failed to allocate internal device buffer for graph binding");
    }
    internal_base = static_cast<char*>(options.process_state->getInternalBufferForComm(
        options.comm_hash, options.comm_id, graph.rank, options.local_device_id));
  } else if (options.process_state) {
    internal_base = static_cast<char*>(options.process_state->getInternalBufferForComm(
        options.comm_hash, options.comm_id, graph.rank, options.local_device_id));
  }

  for (int r = 0; r < graph.world_size; ++r) {
    if (r == graph.rank || !options.process_state) {
      binding.device_base[static_cast<int>(ProgressSpace::INPUT)][r] = send_base;
      binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r] = recv_base;
      binding.device_base[static_cast<int>(ProgressSpace::INTERNAL)][r] = internal_base;
    } else {
      binding.device_base[static_cast<int>(ProgressSpace::INPUT)][r] = nullptr;
      binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r] = nullptr;
      binding.device_base[static_cast<int>(ProgressSpace::INTERNAL)][r] = nullptr;
    }
    binding.device_base[static_cast<int>(ProgressSpace::SCRATCH)][r] =
        binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r];
  }

  return binding;
}

}  // namespace pccl
