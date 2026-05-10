#include "compiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <thread>

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

static bool enableRemoteD2DPush(const CompileOptions& options) {
  (void)options;
  const char* env = std::getenv("PCCL_D2D_PUSH");
  return !env || std::atoi(env) != 0;
}

static bool envOverridesD2DInputFirst(bool* enabled) {
  const char* env = std::getenv("PCCL_D2D_INPUT_FIRST");
  if (!env)
    return false;
  *enabled = std::atoi(env) != 0;
  return true;
}

static int nodeScheduleClass(const InstructionNode& node) {
  if (node.op == OpCode::D2H)
    return 0;
  if (node.op == OpCode::D2D && !node.inputs.empty() && node.inputs[0].key.space == ProgressSpace::INPUT)
    return 1;
  if (node.op == OpCode::D2D && node.d2d_placement == D2DPlacement::Local)
    return 2;
  if (node.op == OpCode::H2D)
    return 3;
  if (node.op == OpCode::D2DReduce)
    return 4;
  if (node.op == OpCode::D2D)
    return 5;
  return 6;
}

CompiledGraph Compiler::compileGraph(const IRProgram& program, int rank, int world_size,
                                     const GraphCompileOptions& graph_options) {
  const auto& options = graph_options.legacy_layout;
  const bool remote_d2d_push = enableRemoteD2DPush(options);

  CompiledGraph graph;
  graph.rank = rank;
  graph.world_size = world_size;
  graph.input_chunk_count = maxInputChunkCount(program, rank);
  graph.output_chunk_count = program.output_chunk_count;
  graph.element_size = options.element_size;
  graph.scalar_type = options.scalar_type;
  graph.reduce_op = options.reduce_op;
  graph.is_in_place = options.is_in_place;
  for (const auto& inst : program.all_instructions) {
    int src_rank = (inst.src_rank >= 0) ? inst.src_rank : inst.rank;
    if (inst.op == OpCode::D2D && src_rank != inst.rank) {
      graph.has_remote_d2d = true;
      break;
    }
  }

  size_t segment_size = options.seg_count * options.element_size;
  size_t stride_offset = options.stride * options.element_size;
  size_t pcie_per_seg = segment_size - stride_offset;
  size_t active_elems = options.seg_count - options.stride;
  int d2h_cps = std::max(1, program.input_chunk_count / std::max(1, options.d2h_segments));
  int h2d_cps = std::max(1, program.output_chunk_count / std::max(1, options.h2d_segments));
  graph.chunk_size = std::max(maxChunkBytes(active_elems, options.element_size, d2h_cps),
                              maxChunkBytes(active_elems, options.element_size, h2d_cps));
  graph.total_bytes = pcie_per_seg * std::max(options.d2h_segments, options.h2d_segments);
  const size_t slice = pcclSubchunkSliceFor(graph.chunk_size);
  graph.num_subchunks = (graph.chunk_size + slice - 1) / slice;
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
    bool is_h2h_dst = false;
    int d2h_writers = 0;
    int readers = 0;
    bool ring = false;
  };
  ChunkMeta host_meta[MAX_NUMA_NODES][MAX_CHUNKS] = {};

  for (const auto& inst : program.all_instructions) {
    switch (inst.op) {
      case OpCode::D2H:
        host_meta[inst.dst.numa_id][inst.dst.chunk_idx].is_d2h_dst = true;
        host_meta[inst.dst.numa_id][inst.dst.chunk_idx].d2h_writers++;
        break;
      case OpCode::H2D:
        host_meta[inst.src.numa_id][inst.src.chunk_idx].is_h2d_src = true;
        break;
      case OpCode::H2H:
      case OpCode::H2HReduce:
        host_meta[inst.dst.numa_id][inst.dst.chunk_idx].is_h2h_dst = true;
        break;
      default:
        break;
    }
  }

  // Staging rings: a host chunk written by exactly one D2H and only read by H2D
  // can reuse a few slice-sized slots, as long as the graph has more slices than
  // slots and the slots fit in the chunk. The executor holds a writer back until
  // every reader has drained the slot it is about to overwrite.
  const size_t ring_slots = pcclRingSlotsFor(graph.chunk_size);
  const bool staging_ring = pcclHostRingSlots() >= 2 && graph.num_subchunks > ring_slots &&
                            ring_slots * pcclSubchunkSliceFor(graph.chunk_size) <= graph.chunk_size;
  for (int n = 0; n < MAX_NUMA_NODES; ++n) {
    for (int c = 0; c < MAX_CHUNKS; ++c) {
      auto& meta = host_meta[n][c];
      meta.ring = meta.is_h2h_dst && !meta.is_d2h_dst && !meta.is_h2d_src;
      if (staging_ring && meta.is_d2h_dst && meta.d2h_writers == 1 && meta.is_h2d_src && !meta.is_h2h_dst)
        meta.ring = true;
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
      case OpCode::H2HReduce:
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

  // Initialize all ranks' input base keys so cross-rank D2D dependencies
  // resolve to version 0 even on ranks that do not own that input.
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
      case OpCode::H2HReduce:
        inputs.push_back(latest(hostKey(inst.src.numa_id, inst.src.chunk_idx, 0)));
        inputs.push_back(latest(hostKey(inst.src2.numa_id, inst.src2.chunk_idx, 0)));
        old_output = latest_existing(hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0));
        output = produce(hostKey(inst.dst.numa_id, inst.dst.chunk_idx, 0));
        break;
      case OpCode::D2D:
        inputs.push_back(latest(deviceKey(visible_src_buf(inst.src_buf, src_rank, inst.src.chunk_idx),
                                          src_rank, inst.src.chunk_idx, 0)));
        old_output = latest_existing(deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0));
        output = produce(deviceKey(visible_dst_buf(inst), inst.rank, inst.dst.chunk_idx, 0));
        break;
      case OpCode::D2DReduce: {
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

    int execution_rank = inst.rank;
    D2DPlacement d2d_placement = D2DPlacement::Local;
    if (inst.op == OpCode::D2D) {
      if (src_rank == inst.rank) {
        d2d_placement = D2DPlacement::Local;
      } else if (remote_d2d_push) {
        execution_rank = src_rank;
        d2d_placement = D2DPlacement::RemoteWrite;
      } else {
        d2d_placement = D2DPlacement::RemoteRead;
      }
    } else if (inst.op == OpCode::D2DReduce) {
      if (src_rank != inst.rank || inst.src_buf == DeviceBufferKind::INPUT) {
        d2d_placement = D2DPlacement::Local;
      }
    }

    if (execution_rank != rank || !has_output)
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
      case OpCode::D2D:
        graph.has_remote_d2d = graph.has_remote_d2d || (src_rank != rank) || (inst.rank != rank);
        if (d2d_placement == D2DPlacement::Local) {
          node.resource_key = "d2d:same";
        } else if (d2d_placement == D2DPlacement::RemoteWrite) {
          node.resource_key = "d2d:push:" + std::to_string(src_rank) + "->" + std::to_string(inst.rank);
        } else {
          node.resource_key = "d2d:pull:" + std::to_string(src_rank) + "->" + std::to_string(inst.rank);
        }
        break;
      case OpCode::D2DReduce:
        node.resource_key = "d2d_reduce";
        break;
      case OpCode::H2H:
      case OpCode::H2HReduce:
        node.resource_key = "h2h:numa" + std::to_string(inst.dst.numa_id);
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

  bool prioritize_input_d2d = graph_options.prioritize_input_d2d;
  envOverridesD2DInputFirst(&prioritize_input_d2d);
  if (prioritize_input_d2d && graph.has_remote_d2d) {
    std::stable_sort(graph.nodes.begin(), graph.nodes.end(), [](const InstructionNode& a, const InstructionNode& b) {
      return nodeScheduleClass(a) < nodeScheduleClass(b);
    });
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

  for (const auto& key : graph.active_progress_keys) {
    if (host_pool && key.space == ProgressSpace::HOST && key.owner >= 0 && key.owner < MAX_NUMA_NODES &&
        !binding.host_base[key.owner]) {
      binding.host_base[key.owner] = static_cast<char*>(host_pool->getChunkBuffer(key.owner, 0, graph.chunk_size));
    }
  }

  char* send_base = static_cast<char*>(sendbuff);
  char* recv_base = static_cast<char*>(recvbuff);
  char* internal_base = nullptr;
  if (graph.internal_bytes > 0) {
    if (!options.process_state || options.process_state->ensureInternalBuffer(graph.rank, options.local_device_id,
                                                                              graph.internal_bytes) != 0) {
      throw std::runtime_error("Compiler: failed to allocate internal device buffer for graph binding");
    }
    internal_base = static_cast<char*>(options.process_state->getInternalBufferForRank(graph.rank));
  } else if (options.process_state) {
    internal_base = static_cast<char*>(options.process_state->getInternalBufferForRank(graph.rank));
  }

  auto wait_for_remote = [](const char* label, int owner, auto&& getter) -> char* {
    char* base = nullptr;
    for (int attempt = 0; attempt < 10000 && !base; ++attempt) {
      base = static_cast<char*>(getter());
      if (!base)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (!base) {
      throw std::runtime_error(std::string("Compiler: timeout waiting for remote device buffer ") + label +
                               " owner=" + std::to_string(owner));
    }
    return base;
  };

  auto device_ipc_kind_for_space = [](ProgressSpace space) -> int {
    switch (space) {
      case ProgressSpace::INPUT:
        return 0;
      case ProgressSpace::OUTPUT:
      case ProgressSpace::SCRATCH:
        return 1;
      case ProgressSpace::INTERNAL:
        return 2;
      case ProgressSpace::HOST:
        break;
    }
    return -1;
  };

  auto remote_device_base = [&](ProgressSpace space, int owner, auto&& fallback) -> char* {
    if (options.process_state && options.process_state->isMultiProcess()) {
      int kind = device_ipc_kind_for_space(space);
      if (kind >= 0) {
        return static_cast<char*>(options.process_state->getRemoteDeviceBuffer(
            options.comm_id, options.local_rank >= 0 ? options.local_rank : graph.rank,
            options.local_device_id, kind, owner));
      }
    }
    return static_cast<char*>(fallback());
  };

  bool needs_device_base[ProgressSpaceCount][MAX_RANKS] = {};
  for (const auto& key : graph.active_progress_keys) {
    if (key.space != ProgressSpace::HOST && key.owner >= 0 && key.owner < MAX_RANKS)
      needs_device_base[static_cast<int>(key.space)][key.owner] = true;
  }

  for (int r = 0; r < graph.world_size; ++r) {
    if (r == graph.rank || !options.process_state) {
      binding.device_base[static_cast<int>(ProgressSpace::INPUT)][r] = send_base;
      binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r] = recv_base;
      binding.device_base[static_cast<int>(ProgressSpace::INTERNAL)][r] = internal_base;
    } else {
      ProcessState* ps = options.process_state;
      if (needs_device_base[static_cast<int>(ProgressSpace::INPUT)][r]) {
        binding.device_base[static_cast<int>(ProgressSpace::INPUT)][r] =
            wait_for_remote("INPUT", r, [&]() -> void* {
              return remote_device_base(ProgressSpace::INPUT, r,
                                        [&]() -> void* { return const_cast<void*>(ps->getInputBufferForRank(r)); });
            });
      } else {
        binding.device_base[static_cast<int>(ProgressSpace::INPUT)][r] =
            static_cast<char*>(const_cast<void*>(ps->getInputBufferForRank(r)));
      }
      if (needs_device_base[static_cast<int>(ProgressSpace::OUTPUT)][r]) {
        binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r] =
            wait_for_remote("OUTPUT", r, [&]() -> void* {
              return remote_device_base(ProgressSpace::OUTPUT, r,
                                        [&]() -> void* { return ps->getOutputBufferForRank(r); });
            });
      } else {
        binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r] =
            static_cast<char*>(ps->getOutputBufferForRank(r));
      }
      if (needs_device_base[static_cast<int>(ProgressSpace::INTERNAL)][r]) {
        binding.device_base[static_cast<int>(ProgressSpace::INTERNAL)][r] =
            wait_for_remote("INTERNAL", r, [&]() -> void* {
              return remote_device_base(ProgressSpace::INTERNAL, r,
                                        [&]() -> void* { return ps->getInternalBufferForRank(r); });
            });
      } else {
        binding.device_base[static_cast<int>(ProgressSpace::INTERNAL)][r] =
            static_cast<char*>(ps->getInternalBufferForRank(r));
      }
    }
    binding.device_base[static_cast<int>(ProgressSpace::SCRATCH)][r] =
        binding.device_base[static_cast<int>(ProgressSpace::OUTPUT)][r];
  }

  return binding;
}

}  // namespace pccl
