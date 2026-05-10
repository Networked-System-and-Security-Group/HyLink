#include <vector>
#include <algorithm>

#include "ir/compiler.hpp"
#include "ir/compiled_graph.hpp"

namespace {

bool expect(bool condition) {
  return condition;
}

}  // namespace

int main() {
  using pccl::ProgressKey;
  using pccl::ProgressSpace;

  std::vector<ProgressKey> keys = {
      {ProgressSpace::HOST, 0, 56, 0},
      {ProgressSpace::HOST, 0, 1, 0},
      {ProgressSpace::HOST, 0, 16, 0},
      {ProgressSpace::HOST, 0, 8, 0},
      {ProgressSpace::HOST, 0, 16, 1},
      {ProgressSpace::HOST, 1, 3, 0},
      {ProgressSpace::INPUT, 0, 99, 0},
  };

  const auto numa0 = pccl::compiler_detail::BuildDenseHostChunkMap(keys, 0);
  if (!expect(numa0.size() == pccl::MAX_CHUNKS) ||
      !expect(numa0[1] == 0) ||
      !expect(numa0[8] == 1) ||
      !expect(numa0[16] == 2) ||
      !expect(numa0[56] == 3) ||
      !expect(numa0[3] == -1)) {
    return 1;
  }
  if (std::count_if(numa0.begin(), numa0.end(), [](int slot) { return slot >= 0; }) != 4)
    return 3;

  const auto numa1 = pccl::compiler_detail::BuildDenseHostChunkMap(keys, 1);
  if (!expect(numa1[3] == 0) || !expect(numa1[1] == -1) || !expect(numa1[56] == -1) ||
      std::count_if(numa1.begin(), numa1.end(), [](int slot) { return slot >= 0; }) != 1)
    return 2;

  pccl::IRProgram program;
  program.nranks = 2;
  program.input_chunk_count = 1;
  program.output_chunk_count = 1;
  program.all_instructions = {
      {pccl::OpCode::D2H, 0, -1, pccl::DeviceBufferKind::INPUT, pccl::DeviceBufferKind::OUTPUT,
       {0, 1}, {-1, -1}, {0, 8}},
      {pccl::OpCode::H2D, 1, -1, pccl::DeviceBufferKind::INPUT, pccl::DeviceBufferKind::OUTPUT,
       {0, 56}, {-1, -1}, {0, 2}},
  };
  pccl::GraphCompileOptions options;
  options.legacy_layout.seg_count = 128;
  options.legacy_layout.element_size = sizeof(float);
  options.legacy_layout.d2h_segments = 1;
  options.legacy_layout.h2d_segments = 1;
  const auto graph0 = pccl::Compiler::compileGraph(program, 0, 2, options);
  const auto graph1 = pccl::Compiler::compileGraph(program, 1, 2, options);
  const auto global0 = pccl::compiler_detail::BuildDenseHostChunkMap(graph0.global_host_progress_keys, 0);
  const auto global1 = pccl::compiler_detail::BuildDenseHostChunkMap(graph1.global_host_progress_keys, 0);
  if (!expect(global0[8] == 0) || !expect(global0[56] == 1) ||
      !expect(global1[8] == global0[8]) ||
      !expect(global1[56] == global0[56])) {
    return 4;
  }

  return 0;
}
