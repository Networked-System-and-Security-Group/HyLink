#ifndef IR_HPP
#define IR_HPP
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pccl {

#define MAX_CHUNKS 20
#define MAX_RANKS 8

static constexpr size_t SUBCHUNK_SLICE_SIZE = 1024 * 1024;  // 1MB per subchunk
static constexpr size_t MAX_SLICES_PER_CHUNK = 1024;        // Supports up to 1GB chunks

// OpCode definitions for IR instructions
// Each OpCode specifies the data movement direction:
//   - D2D: Device internal copy, sendbuff[src_chunk] -> recvbuff[dst_chunk]
//   - D2H: Device to Host, sendbuff[src_chunk] -> local_numa[dst_chunk]
//   - H2D: Host to Device, local_numa[src_chunk] -> recvbuff[dst_chunk]
//   - H2H: Host to Host, numa[src_numa][src_chunk] -> local_numa[dst_chunk]
//   - H2H_REDUCE: Host reduce operation (future)
enum class OpCode : uint8_t {
  NOP = 0,
  D2D,         // Device internal copy: sendbuff[src_chunk] -> recvbuff[dst_chunk]
  D2H,         // Device to Host: sendbuff[src_chunk] -> local_numa[dst_chunk]
  H2D,         // Host to Device: local_numa[src_chunk] -> recvbuff[dst_chunk]
  H2H,         // Host to Host: numa[src_numa][src_chunk] -> local_numa[dst_chunk]
  H2H_REDUCE,  // Host reduce operation (future)
};

// DataDependency: Specifies a dependency on a chunk version in a NUMA node
// An instruction can only execute when all its dependencies are satisfied
struct DataDependency {
  int numa_node;          // NUMA node of the dependency
  int chunk_idx;          // Chunk index within the NUMA node
  uint64_t required_ver;  // Minimum version required for this chunk
};

// OutputEffect: Specifies which chunk is modified by an instruction
// Output is always to the local NUMA node, so numa_node is not needed
struct OutputEffect {
  int chunk_idx;           // Chunk index in local NUMA that is modified
  uint64_t version_delta;  // Version increment this instruction contributes (default = 1)
};

// Instruction: A single IR instruction for data movement
// Field usage depends on OpCode:
//   | OpCode | src_numa | src_chunk_idx    | dst_chunk_idx    |
//   |--------|----------|------------------|------------------|
//   | D2D    | ignored  | sendbuff offset  | recvbuff offset  |
//   | D2H    | ignored  | sendbuff offset  | local NUMA chunk |
//   | H2D    | ignored  | local NUMA chunk | recvbuff offset  |
//   | H2H    | src NUMA | src NUMA chunk   | local NUMA chunk |
struct Instruction {
  OpCode op;

  int src_numa;       // Source NUMA node (only used by H2H)
  int src_chunk_idx;  // Source chunk index
  int dst_chunk_idx;  // Destination chunk index

  std::vector<DataDependency> deps;   // Dependencies to check before execution
  std::vector<OutputEffect> effects;  // Effects after execution (version updates)
};

struct IRProgram {
  int input_chunk_count;                  // Number of input chunks
  int output_chunk_count;                 // Number of output chunks
  std::vector<Instruction> instructions;  // IR instruction sequence
};

}  // namespace pccl

#endif /* IR_HPP */
