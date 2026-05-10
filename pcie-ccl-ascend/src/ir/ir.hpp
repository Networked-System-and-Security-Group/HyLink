#ifndef PCCL_IR_HPP
#define PCCL_IR_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pccl {

constexpr int MAX_CHUNKS = 128;
constexpr int MAX_RANKS = 8;
constexpr int MAX_NUMA_NODES = 8;
constexpr int MAX_PROGRESS_VERSIONS = 16;
constexpr int MAX_PROGRESS_READERS = 32;

#ifndef PCCL_SUBCHUNK_SLICE_SIZE
#define PCCL_SUBCHUNK_SLICE_SIZE (1 * 1024 * 1024)
#endif
static constexpr size_t SUBCHUNK_SLICE_SIZE = PCCL_SUBCHUNK_SLICE_SIZE;
static constexpr size_t MAX_SLICES_PER_CHUNK = 1024;        // Supports up to 1GB chunks

// Double-buffer L3 residency: ring-eligible chunks use 2 subchunk-sized physical
// slots (ping-pong).  Subchunk sc_k maps to slot sc_k & 1.  Working set per chunk
// is 4 MB, fitting comfortably in Kunpeng 920 L3 (~24 MB/CCL).
constexpr int DOUBLE_BUFFER_SLOTS = 2;

#ifndef PCCL_EVENT_POLL_INTERVAL_US
#define PCCL_EVENT_POLL_INTERVAL_US 10
#endif

#ifndef PCCL_TRANSFER_STREAM_COUNT
#define PCCL_TRANSFER_STREAM_COUNT 1
#endif

#ifndef PCCL_D2H_LEAD_WINDOW
#define PCCL_D2H_LEAD_WINDOW 4   // D2H may not lead H2D by more than this many subchunks
#endif

enum class OpCode : uint8_t {
  NOP = 0,
  D2D,     // Device internal copy
  D2H,     // Device to Host
  H2D,     // Host to Device
  H2H,     // Host to Host copy
  D2DReduce,  // Device reduce: dst += src (src and dst are local to this device)
};

enum class DeviceBufferKind : uint8_t {
  INPUT = 0,
  OUTPUT,
  SCRATCH,
  INTERNAL,
};

enum class ProgressSpace : uint8_t {
  HOST = 0,
  INPUT,
  OUTPUT,
  SCRATCH,
  INTERNAL,
};

constexpr int ProgressSpaceCount = 5;
constexpr int DeviceProgressSpaceCount = 4;

struct ProgressKey {
  ProgressSpace space = ProgressSpace::HOST;
  int owner = 0;    // HOST: NUMA id; device spaces: rank/device id
  int chunk = 0;
  int version = 0;

  bool operator==(const ProgressKey& o) const {
    return space == o.space && owner == o.owner && chunk == o.chunk && version == o.version;
  }

  bool operator!=(const ProgressKey& o) const { return !(*this == o); }

  bool operator<(const ProgressKey& o) const {
    if (space != o.space) return static_cast<int>(space) < static_cast<int>(o.space);
    if (owner != o.owner) return owner < o.owner;
    if (chunk != o.chunk) return chunk < o.chunk;
    return version < o.version;
  }
};

inline const char* deviceBufferKindName(DeviceBufferKind kind) {
  switch (kind) {
    case DeviceBufferKind::INPUT:
      return "input";
    case DeviceBufferKind::OUTPUT:
      return "output";
    case DeviceBufferKind::SCRATCH:
      return "scratch";
    case DeviceBufferKind::INTERNAL:
      return "internal";
  }
  return "unknown";
}

inline ProgressSpace progressSpaceForDeviceBuffer(DeviceBufferKind kind) {
  switch (kind) {
    case DeviceBufferKind::INPUT:
      return ProgressSpace::INPUT;
    case DeviceBufferKind::OUTPUT:
      return ProgressSpace::OUTPUT;
    case DeviceBufferKind::SCRATCH:
      return ProgressSpace::SCRATCH;
    case DeviceBufferKind::INTERNAL:
      return ProgressSpace::INTERNAL;
  }
  return ProgressSpace::OUTPUT;
}

inline DeviceBufferKind deviceBufferKindForProgressSpace(ProgressSpace space) {
  switch (space) {
    case ProgressSpace::INPUT:
      return DeviceBufferKind::INPUT;
    case ProgressSpace::OUTPUT:
      return DeviceBufferKind::OUTPUT;
    case ProgressSpace::SCRATCH:
      return DeviceBufferKind::SCRATCH;
    case ProgressSpace::INTERNAL:
      return DeviceBufferKind::INTERNAL;
    case ProgressSpace::HOST:
      break;
  }
  return DeviceBufferKind::OUTPUT;
}

inline bool isDeviceProgressSpace(ProgressSpace space) {
  return space != ProgressSpace::HOST;
}

inline int deviceProgressSpaceIndex(ProgressSpace space) {
  switch (space) {
    case ProgressSpace::INPUT:
      return 0;
    case ProgressSpace::OUTPUT:
      return 1;
    case ProgressSpace::SCRATCH:
      return 2;
    case ProgressSpace::INTERNAL:
      return 3;
    case ProgressSpace::HOST:
      break;
  }
  return 0;
}

struct ChunkAddress {
  int numa_id;
  int chunk_idx;
};

struct Instruction {
  OpCode op;
  int rank = -1;                           // rank that executes this instruction
  int src_rank = -1;                       // D2D only. -1 means current rank.
  DeviceBufferKind src_buf = DeviceBufferKind::INPUT;
  DeviceBufferKind dst_buf = DeviceBufferKind::OUTPUT;
  ChunkAddress src;
  ChunkAddress src2{-1, -1};
  ChunkAddress dst;
};

struct IRProgram {
  std::string name;
  int nranks;
  int input_chunk_count;
  int output_chunk_count;
  std::vector<Instruction> instructions;           // this rank only
  std::vector<Instruction> all_instructions;       // all ranks (for dependency analysis)
};

}  // namespace pccl

#endif /* PCCL_IR_HPP */
