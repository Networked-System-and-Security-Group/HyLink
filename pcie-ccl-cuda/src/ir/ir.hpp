#ifndef PCCL_IR_HPP
#define PCCL_IR_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
// Subchunk slice size for a graph with the given chunk size. Every user (compiler,
// executor, H2H engine) derives it from graph.chunk_size so they always agree.
// Default: chunk/8 rounded down to a power of two, clamped to [512KB, 4MB]
// (chunk/64 up to 16MB for chunks >= 512MB, see below);
// large chunks issue fewer, bigger DMAs while small chunks still get >= 2
// pipelined slices. The env var PCCL_SUBCHUNK_SLICE_SIZE=<bytes> forces a fixed
// value. Ring-buffered host chunks keep 2 * slice <= chunk_size whenever they have
// more than two slices, so DOUBLE_BUFFER_SLOTS stays valid.
inline size_t pcclSubchunkSliceFor(size_t chunk_size) {
  static const size_t forced = []() -> size_t {
    const char* e = std::getenv("PCCL_SUBCHUNK_SLICE_SIZE");
    if (!e || e[0] == '\0') return 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(e, &end, 10);
    if (end == e || parsed < 4096) return 0;
    return static_cast<size_t>(parsed);
  }();
  if (forced) return forced;
  const size_t lo = static_cast<size_t>(512) << 10;
  const size_t hi = static_cast<size_t>(4) << 20;
  size_t target = chunk_size / 8;
  size_t s = lo;
  while (s * 2 <= target && s * 2 <= hi) s *= 2;
  // Very large chunks (>= 512MB): grow the slice to chunk/64, up to 16MB. With
  // 4MB slices the D2H->H2D staging ring is only slots * 4MB deep, and on GPUs
  // that share a PCIe switch uplink, 2-4GB collectives lose 5-8% to producer/
  // consumer stalls (4x V100, pairwise PEX 8747: AG 4G 7.8-8.05 -> 8.5 GB/s).
  const size_t big_hi = static_cast<size_t>(16) << 20;
  const size_t big_target = chunk_size / 64;
  if (chunk_size >= (static_cast<size_t>(512) << 20)) {
    while (s * 2 <= big_target && s * 2 <= big_hi) s *= 2;
  }
  return s;
}
static constexpr size_t MAX_SLICES_PER_CHUNK = 1024;        // Supports up to 1GB chunks

// Double-buffer L3 residency: ring-eligible chunks use 2 subchunk-sized physical
// slots (ping-pong); see pcclRingSlotsFor() for the slot count.  Working set per chunk
// is 4 MB, fitting comfortably in Kunpeng 920 L3 (~24 MB/CCL).
constexpr int DOUBLE_BUFFER_SLOTS = 2;

// Number of physical slots for a ring-buffered host chunk. H2H destination
// chunks always ring with DOUBLE_BUFFER_SLOTS. With PCCL_HOST_RING_SLOTS=<n>
// (n >= 2), D2H->H2D staging chunks also ring through n slice-sized slots
// instead of occupying the whole chunk: host DMA bandwidth drops sharply as the
// pinned working set grows (IOMMU/IOTLB reach), e.g. 9.2 GB/s at 32MB vs 7.3
// GB/s at 2GB per GPU on EPYC + RTX 3090. Default 3; 0 disables staging rings.
inline int pcclHostRingSlots() {
  static const int value = []() -> int {
    const char* e = std::getenv("PCCL_HOST_RING_SLOTS");
    if (!e || e[0] == '\0') return 3;
    int v = std::atoi(e);
    return v >= 2 ? v : 0;
  }();
  return value;
}

// Slot count used by every ring-buffered host chunk of a graph.
inline size_t pcclRingSlotsFor(size_t chunk_size) {
  (void)chunk_size;
  size_t slots = static_cast<size_t>(pcclHostRingSlots());
  return slots >= 2 ? slots : DOUBLE_BUFFER_SLOTS;
}

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
  H2H,     // Host to Host copy: dst = src
  H2HReduce,  // Three-address host reduce: dst = src1 + src2
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
  ChunkAddress src;            // H2H/D2D/D2DReduce: source. H2HReduce: src1
  ChunkAddress src2{-1, -1};   // H2HReduce only: second source
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
