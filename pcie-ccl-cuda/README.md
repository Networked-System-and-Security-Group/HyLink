English | [中文](README_zh.md)

# PCCL — PCIe Collective Communication Library

PCCL is a collective communication library optimized for PCIe interconnects, providing high-performance collective operations for multi-GPU, multi-NUMA-node topologies.

Key features:

- **IR programming model** — Data movement patterns are defined declaratively through an XML intermediate representation, decoupling algorithm design from execution
- **NUMA-aware execution** — Automatically discovers the topology and allocates host memory and worker threads per NUMA node to minimize cross-NUMA traffic
- **Single-process / multi-process modes** — Supports both single-process multi-threaded and multi-process (one GPU per process) deployment, switching automatically based on `pcclUniqueId`
- **Multi-backend support** — Supports both NVIDIA CUDA and Huawei Ascend CANN through a HAL abstraction layer
- **Vendor CCL compatible APIs** — Provides NCCL / HCCL compatible interfaces that can serve as drop-in replacements
- **Strided collectives** — Supports MPCCL multi-path splitting scenarios (e.g., mixed NVLink + PCIe paths)

## System Architecture

PCCL uses a four-layer architecture:

- **API layer** (`include/pccl/`, `src/api/`) — Public interfaces, including the native PCCL C API and vendor CCL compatible APIs (NCCL / HCCL), providing collective operations and communicator management
- **Orchestration layer** (`src/ir/`, `src/topology/`) — Loads XML IR programs and, combined with topology information (NUMA discovery, GPU-NUMA mapping), compiles them into an address-independent `CompiledGraph`; real device/host addresses are bound at runtime
- **Execution layer** (`src/executor/`, `src/memory/`) — Each rank uses one `GraphExecutor` that polls graph nodes and drives asynchronous D2H / H2D / D2D CUDA lanes; H2H is still executed by a resident NUMA worker pool as the backend. Versioned progress and subchunk-level progress counters provide pipelining, reader consumption cursors, and overwrite protection. This layer also manages the host memory pool (`devMallocHost` in single-process mode, shared hugepages + `devHostRegister` in multi-process mode), cross-rank shared state (heap memory in single-process mode, POSIX shared memory in multi-process mode), and GPU-visible completion flags. The CUDA backend uses stream memory wait/write value operations to make the user stream wait for collective completion, avoiding launching SM-occupying spin kernels on the wait path
- **HAL layer** (`src/hal/`) — Device runtime abstraction that adapts to CUDA and Ascend CANN via tag dispatch

Execution flow: API call → IR loading → compilation to `CompiledGraph` → runtime address binding → `GraphExecutor` polls and submits ready nodes → terminal outputs complete → stream synchronization

## Multi-Process Mode

PCCL supports two deployment modes:

- **Single-process mode (SP)** — One process manages all GPUs and submits collective operations in parallel from multiple threads. Passing an all-zero `pcclUniqueId` selects this mode automatically. Host memory is allocated with `devMallocHost`, one block per NUMA node.
- **Multi-process mode (MP)** — Each process manages one GPU, coordinating via shared memory. Rank 0 calls `pcclGetUniqueId` to generate a unique ID and broadcasts it to all ranks; PCCL uses this ID to set up cross-process sharing.

### Multi-Process Shared Memory Architecture

Multi-process mode uses two layers of shared memory:

| Component | Purpose | Implementation |
|------|------|------|
| **SharedState** | Cross-rank synchronization (progress counters, barriers, chunk state) | POSIX `shm_open` + `mmap` |
| **HostMemoryPool** | Host DMA buffers shared across processes | `memfd_create` (MFD_HUGETLB) + `devHostRegister` |

Bootstrap flow:
1. The creator (the process hosting rank 0) allocates hugepage memory via `memfd_create` and shares the fd through a bootstrap shm
2. Joiners (other processes) open the memfd via `/proc/<creator_pid>/fd/<N>` and `mmap` it into their own address space
3. All processes call `devHostRegister` on the shared buffer to register it as DMA-accessible
4. A barrier ensures that all processes have finished initialization before communication begins

### Hugepage Configuration

Multi-process mode uses 2MB hugepages as shared DMA buffers. Enough hugepages must be reserved on the system:

```bash
# Check current hugepage status
cat /proc/meminfo | grep Huge

# Reserve hugepages (each NUMA node needs PCCL_HOST_BUFFER_SIZE, 4GB by default)
# e.g., 4 NUMA nodes × 4GB = 16GB = 8192 2MB hugepages
echo 8192 > /proc/sys/vm/nr_hugepages

# Or reserve per NUMA node (recommended, avoids cross-NUMA allocation)
echo 2048 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 2048 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
echo 2048 > /sys/devices/system/node/node2/hugepages/hugepages-2048kB/nr_hugepages
echo 2048 > /sys/devices/system/node/node3/hugepages/hugepages-2048kB/nr_hugepages

# To persist (takes effect after reboot), add to /etc/sysctl.conf:
vm.nr_hugepages = 8192
```

If there are not enough hugepages, `memfd_create` fails and multi-process initialization exits with an error.

## Building

### Prerequisites

- CMake >= 3.18
- C++14 compiler
- NVIDIA CUDA Toolkit or Ascend CANN (auto-detected, or can be specified manually)
- libnuma (required; the foundation for NUMA-aware memory allocation and thread binding)
- librt (POSIX shared memory, required for multi-process mode)
- MPI (optional, only needed for multi-process tests)

### Compiling

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Manually selecting a backend:

```bash
cmake .. -DPCCL_USE_CUDA=ON   # Force CUDA
cmake .. -DPCCL_USE_CANN=ON   # Force Ascend CANN
```

### CMake Options

| Option | Default | Description |
|------|--------|------|
| `PCCL_USE_CUDA` | OFF (auto-detect) | Force-enable the CUDA backend |
| `PCCL_USE_CANN` | OFF (auto-detect) | Force-enable the Ascend CANN backend |
| `PCCL_BUILD_NCCL_COMPAT` | ON | Build the NCCL compatibility layer |
| `PCCL_TRACE` | OFF | Enable Perfetto-compatible trace output (Chrome JSON format) |
| `PCCL_LOG_LEVEL` | 2 | Log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR) |
| `PCCL_CUDA_ARCHITECTURES` | auto | CUDA kernel architectures; `auto` prefers CMake `native`, and on older CMake versions tries to detect via `nvidia-smi`. Can also be set explicitly to `70;80`, `native`, `all`, `all-major`, or `OFF` |
| `ASCEND_PRODUCT_TYPE` | Ascend310P1 | Ascend chip model (Ascend910A, Ascend310P1, Ascend310B1, Ascend910B1, Ascend910B3) |
| `ASCEND_CORE_TYPE` | VectorCore | Ascend core type (AiCore, VectorCore) |

### Installation

```bash
cmake --install build --prefix /usr/local
```

Installed artifacts:
- Libraries → `lib/`
- Headers → `include/pccl/`
- IR files → `share/pccl/ir/`
- CMake config → `lib/cmake/pccl/`

## Quick Start

**Single-process mode** (one process manages all GPUs, submitting in parallel from multiple threads):

```cpp
#include <pccl/pccl.h>

devInit();

// Single-process mode: pass an all-zero UniqueId
pcclUniqueId id = {};
pcclComm_t comms[NRANKS];
devStream  streams[NRANKS];
for (int r = 0; r < NRANKS; ++r) {
    devSetDevice(r);
    pcclCommInitRank(&comms[r], NRANKS, id, r);
    devCreateStream(&streams[r]);
}

// Each thread submits collective operations in parallel
for (int r = 0; r < NRANKS; ++r) {
    devSetDevice(r);
    pcclAllReduce(d_send[r], d_recv[r], count, comms[r], streams[r]);
}
for (int r = 0; r < NRANKS; ++r) {
    devSetDevice(r);
    devSynchronizeStream(streams[r]);
}

for (int r = 0; r < NRANKS; ++r) pcclCommDestroy(comms[r]);
pcclFinalize();
```

**Multi-process mode** (each process manages one GPU, launched via MPI or similar):

```cpp
#include <pccl/pccl.h>
#include <mpi.h>

MPI_Init(&argc, &argv);
int rank, nranks;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nranks);

devInit();
devSetDevice(rank);

// Rank 0 generates the UniqueId and broadcasts it
pcclUniqueId id;
if (rank == 0) pcclGetUniqueId(&id);
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

pcclComm_t comm;
devStream stream;
pcclCommInitRank(&comm, nranks, id, rank);
devCreateStream(&stream);

pcclAllReduce(d_send, d_recv, count, comm, stream);
devSynchronizeStream(stream);

pcclCommDestroy(comm);
pcclFinalize();
MPI_Finalize();
```

For complete examples, see the test programs under `tests/coll/` (SP versions, and MP versions with the `_mp` suffix).

## API Reference

### Communicator Management

```c
// Generate a unique ID (in multi-process mode, called by rank 0 and broadcast to all ranks)
pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId);

// Create a communicator (process-level state is initialized automatically on the first call)
// uniqueId: in multi-process mode, pass the ID generated by pcclGetUniqueId;
//           in single-process mode, pass an all-zero pcclUniqueId
pcclResult_t pcclCommInitRank(pcclComm_t* comm, int nranks, pcclUniqueId uniqueId, int rank);

// Destroy a communicator
pcclResult_t pcclCommDestroy(pcclComm_t comm);

// Query rank and world size
pcclResult_t pcclCommRank(pcclComm_t comm, int* rank);
pcclResult_t pcclCommSize(pcclComm_t comm, int* size);

// Process-level cleanup (destroys all shared resources)
pcclResult_t pcclFinalize(void);
```

### Collective Operations

All collective operations are asynchronous: they return immediately after being submitted to the given device stream, and completion is awaited via stream synchronization.

```c
// AllReduce: sum the data from all ranks and broadcast the result to all ranks
// count — number of elements
pcclResult_t pcclAllReduce(
    const void* sendbuff, void* recvbuff,
    size_t count, pcclComm_t comm, pcclStream_t stream);

// AllGather: each rank sends sendcount elements and receives the concatenation from all ranks
// sendcount — number of elements sent by each rank
pcclResult_t pcclAllGather(
    const void* sendbuff, void* recvbuff,
    size_t sendcount, pcclComm_t comm, pcclStream_t stream);

// ReduceScatter: sum the data from all ranks, then scatter the result across ranks
// recvcount — number of elements received by each rank
pcclResult_t pcclReduceScatter(
    const void* sendbuff, void* recvbuff,
    size_t recvcount, pcclComm_t comm, pcclStream_t stream);

// AllToAll: each rank sends a distinct data segment to every other rank
// count — total number of elements sent/received by each rank
pcclResult_t pcclAllToAll(
    const void* sendbuff, void* recvbuff,
    size_t count, pcclComm_t comm, pcclStream_t stream);
```

### Types

```c
// Unique identifier for multi-process coordination (128 bytes, value type)
#define PCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[PCCL_UNIQUE_ID_BYTES]; } pcclUniqueId;
```

### Error Handling

```c
// Return value type
typedef enum {
    pcclSuccess          = 0,
    pcclInvalidArgument  = 1,
    pcclSystemError      = 2,
    pcclInternalError    = 3,
    pcclDeviceError      = 4,
    pcclNetworkError     = 5,
    pcclTimeout          = 6
} pcclResult_t;

// Convert an error code to a string
const char* pcclGetErrorString(pcclResult_t result);
```

## IR Semantics

### Overview

PCCL uses an XML-based intermediate representation (IR) to declaratively describe the data-movement orchestration of collective operations. Each IR program defines an ordered list of instructions for every rank, specifying how data moves between device memory, host memory, and NUMA nodes.

IR program files live in the `ir/` directory and are named `<collective>_<num_gpus>gpu_<num_numa>numa.xml`.

### Segments and Chunks

The IR data model is organized around two core concepts:

A **segment** is a contiguous per-rank region of data in a device buffer. Different collective operations have different buffer segment layouts:

| Collective | sendbuf segments | recvbuf segments | Size of each segment |
|----------|-------------------|-------------------|------------------|
| AllReduce | 1 | 1 | `count` elements |
| AllGather | 1 | nranks | `sendcount` elements |
| ReduceScatter | nranks | 1 | `recvcount` elements |
| AllToAll | nranks | nranks | `count / nranks` elements |

A **chunk** is the unit of data operated on by IR instructions and is a further subdivision of a segment. Each IR instruction (D2H, H2D, H2H, etc.) moves exactly one chunk of data. A segment may contain one or more chunks; the chunk size is computed automatically by the compiler from the segment size and the number of chunks in the IR program: `chunk_size = pcie_per_segment / chunks_per_segment`.

On the device side, chunks are mapped by index to physical addresses in the device buffer. On the host side, each chunk is uniquely identified by a `(numa_id, chunk_idx)` pair and corresponds to a temporary buffer in the host memory pool. There are usually more host chunks than device chunks, because operations such as reductions need extra intermediate storage.

At execution time, each chunk is further split into 1MB **subchunks** so that D2H/H2D transfers can be pipelined and overlapped with H2H computation.

### Opcodes

| Opcode | Semantics |
|--------|------|
| `D2D` | Device-side copy (e.g., an intra-GPU D2D transfer) |
| `D2H` | Device-to-host transfer (GPU → host, over PCIe) |
| `H2D` | Host-to-device transfer (host → GPU, over PCIe) |
| `H2H` | Host-to-host memory copy (`dst = src`) |
| `REDUCE` | Three-address host reduction (`dst = src1 + src2`), with the result written to a new chunk |

### Instruction Format

Each instruction contains the following fields:

| Field | Description |
|------|------|
| `op` | Opcode (see table above) |
| `src_chunk` / `src_numa` | Source chunk index and NUMA node ID (used by H2H/H2D) |
| `dst_chunk` / `dst_numa` | Destination chunk index and NUMA node ID (used by D2H/H2H/REDUCE) |
| `src1_chunk` / `src1_numa` | First source operand of REDUCE |
| `src2_chunk` / `src2_numa` | Second source operand of REDUCE |

For D2H instructions, `src_chunk` refers to a device-side chunk index; for H2D instructions, `dst_chunk` refers to a device-side chunk index. Host-side chunks are uniquely identified by a `(numa_id, chunk_idx)` pair.

**Single-writer model:** Each host chunk has exactly one writer. Reductions do not accumulate in place (`dst += src`); instead, the result is written to a new chunk (`dst = src1 + src2`). Multi-input reductions are implemented via binary-tree decomposition (e.g., 4 inputs are split into two levels: two pairwise reductions plus one merge). This design eliminates the complexity of multi-writer pipeline synchronization; all dependencies are resolved by SpinWaiting on the source chunks' progress counters.

### XML Format Example

Taking AllReduce on 2 GPUs with a single NUMA node as an example (`ir/allreduce_00.xml`):

```xml
<ir_program name="allreduce" nranks="2" topology="1numa_2gpu_per_numa" nnodes="1">
  <description>
    D2H to hc0,hc1. REDUCE(hc0, hc1) → hc2. H2D from hc2.
  </description>

  <rank id="0">
    <instruction op="D2H" src_chunk="0" dst_chunk="0" dst_numa="0"/>
    <instruction op="REDUCE" src1_chunk="0" src1_numa="0" src2_chunk="1" src2_numa="0" dst_chunk="2" dst_numa="0"/>
    <instruction op="H2D" src_chunk="2" dst_chunk="0" src_numa="0"/>
  </rank>

  <rank id="1">
    <instruction op="D2H" src_chunk="0" dst_chunk="1" dst_numa="0"/>
    <instruction op="H2D" src_chunk="2" dst_chunk="0" src_numa="0"/>
  </rank>
</ir_program>
```

Execution flow:
1. Rank 0 and Rank 1 each transfer their data via D2H to host chunk 0 and chunk 1, respectively
2. Rank 0 reduces chunk 0 and chunk 1 via REDUCE into a new chunk 2 (`hc2 = hc0 + hc1`)
3. Both ranks write the reduced result from chunk 2 back to their own GPUs via H2D

### Compilation Pipeline

IR programs are compiled by the `Compiler` (`src/ir/compiler.hpp`) as follows:

1. **IR loading** — `IRLoader` automatically locates and parses the XML file based on the collective name and topology
2. **Address mapping** — Maps chunk indices in the IR to physical device addresses and host memory addresses (strided addressing supported)
3. **Subchunk slicing** — Splits large transfers into 1MB subchunks for pipelined transfer
4. **Execution graph generation** — Outputs an address-independent `CompiledGraph` in which dependencies between nodes are expressed via versioned chunk progress; at runtime, a `RuntimeBinding` is generated from the actual addresses of the current collective

## Strided Collectives

### Background

When MPCCL splits a collective operation into a fast path (e.g., NVLink) and a PCIe path, the fast path handles the first `stride` elements of each segment and PCCL handles the rest. Because PCCL has to skip the fast-path region in every segment, this results in a non-contiguous, strided access pattern.

### API

```c
pcclResult_t pcclAllGatherStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclReduceScatterStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);

pcclResult_t pcclAllToAllStrided(
    const void* sendbuff, void* recvbuff,
    size_t count, size_t stride,
    pcclComm_t comm, pcclStream_t stream);
```

### Parameter Semantics

- **`count`** — Meaning depends on the collective:
  - `AllGatherStrided`: number of elements per segment (= sendcount)
  - `ReduceScatterStrided`: total number of elements (= recvcount × nranks); divided by nranks internally to obtain seg_count
  - `AllToAllStrided`: total number of elements (= per-rank count × nranks); divided by nranks internally to obtain seg_count
- **`stride`** — Number of leading elements in each segment handled by the fast path. PCCL operates on `(seg_count - stride)` elements. Setting it to 0 is equivalent to the non-strided version.

The per-segment size includes both the fast-path and PCIe portions: `seg_count = count` (or `count / nranks`).

### Address Calculation Formula

```
segment_size   = seg_count * elem_size
stride_offset  = stride * elem_size
pcie_per_seg   = segment_size - stride_offset
chunks_per_seg = device_chunks / num_segments
chunk_size     = pcie_per_seg / chunks_per_seg

addr = base + seg * segment_size + stride_offset + chunk_in_seg * chunk_size
       where seg          = chunk_idx / chunks_per_seg
             chunk_in_seg = chunk_idx % chunks_per_seg
```

`INPUT` / `OUTPUT` / `SCRATCH` use the strided address mapping above. PCCL's temporary `INTERNAL` device buffer is a compactly allocated staging area without fast-path holes, so internal chunks use a contiguous address mapping; this is especially important for the temporary reductions in ReduceScatterStrided.

### Example: AllGatherStrided

Scenario: 4 ranks, `count = 64M` (elements), `stride = 48M` (elements), `elem_size = 4B` (float32).

**Deriving the basic parameters:**

- `seg_count = count = 64M` elements (for AllGather, count is the number of elements per segment)
- `segment_size = seg_count × elem_size = 64M × 4B = 256MB`
- `stride_offset = stride × elem_size = 48M × 4B = 192MB` (fast-path region)
- `pcie_per_seg = segment_size - stride_offset = 256MB - 192MB = 64MB` (region handled by PCCL)

**sendbuf (input, 1 segment):**

sendbuf holds the current rank's own data and has only 1 segment of 64M elements (256MB). The first 48M elements (192MB) are handled by the fast path; PCCL reads the remaining 16M elements (64MB) starting at offset 192MB.

**recvbuf (output, 4 segments):**

recvbuf holds the concatenated result from all ranks: 4 segments (corresponding to ranks 0–3), 1024MB in total. Each segment is 256MB; the first 192MB is written by the fast path, and PCCL is responsible for writing the last 64MB.

PCCL writes chunk k (corresponding to rank k) to address `recvbuf + k × 256MB + 192MB`. For example, chunk 2 is written to `recvbuf + 512MB + 192MB = recvbuf + 704MB`.

### Test Coverage

`test_allgather_strided`, `test_reducescatter_strided`, and `test_alltoall_strided` cover the following configurations:

| Configuration | `seg_count` | `stride` | Share of data handled by PCCL |
|------|-------------|----------|---------------------|
| `75% fast-path` | `32M` float | `24M` float | 25% |
| `50% fast-path` | `32M` float | `16M` float | 50% |
| `25% fast-path` | `32M` float | `8M` float | 75% |
| `contiguous` | `32M` float | `0` | 100% |

These tests verify both that the fast-path region remains untouched and that the PCCL region is correct, and they report latency, algorithm bandwidth, and bus bandwidth in the same format as the regular collective tests.

## Performance Tracing (Perfetto Trace)

PCCL has a built-in performance tracing system based on the Chrome Trace Event Format, producing JSON files that can be viewed directly in the [Perfetto UI](https://ui.perfetto.dev).

### Enabling

```bash
cmake .. -DPCCL_TRACE=ON
make -j$(nproc)

# Specify the output path at runtime via an environment variable
PCCL_TRACE_FILE=trace.json ./test_allgather 4
```

### Trace Contents

The generated trace contains the following information, displayed on separate Perfetto tracks:

| Track | Description | Timing method |
|-------|------|----------|
| `Rank N D2H` | D2H executor CPU thread | Total batch duration |
| `Rank N D2H Stream K` | GPU D2H DMA transfers | Event query + per-stream chained timing |
| `Rank N H2D` | H2D executor CPU thread | Total batch duration |
| `Rank N H2D Stream K` | GPU H2D DMA transfers | Event query + per-stream chained timing |
| `H2H NUMA M/#W` | H2H worker threads | Precise per-subchunk timing |

Each track's tid is assigned by a fixed rule so that tracks are ordered in Perfetto as Rank → D2H/H2D → H2H NUMA.

### GPU DMA Timing

After the CPU submits a DMA, it marks the completion point on the GPU stream with `devRecordEvent`, then polls for completion with `devQueryEvent`. For multiple DMAs on the same stream, **chained timing** is used to avoid overlapping intervals:

```
begin = max(submit_ts, complete_ts of the previous DMA on the same stream)
end   = CPU time when devQueryEvent returns Complete
```

### H2H Event Details

The args field of H2H Copy/Reduce events contains:

| Field | Description |
|------|------|
| `bw_gbps` | Instantaneous bandwidth for this subchunk (GB/s) |
| `src1_numa` / `dst_numa` | Source/destination NUMA node |
| `bytes` | Number of bytes transferred |
| `sc_idx` | Subchunk index |
| `cpu_us` | CPU time (microseconds) |
| `cpu_before` / `cpu_after` | CPU core ID before and after execution |

Click any H2H event slice in Perfetto to view these fields in the Arguments table of the bottom panel.

### Design Highlights

- **Zero external dependencies** — Pure C++14 implementation, with no dependency on the Perfetto SDK or protobuf
- **Thread-local buffering** — Each thread has its own event shard; the hot path is lock-free (only `vector::push_back` + a timestamp read)
- **Unified backend** — All platforms (CUDA / CANN) use the same tracing system
- **Runtime control** — Enabled at compile time with `-DPCCL_TRACE=ON`; at runtime, the `PCCL_TRACE_FILE` environment variable controls whether output is actually produced (no overhead if unset)
- **Multi-process alignment** — Each trace file records `wall_epoch_us` (a wall-clock timestamp), so in multi-process mode the timelines of different ranks can be aligned in Perfetto

## Configuration

### Environment Variables

| Variable | Description | Default |
|------|------|--------|
| `PCCL_TRACE_FILE` | Trace output path; tracing is enabled when set | No output |
| `PCCL_GPU_NUMA_MAP` | GPU-to-NUMA-node mapping, e.g., `"0,0,1,1"` | Auto-detected |
| `PCCL_HOST_BUFFER_SIZE` | Host buffer size per NUMA node (bytes) | 4 GB |
| `PCCL_IR_PATH` | Search path for IR XML files | `./ir`, `/usr/local/share/pccl/ir` |
| `PCCL_IR_DIR` | Compatibility alias for `PCCL_IR_PATH` | `./ir`, `/usr/local/share/pccl/ir` |
| `PCCL_NRANKS` | Total number of ranks | — |
| `PCCL_BOOTSTRAP_ADDR` | Multi-node bootstrap address, in the form `"host:port"` | Single-node mode |
| `PCCL_D2D_PUSH` | For remote D2D, the source rank actively writes to the destination rank; set to `0` to have the destination rank read from the source rank instead | `1` |
| `PCCL_D2D_INPUT_FIRST` | Overrides the scheduling policy for remote D2D graph nodes; set to `1` to prioritize D2D nodes that read INPUT | Collective default policy |
| `PCCL_BARRIER_TIMEOUT_MS` | Wait timeout for shared-state barriers; on timeout, returns `pcclTimeout` and prints diagnostics | `120000` |
| `PCCL_EXECUTOR_TIMEOUT_MS` | Diagnostic timeout for the graph executor waiting for idle, CUDA events, H2H completion, runGraph spinning without progress, etc. | `120000` |
| `PCCL_REMOTE_BUFFER_TIMEOUT_MS` | In multi-process CUDA IPC mode, timeout for waiting for remote ranks to publish their device buffers | `120000` |
| `PCCL_CUDA_IPC_CLOSE_ON_FINALIZE` | Whether to force `cudaIpcCloseMemHandle` on remote CUDA IPC mappings at finalize; non-`0` enables it | `0` |
| `PCCL_NCCL_COMPAT_MODE` | SP/MP mode selection for the NCCL compatibility layer: `auto`, `sp`/`single`, `mp`/`multi` | `auto` |
| `PCCL_NCCL_COMPAT_MODE_TIMEOUT_MS` | Timeout for the NCCL compatibility layer's SP/MP auto-detection while waiting for the rendezvous shared memory and rank state | `120000` |
| `PCCL_NCCL_COMPAT_MODE_GRACE_MS` | During NCCL compatibility layer auto-detection, grace period before selecting single-process mode when no other process PIDs have been observed | `1000` |
| `PCCL_NCCL_COMPAT_HOST_FALLBACK_BYTES` | Small-message host fallback threshold for the NCCL compatibility layer in multi-process mode | Determined by the compatibility layer default |
| `PCCL_NCCL_COMPAT_HOST_COLL_TIMEOUT_MS` | Wait timeout for host fallback collectives in the NCCL compatibility layer | `120000` |
| `PCCL_NCCL_COMPAT_BCAST_STAGE_BYTES` | Host staging chunk size for multi-process Broadcast in the NCCL compatibility layer | `4194304` |
| `PCCL_NCCL_COMPAT_BCAST_TIMEOUT_MS` | Wait timeout for multi-process Broadcast in the NCCL compatibility layer | `120000` |

### pcclConfig_t

```c
typedef struct {
    int nranks;                 // Total number of ranks
    int device_count;           // Number of local GPUs/NPUs
    const int* gpu_ids;         // List of device IDs (NULL = auto)
    const char* bootstrap_addr; // Multi-node bootstrap address (NULL = single node)
    size_t host_buffer_size;    // Host buffer size per NUMA node (0 = auto)
    const char* gpu_numa_map;   // GPU-NUMA mapping string (NULL = auto-detect)
    const char* ir_path;        // IR XML search path (NULL = default)
} pcclConfig_t;
```

## Testing

After building, the test executables are located in the `build/` directory:

```bash
cd build

# Single-process collective tests (argument is the number of ranks)
./test_allgather 4
./test_allreduce 4
./test_reducescatter 4
./test_alltoall 4

# Strided variants
./test_allgather_strided 4
./test_reducescatter_strided 4
./test_alltoall_strided 4

# Multi-process collective tests (requires MPI)
mpirun -np 4 ./test_allgather_mp
mpirun -np 4 ./test_allreduce_mp

# With trace output
PCCL_TRACE_FILE=trace.json ./test_allgather 4
# Open trace.json at https://ui.perfetto.dev to view it
```

Test coverage:

- **Collectives**: AllReduce, AllGather, ReduceScatter, AllToAll
- **Deployment modes**: single-process multi-threaded (SP), multi-process (MP, requires MPI)
- **GPU configurations**: 2 GPUs, 4 GPUs, 8 GPUs
- **NUMA topologies**: 1 NUMA, 2 NUMA, 4 NUMA
- **Strided variants**: strided versions of AllGather / ReduceScatter / AllToAll
- **Strided split ratios**: 75% / 50% / 25% fast-path and contiguous, with correctness and bandwidth validation for AllGather / ReduceScatter / AllToAll

Each test performs correctness verification and bandwidth benchmarking, reporting algorithm bandwidth and bus bandwidth.

### Testing with nccl-tests

PCCL provides an NCCL compatibility layer, so it can be tested directly with [nccl-tests](https://github.com/NVIDIA/nccl-tests) via `LD_PRELOAD`:

The NCCL compatibility layer currently covers the common call paths of Megatron / PyTorch ProcessGroupNCCL:

- Communicator lifecycle: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommInitRankConfig`, `ncclCommInitAll`, `ncclCommSplit`, `ncclCommDestroy`, `ncclCommAbort`, `ncclCommFinalize`
- Query and registration: `ncclCommCount`, `ncclCommUserRank`, `ncclCommCuDevice`, `ncclCommGetAsyncError`, `ncclCommRegister`, `ncclCommDeregister`, `ncclMemAlloc`, `ncclMemFree`
- Group semantics: `ncclGroupStart`, `ncclGroupEnd`, `ncclGroupSimulateEnd`
- Collectives: `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast` / `ncclBcast`, `ncclReduce`
- dtypes: `int8`, `uint8`, `int32`, `uint32`, `int64`, `uint64`, `float16`, `float32`, `float64`, `bfloat16`
- Reduce ops: `sum`, `max`, `min`

Copy-only collectives move data byte-wise; reduction collectives are dispatched to typed device reduce kernels according to dtype and reduce op. 16-bit floating-point reductions rely on the device reduce path; if an IR requires host-side `H2HReduce`, the compatibility layer rejects that 16-bit path.

Note: `ncclSend` / `ncclRecv` are not yet implemented. `ncclCommSplit` currently creates a usable new communicator but does not yet fully cover NCCL's `color` / `key` ordering semantics.

```bash
# AllReduce (4M-256M, 2GPU, validation)
PCCL_IR_PATH=/workspace/pcieccl/ir PCCL_LOG_LEVEL=3 \
  LD_PRELOAD=/workspace/pcieccl/build/libpccl.so \
  /workspace/nccl-tests/build/all_reduce_perf -b 4M -e 256M -f 2 -g 2 -p 1 -w 1 -n 5

# AllGather
PCCL_IR_PATH=/workspace/pcieccl/ir PCCL_LOG_LEVEL=3 \
  LD_PRELOAD=/workspace/pcieccl/build/libpccl.so \
  /workspace/nccl-tests/build/all_gather_perf -b 4M -e 256M -f 2 -g 2 -p 1 -w 1 -n 5

# ReduceScatter
PCCL_IR_PATH=/workspace/pcieccl/ir PCCL_LOG_LEVEL=3 \
  LD_PRELOAD=/workspace/pcieccl/build/libpccl.so \
  /workspace/nccl-tests/build/reduce_scatter_perf -b 4M -e 256M -f 2 -g 2 -p 1 -w 1 -n 5

# MPI AllGather (1 GPU per process)
mpirun --mca coll_hcoll_enable 0 -np 4 env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PCCL_IR_PATH=/workspace/pcieccl/ir PCCL_LOG_LEVEL=3 \
  LD_PRELOAD=/workspace/pcieccl/build/libpccl.so \
  /workspace/nccl-tests/build_mpi/all_gather_perf_mpi -b 4M -e 256M -f 2 -g 1 -p 1 -w 1 -n 5
```

**Environment variables:**

- `PCCL_IR_PATH` — Points to the directory containing the IR XML files
- `PCCL_LOG_LEVEL=3` — Sets the level to ERROR to reduce output noise
- `LD_PRELOAD` — Intercepts NCCL API calls by preloading the PCCL library

**nccl-tests parameters:**

| Parameter | Description |
|------|------|
| `-b 4M -e 256M -f 2` | Data size from 4M to 256M, multiplied by 2 at each step |
| `-g 2` | Use 2 GPUs |
| `-p 1` | Parallel initialization (must be enabled due to a bug in nccl-tests itself) |
| `-w 1 -n 5` | 1 warmup iteration, 5 measured iterations |

### Megatron End-to-End Benchmark

The Megatron end-to-end test verifies that PCCL's NCCL compatibility layer works on the PyTorch `ProcessGroupNCCL` path and measures real training iteration times. The currently recommended large-message benchmark configuration uses `/workspace/megatron`, with 4 processes bound to GPUs `0,2,4,6`, TP=4, PP=1, mock data, fp16, and `transformer_engine`.

This configuration mainly exercises the large-message AllGather / ReduceScatter / AllReduce paths in Megatron TP. Do not use `torch_dist_smoke/run_megatron.sh` as a source for official performance conclusions; that script is a smoke test with a smaller model and also includes `--sequence-parallel` by default.

**Common configuration:**

| Item | Value |
|----|----|
| Megatron path | `/workspace/megatron` |
| GPU | `CUDA_VISIBLE_DEVICES=0,2,4,6` |
| Number of processes | `torch.distributed.run --nproc_per_node=4` |
| Parallelism | `--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1` |
| Model | `--num-layers 4 --hidden-size 8192 --ffn-hidden-size 28672 --num-attention-heads 64` |
| Sequence | `--seq-length 4096 --max-position-embeddings 4096` |
| Batch | `--micro-batch-size 1 --global-batch-size 4` |
| Precision | `--fp16` |
| Transformer implementation | `--transformer-impl transformer_engine` |
| Data | `--mock-data --tokenizer-type NullTokenizer --vocab-size 8192` |
| Iterations | `--train-iters 12 --log-interval 1` |
| Backend argument | `--distributed-backend nccl` |
| Not used | Do not add `--sequence-parallel` |
| CUDA connections | `CUDA_DEVICE_MAX_CONNECTIONS=8` |

`CUDA_DEVICE_MAX_CONNECTIONS=8` is the recommended setting for the current PCCL Megatron benchmark. `CUDA_DEVICE_MAX_CONNECTIONS=1` limits the number of hardware connections, which does not match PCCL's CPU-orchestrated, multi-CUDA-stream execution model. Megatron / TransformerEngine may still print a warning recommending a value of 1; it can be ignored.

**Build the library under test:**

```bash
cd /workspace/pcieccl-noctx
cmake --build build_quiet -j$(nproc)
```

To compare against a clean HEAD, create a separate worktree so that uncommitted changes in the current working tree do not affect the HEAD baseline:

```bash
HEAD_ROOT=/workspace/pcieccl_no_fastpath_head_$(git rev-parse --short HEAD)
git worktree add "$HEAD_ROOT" HEAD
cmake -S "$HEAD_ROOT" \
      -B "$HEAD_ROOT/build_quiet" \
      -DPCCL_USE_CUDA=ON -DPCCL_BUILD_NCCL_COMPAT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$HEAD_ROOT/build_quiet" -j$(nproc)
```

**Recommended launcher:**

```bash
cat >/tmp/pccl_run_megatron_bench.sh <<'EOF'
#!/usr/bin/env bash

PCCL_BASE_LD_LIBRARY_PATH="${PCCL_BASE_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"

run_megatron_bench() {
  local name="$1"
  local pccl_root="$2"
  local build_dir="$3"
  local fastpath_mode="$4"   # on, off, or nccl
  local out_dir="${5:-/workspace/megatron/bench_results/pccl_megatron_0246_$(date +%Y%m%d_%H%M%S)}"

  mkdir -p "$out_dir"
  cd /workspace/megatron

  export CUDA_VISIBLE_DEVICES=0,2,4,6
  export CUDA_DEVICE_MAX_CONNECTIONS=8
  export CUDA_MODULE_LOADING=EAGER
  export PYTHONNOUSERSITE=1
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="${MASTER_PORT:-29771}"

  if [ "$fastpath_mode" = "nccl" ]; then
    unset LD_PRELOAD PCCL_IR_PATH PCCL_SMALL_MSG_FASTPATH PCCL_SMALL_MSG_THRESHOLD_BYTES
    unset PCCL_SMALL_MSG_FASTPATH_TIMEOUT_MS PCCL_NCCL_COMPAT_MODE PCCL_EXECUTOR_TIMEOUT_MS
    unset PCCL_BARRIER_TIMEOUT_MS PCCL_LOG_LEVEL
    export LD_LIBRARY_PATH="$PCCL_BASE_LD_LIBRARY_PATH"
  else
    export LD_PRELOAD="$pccl_root/$build_dir/libpccl.so"
    export LD_LIBRARY_PATH="$pccl_root/$build_dir:${PCCL_BASE_LD_LIBRARY_PATH:-}"
    export PCCL_IR_PATH="$pccl_root/ir"
    export PCCL_NCCL_COMPAT_MODE=mp
    export PCCL_EXECUTOR_TIMEOUT_MS=30000
    export PCCL_BARRIER_TIMEOUT_MS=40000
    export PCCL_LOG_LEVEL="${PCCL_LOG_LEVEL:-2}"
    if [ "$fastpath_mode" = "off" ]; then
      export PCCL_SMALL_MSG_FASTPATH=0
    else
      unset PCCL_SMALL_MSG_FASTPATH
    fi
  fi

  local log="$out_dir/${name}.log"
  local cache="/tmp/pccl_megatron_${name}/cache"
  local status=0
  rm -rf "$cache"

  timeout 600 python3 -m torch.distributed.run --nproc_per_node=4 --master_port="$MASTER_PORT" \
    /workspace/megatron/pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 4 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --train-iters 12 \
    --log-interval 1 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 100 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type NullTokenizer \
    --vocab-size 8192 \
    --mock-data \
    --data-cache-path "$cache" \
    --eval-iters 0 \
    --transformer-impl transformer_engine \
    --distributed-backend nccl \
    >"$log" 2>&1 || status=$?

  printf 'EXIT=%d name=%s log=%s\n' "$status" "$name" "$log" | tee "$out_dir/${name}.status"
  return "$status"
}
EOF

source /tmp/pccl_run_megatron_bench.sh
```

**Run the current version, fastpath disabled, HEAD, or native NCCL:**

```bash
# Current working tree, small-message fastpath enabled by default
run_megatron_bench current_fastpath /workspace/pcieccl-noctx build_quiet on \
  /workspace/megatron/bench_results/megatron_0246

# Current working tree, small-message fastpath explicitly disabled, to check the large-message path outside the fastpath
run_megatron_bench current_fastpath_disabled /workspace/pcieccl-noctx build_quiet off \
  /workspace/megatron/bench_results/megatron_0246

# Clean HEAD worktree; if the fastpath changes are not yet committed, this is the no-fastpath baseline
HEAD_ROOT=${HEAD_ROOT:-/workspace/pcieccl_no_fastpath_head_1d2de024}
run_megatron_bench head_clean "$HEAD_ROOT" build_quiet on \
  /workspace/megatron/bench_results/megatron_0246

# Native NCCL, without preloading PCCL
run_megatron_bench native_nccl /workspace/pcieccl-noctx build_quiet nccl \
  /workspace/megatron/bench_results/megatron_0246
```

If your HEAD worktree directory is not `/workspace/pcieccl_no_fastpath_head_1d2de024`, simply replace `HEAD_ROOT` with the actual worktree path. When comparing current fastpath-disabled against clean HEAD, both should take the large-message IR/GraphExecutor path; `PCCL_SMALL_MSG_FASTPATH=0` only disables the small-message fastpath initialization and routing in the current version.

**Summarizing results:**

Megatron's timing lines have the format `elapsed time per iteration (ms): ...`. Typically, the first warmup iteration is skipped and iterations 2-12 are aggregated:

```bash
for f in /workspace/megatron/bench_results/megatron_0246/*.log; do
  printf '%s\n' "$(basename "$f")"
  rg "iteration +[0-9]+/ +12" "$f" |
    sed -E 's/.*iteration +([0-9]+).*elapsed time per iteration \(ms\): ([0-9.]+).*/\1 \2/' |
    awk '$1>=2{print $2}' |
    sort -n |
    awk 'NF{v[++n]=$1; sum+=$1; if(n==1||$1<min)min=$1; if($1>max)max=$1}
         END{if(n){mid=int((n+1)/2); printf("iters2-12 n=%d avg=%.1f p50=%.1f min=%.1f max=%.1f\n", n, sum/n, v[mid], min, max)}
             else {print "no iteration samples"}}'
done
```

For correctness, at minimum check that the `lm loss` and `grad norm` of the final iteration match, and confirm that the process exit codes are 0:

```bash
rg "iteration +12/ +12|lm loss|grad norm|Traceback|RuntimeError|internal error|timed out" \
  /workspace/megatron/bench_results/megatron_0246/*.log
cat /workspace/megatron/bench_results/megatron_0246/*.status
```

Under the current 0246 configuration, stability should be judged by the p50 and avg over multiple alternating runs. Tail iterations of a single run may fluctuate by up to about 1%; to detect small regressions, alternate 3-5 rounds in the order `HEAD -> current disabled -> current fastpath` before comparing.
