English | [中文](README_zh.md)

# PCIeCCL Ascend

This repository is the Ascend CANN port of PCIeCCL for single-node Ascend 910B3 systems.
The build produces `libpccl.so` and creates a `libhccl.so` symlink pointing at the same library,
so that some tests based on HCCL root-info can run on top of the PCIeCCL implementation.

## Scope

- Backend: Ascend CANN only. The build does not probe for or link against CUDA, NCCL or NVML.
- Topology: single node. The IR bundled in the repository covers the standard 2 / 4 / 8 card
  layouts, as well as the logical NUMA patterns `00`, `0011` and `00112233`.
- IR operations: `NOP`, `D2H`, `H2D`, `H2H` copy, local `D2D` and local
  `D2DReduce`.
- Not supported: CUDA, NCCL compatibility, direct cross-rank D2D, device IPC, `H2HReduce`,
  CPU-side reduce, multi-node bootstrap, PyTorch scripts, and variable-count HCCL
  collectives.
- The strided PCCL primitives are retained for correctness testing of segmented buffers where
  each segment has a reserved prefix.
- `D2DReduce` uses the CANN `aclrtReduceAsync`. HCCL allreduce and
  reducescatter support `int8`, `int32`, `fp16`, `fp32` and `bfp16`, with
  `sum`, `max` or `min`. Other reduce dtype/op combinations return
  `HCCL_E_NOT_SUPPORT`.
- H2H is used for copies only, and for host staging relay. Reduction collectives use the local
  Ascend `D2DReduce`; no CPU reduction is used.

## Build

```bash
cd /workspace/migrate/pcieccl-ascend
cmake -S . -B build -DASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
cmake --build build -j
```

Common CMake options:

- `ASCEND_HOME`: path to the CANN toolkit, defaults to
  `/usr/local/Ascend/ascend-toolkit/latest`
- `PCCL_BUILD_TESTS`: build the native tests and benchmarks, defaults to `ON`
- `PCCL_BUILD_HCCL_TEST`: build the repository's bundled `hccl_test` binaries when MPI is
  available, defaults to `ON`
- `PCCL_TRACE`: compile in Perfetto-compatible trace support, defaults to `OFF`
- `PCCL_LOG_LEVEL`: compile-time log level, defaults to `2` (`WARN`)

Main build artifacts:

- `build/libpccl.so`
- `build/libhccl.so -> libpccl.so`
- `build/test_allgather_strided`
- `build/test_reducescatter_strided`
- `build/test_alltoall_strided`
- `build/bench_alltoall_lanes`
- `build/bench_hccl_compat_sp`
- `build/hccl_test/bin/*_test` when MPI is found

## Runtime environment

| Variable | Purpose | Default |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | List of NPUs visible to ACL/CANN | runtime default |
| `PCCL_HOST_BUFFER_SIZE` | Host staging buffer size per logical NUMA; accepts raw bytes or a `K`/`M`/`G` suffix | `8G` |
| `PCCL_IR_PATH` / `PCCL_IR_DIR` | Override for the IR XML search directory | built-in `ir/` search path |
| `PCCL_NPU_NUMA_MAP` | Optional logical NUMA map, for example `0,0,1,1,2,2,3,3` | auto-discovered from PCI sysfs |
| `PCCL_GPU_NUMA_MAP` | Legacy alias for `PCCL_NPU_NUMA_MAP` | unset |
| `PCCL_EXECUTOR_TIMEOUT_MS` | Executor wait and diagnostic timeout | `120000` |
| `PCCL_BOOTSTRAP_TIMEOUT_MS` | Timeout for multi-process shared state and host-buffer bootstrap | `120000` |
| `PCCL_BARRIER_TIMEOUT_MS` | Multi-process shared barrier timeout | `120000` |
| `PCCL_EXECUTOR_QUEUE_DEPTH` | Number of async collective queue/doorbell slots; clamped to 1..512 | `512` |
| `PCCL_FORCE_DMA_LANES` | Force the number of D2H/H2D host DMA lanes, for performance tuning; truncated when above the compile-time limit | unset |
| `PCCL_STRIDED_PERF_ITERS` | Number of measurement iterations used by the strided native tests | `5` |
| `PCCL_STRIDED_PERF_WARMUP` | Number of warmup iterations used by the strided native tests | `2` |
| `PCCL_PERF_ITERS` | Number of iterations used by `run_pcieccl_perf_matrix.py` | `10` |
| `PCCL_PERF_WARMUP` | Number of warmups used by `run_pcieccl_perf_matrix.py` | `2` |

`PCCL_NPU_NUMA_MAP` usually does not need to be set. When it is unset, PCIeCCL reads each
NPU's PCI NUMA node from sysfs, compresses the physical NUMA nodes into logical IDs, and
selects an IR file such as `allreduce_00.xml`, `allreduce_0011.xml` or
`allreduce_00112233.xml`.

Multi-process mode uses shared 2 MB hugepage host staging together with
`aclrtHostRegister`. Before running the MPI tests, reserve enough hugepages on the physical
NUMA nodes used by the selected cards.

## Strided PCCL primitives

The strided native PCCL API supports segmented buffers in which every per-rank segment has a
reserved prefix at its front. PCCL skips that prefix, operates only on the active region that
follows it, and leaves the prefix contents unchanged. This is the same layout required when
another path already handles the first `stride` elements of each segment.

The exported C API is:

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

Parameter semantics:

- `stride` is the number of reserved elements at the start of each segment. It must be smaller
  than the number of elements per segment. `stride=0` means the whole segment is the active
  region.
- `pcclAllGatherStrided`: `count` is the number of segment elements per rank. The send
  buffer has one segment and the receive buffer has `nranks` segments.
- `pcclReduceScatterStrided`: `count` is the total number of send buffer elements across
  `nranks` segments, and must be divisible by `nranks`. The receive buffer has a single
  segment of `count / nranks` elements.
- `pcclAllToAllStrided`: `count` is the total number of elements across the `nranks`
  send/receive segments, and must be divisible by `nranks`.

External `INPUT`, `OUTPUT` and `SCRATCH` device buffers use strided address mapping:

```text
segment_size  = seg_count * elem_size
stride_offset = stride * elem_size
active_elems  = seg_count - stride

seg          = chunk_idx / chunks_per_segment
chunk_in_seg = chunk_idx % chunks_per_segment
begin_elem   = active_elems * chunk_in_seg / chunks_per_segment
end_elem     = active_elems * (chunk_in_seg + 1) / chunks_per_segment

addr  = base + seg * segment_size + stride_offset + begin_elem * elem_size
bytes = (end_elem - begin_elem) * elem_size
```

The internal staging buffers use a compact layout with no holes from the reserved prefixes. The
native strided API is not part of the HCCL compatibility layer.

The strided tests bundled in the repository use `float` buffers and cover four active-region
ratios:

| Case | `seg_count` | `stride` | Active region |
|---|---:|---:|---:|
| `25% active region` | `32M` floats | `24M` floats | 25% |
| `50% active region` | `32M` floats | `16M` floats | 50% |
| `75% active region` | `32M` floats | `8M` floats | 75% |
| `contiguous layout` | `32M` floats | `0` | 100% |

Each test verifies that the reserved prefix is still zero and that the active region contains
the expected collective result. The timing lines report average latency, plus algorithm
bandwidth and bus bandwidth computed from the active bytes only.

## Native tests and benchmarks

The native PCCL strided tests cover correctness and print simple timing and bandwidth metrics
based on the active bytes only:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/test_allgather_strided 2
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/test_reducescatter_strided 2
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/test_alltoall_strided 2
```

The AllToAll lane benchmark tests how message size and the number of host DMA lanes fit
together. Each process pins a single `PCCL_FORCE_DMA_LANES` value and by default tests total
message sizes of 1 / 4 / 8 / 16 / 32 / 64 / 128 MiB per rank, while verifying the AllToAll
rank/destination data pattern:

```bash
for lanes in 1 4 8 12; do
  PCCL_STRIDED_PERF_ITERS=10 PCCL_STRIDED_PERF_WARMUP=2 \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    ./build/bench_alltoall_lanes 4 "$lanes" 128M
done

# You can also directly verify the automatic policy selected by world size / message size.
PCCL_STRIDED_PERF_ITERS=10 PCCL_STRIDED_PERF_WARMUP=2 \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  ./build/bench_alltoall_lanes 4 auto 128M
```

Lane configurations must be run in separate processes, because the force option is parsed only
once per executor process. When comparing results, keep the topology, rank count, dtype,
warmup and iteration count identical. The current automatic policy distinguishes by rank count:
for 4 cards, `<=4 MiB: 8`, `<=8 MiB: 12`, `<=16 MiB: 8`, `<=32 MiB: 1`,
`<=64 MiB: 8`, `>64 MiB: 1`; for 8 cards, `<=1 MiB: 4`, `<=4 MiB: 12`,
`<=32 MiB: 4`, `<=64 MiB: 12`, `>64 MiB: 8`. Other rank counts use a generic fallback.
These thresholds come from measured starting points on the current machine; the final values
should be based on benchmark results on the target topology.

The single-process benchmark links against the native CANN HCCL by default, and can also be run
against PCIeCCL via `LD_PRELOAD`. It covers AllReduce, AllGather and
ReduceScatter with `fp32`/`sum`.

```bash
# Native HCCL baseline.
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  ./build/bench_hccl_compat_sp 4 16M 512M 10 2 --devices=0,1,2,3

# PCIeCCL through the HCCL compatibility layer.
LD_PRELOAD=$PWD/build/libpccl.so ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  ./build/bench_hccl_compat_sp 4 16M 512M 10 2 \
  --init=root --rootinfo=zero --devices=0,1,2,3
```

Use `--init=root --rootinfo=zero` for the PCIeCCL benchmark path, because the compatibility
layer implements root-info communicator initialization rather than `HcclCommInitAll`.

The performance matrix runner compares native HCCL with PCIeCCL for AllGather, ReduceScatter
and AllReduce on 2 / 4 / 8 cards, in both single-process and MPI mode:

```bash
PCCL_PERF_ITERS=10 PCCL_PERF_WARMUP=2 python3 ./run_pcieccl_perf_matrix.py
```

It writes the raw logs to `perf_results/<timestamp>/logs`, produces a parsed
`results.json` and a timestamped report, and updates `PCIe_CCL_perf_report.md`.

## HCCL compatibility

The compatibility layer exports the HCCL symbols implemented in
`src/api/hccl_compat/hccl_hook.cpp`:

- `HcclGetRootInfo`
- `HcclCommInitRootInfo`
- `HcclCommInitRootInfoConfig`
- `HcclCommDestroy`
- `HcclGetCommConfigCapability`
- `HcclAllReduce`
- `HcclAllGather`
- `HcclReduceScatter`
- `HcclAlltoAll`
- `HcclBroadcast`, with `root=0`
- `HcclCommSetMemoryRange`
- `HcclCommUnsetMemoryRange`
- `HcclCommActivateCommMemory`
- `HcclCommDeactivateCommMemory`
- `HcclCommSymWinRegister`
- `HcclCommSymWinDeregister`

`HcclAllGather`, `HcclAlltoAll` and `HcclBroadcast(root=0)` accept the fixed-count HCCL data
type mapping implemented by `mapDataType`: `int8`, `uint8`,
`int32`, `uint32`, `int64`, `uint64`, `fp16`, `fp32`, `fp64` and `bfp16`.
`HcclAlltoAll` additionally requires equal send/recv counts and identical send/recv types.

The compatibility layer does not implement `HcclCommInitAll`, `HcclReduce`, `HcclScatter`,
`HcclAllGatherV`, `HcclReduceScatterV`, `HcclAlltoAllV` or
`HcclAlltoAllVC`.

You can either use `LD_PRELOAD=build/libpccl.so`, or place `build/libhccl.so` ahead of the
native CANN HCCL library on `LD_LIBRARY_PATH`.

An example of running MPI through the repository's bundled `hccl_test` build:

```bash
mpirun --allow-run-as-root -n 4 env \
  LD_PRELOAD=$PWD/build/libpccl.so \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  HCCL_TEST_USE_DEVS=0,1,2,3 \
  ./build/hccl_test/bin/all_reduce_test \
  -b 16M -e 512M -f 2 -d fp32 -o sum -p 4 -n 10 -w 2 -c 1
```

For `all_gather_test`, omit `-o sum`. For `reduce_scatter_test`, keep
`-o sum`. For `alltoall_test`, use equal fixed counts and types. For
`broadcast_test`, use `-r 0`.
