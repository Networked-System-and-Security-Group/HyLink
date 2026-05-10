[English](README.md) | 中文

# PCIeCCL Ascend

本仓库是 PCIeCCL 面向单节点 Ascend 910B3 系统的 Ascend CANN 移植版本。
构建会生成 `libpccl.so`，并创建指向同一库的 `libhccl.so` 符号链接，
以便部分基于 HCCL root-info 的测试可以通过 PCIeCCL 实现运行。

## 范围

- 后端：仅支持 Ascend CANN。构建过程不会探测或链接 CUDA、NCCL 或 NVML。
- 拓扑：单节点。仓库内置的 IR 覆盖标准 2 / 4 / 8 卡布局，以及逻辑 NUMA
  模式 `00`、`0011` 和 `00112233`。
- IR 操作：`NOP`、`D2H`、`H2D`、`H2H` copy、本地 `D2D` 和本地
  `D2DReduce`。
- 不支持：CUDA、NCCL 兼容、跨 rank 直接 D2D、设备 IPC、`H2HReduce`、
  CPU 侧 reduce、多节点 bootstrap、PyTorch 脚本，以及可变 count 的 HCCL
  集合通信。
- Strided PCCL 原语被保留，用于每个 segment 带保留前缀的分段缓冲区正确性
  测试。
- `D2DReduce` 使用 CANN `aclrtReduceAsync`。HCCL allreduce 和
  reducescatter 支持 `int8`、`int32`、`fp16`、`fp32` 和 `bfp16`，并支持
  `sum`、`max` 或 `min`。其他 reduce dtype/op 组合会返回
  `HCCL_E_NOT_SUPPORT`。
- H2H 仅用于 copy，并用于 host staging relay。Reduction 集合通信使用本地
  Ascend `D2DReduce`，不使用 CPU reduction。

## 构建

```bash
cd /workspace/migrate/pcieccl-ascend
cmake -S . -B build -DASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
cmake --build build -j
```

常用 CMake 选项：

- `ASCEND_HOME`：CANN toolkit 路径，默认
  `/usr/local/Ascend/ascend-toolkit/latest`
- `PCCL_BUILD_TESTS`：构建本地测试和 benchmark，默认 `ON`
- `PCCL_BUILD_HCCL_TEST`：MPI 可用时构建仓库内置的 `hccl_test` 二进制文件，
  默认 `ON`
- `PCCL_TRACE`：编译 Perfetto 兼容的 trace 支持，默认 `OFF`
- `PCCL_LOG_LEVEL`：编译期日志级别，默认 `2`（`WARN`）

主要构建产物：

- `build/libpccl.so`
- `build/libhccl.so -> libpccl.so`
- `build/test_allgather_strided`
- `build/test_reducescatter_strided`
- `build/test_alltoall_strided`
- `build/bench_alltoall_lanes`
- `build/bench_hccl_compat_sp`
- 找到 MPI 时生成 `build/hccl_test/bin/*_test`

## 运行时环境

| 变量 | 用途 | 默认值 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | ACL/CANN 使用的可见 NPU 列表 | runtime 默认值 |
| `PCCL_HOST_BUFFER_SIZE` | 每个逻辑 NUMA 的 host staging buffer 大小；接受原始字节数或 `K`/`M`/`G` 后缀 | `8G` |
| `PCCL_IR_PATH` / `PCCL_IR_DIR` | IR XML 搜索目录覆盖项 | 内置 `ir/` 搜索路径 |
| `PCCL_NPU_NUMA_MAP` | 可选的逻辑 NUMA map，例如 `0,0,1,1,2,2,3,3` | 从 PCI sysfs 自动发现 |
| `PCCL_GPU_NUMA_MAP` | `PCCL_NPU_NUMA_MAP` 的旧别名 | 未设置 |
| `PCCL_EXECUTOR_TIMEOUT_MS` | Executor 等待和诊断超时时间 | `120000` |
| `PCCL_BOOTSTRAP_TIMEOUT_MS` | 多进程共享状态和 host-buffer bootstrap 超时时间 | `120000` |
| `PCCL_BARRIER_TIMEOUT_MS` | 多进程共享 barrier 超时时间 | `120000` |
| `PCCL_EXECUTOR_QUEUE_DEPTH` | 异步集合通信队列/doorbell slot 数量；限制在 1..512 | `512` |
| `PCCL_FORCE_DMA_LANES` | 强制 D2H/H2D host DMA lane 数，用于性能调优；超过编译上限时会截断 | 未设置 |
| `PCCL_STRIDED_PERF_ITERS` | strided 原生测试使用的测量迭代次数 | `5` |
| `PCCL_STRIDED_PERF_WARMUP` | strided 原生测试使用的 warmup 迭代次数 | `2` |
| `PCCL_PERF_ITERS` | `run_pcieccl_perf_matrix.py` 使用的迭代次数 | `10` |
| `PCCL_PERF_WARMUP` | `run_pcieccl_perf_matrix.py` 使用的 warmup 次数 | `2` |

通常不需要设置 `PCCL_NPU_NUMA_MAP`。未设置时，PCIeCCL 会从 sysfs 读取每个
NPU 的 PCI NUMA node，将物理 NUMA node 压缩为逻辑 ID，并选择
`allreduce_00.xml`、`allreduce_0011.xml` 或 `allreduce_00112233.xml` 等
IR 文件。

多进程模式使用共享的 2 MB hugepage host staging，并配合
`aclrtHostRegister`。运行 MPI 测试前，请在所选卡使用的物理 NUMA node 上
预留足够的 hugepage。

## Strided PCCL 原语

Strided 原生 PCCL API 支持分段缓冲区，其中每个 per-rank segment 都带有一个
前置保留前缀。PCCL 会跳过该前缀，仅操作其后的有效区域，并保留前缀内容不变。
这与另一条路径已经处理每个 segment 前 `stride` 个元素时所需的布局相同。

导出的 C API 如下：

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

参数语义：

- `stride` 表示每个 segment 开头保留的元素数量。它必须小于每个 segment 的
  元素数。`stride=0` 表示整个 segment 都是有效区域。
- `pcclAllGatherStrided`：`count` 是每个 rank 的 segment 元素数。send
  buffer 有一个 segment，receive buffer 有 `nranks` 个 segment。
- `pcclReduceScatterStrided`：`count` 是跨 `nranks` 个 segment 的 send
  buffer 总元素数，并且必须能被 `nranks` 整除。receive buffer 有一个
  `count / nranks` 个元素的 segment。
- `pcclAllToAllStrided`：`count` 是跨 `nranks` 个 send/receive segment 的
  总元素数，并且必须能被 `nranks` 整除。

外部 `INPUT`、`OUTPUT` 和 `SCRATCH` 设备缓冲区使用 strided 地址映射：

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

内部 staging buffer 是紧凑布局，不包含保留前缀形成的空洞。原生 strided API
不属于 HCCL 兼容层的一部分。

仓库内置的 strided 测试使用 `float` buffer，并覆盖四种有效区域比例：

| 用例 | `seg_count` | `stride` | 有效区域 |
|---|---:|---:|---:|
| `25% 有效区域` | `32M` floats | `24M` floats | 25% |
| `50% 有效区域` | `32M` floats | `16M` floats | 50% |
| `75% 有效区域` | `32M` floats | `8M` floats | 75% |
| `连续布局` | `32M` floats | `0` | 100% |

每个测试都会验证保留前缀仍为零，并验证有效区域包含预期的集合通信结果。计时行
报告平均延迟，以及仅基于有效字节数计算的算法带宽和总线带宽。

## 原生测试和 Benchmark

原生 PCCL strided 测试覆盖正确性，并仅使用有效字节数输出简单的计时和带宽指标：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/test_allgather_strided 2
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/test_reducescatter_strided 2
ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/test_alltoall_strided 2
```

AllToAll lane benchmark 用于测试消息大小和 host DMA lane 数的适配关系。每次进程
固定一个 `PCCL_FORCE_DMA_LANES` 值，默认测试每个 rank 1 / 4 / 8 / 16 / 32 / 64 /
128 MiB 的总消息量，并验证 AllToAll 的 rank/destination 数据模式：

```bash
for lanes in 1 4 8 12; do
  PCCL_STRIDED_PERF_ITERS=10 PCCL_STRIDED_PERF_WARMUP=2 \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    ./build/bench_alltoall_lanes 4 "$lanes" 128M
done

# 也可直接验证按 world size/消息量选择的自动策略。
PCCL_STRIDED_PERF_ITERS=10 PCCL_STRIDED_PERF_WARMUP=2 \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  ./build/bench_alltoall_lanes 4 auto 128M
```

lane 配置需要在不同进程中运行，因为 force 选项在 executor 进程内只解析一次。
比较时应保持拓扑、rank 数、dtype、warmup 和迭代次数一致。当前自动策略按 rank 数区分：
4 卡为 `<=4 MiB: 8`、`<=8 MiB: 12`、`<=16 MiB: 8`、`<=32 MiB: 1`、
`<=64 MiB: 8`、`>64 MiB: 1`；8 卡为 `<=1 MiB: 4`、`<=4 MiB: 12`、
`<=32 MiB: 4`、`<=64 MiB: 12`、`>64 MiB: 8`。其他 rank 数使用通用 fallback。
这些阈值来自当前机器的实测起点，最终取值应以目标拓扑上的 benchmark 结果为准。

单进程 benchmark 默认链接到原生 CANN HCCL，也可以通过 `LD_PRELOAD` 使用
PCIeCCL 运行。它覆盖使用 `fp32`/`sum` 的 AllReduce、AllGather 和
ReduceScatter。

```bash
# 原生 HCCL baseline.
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  ./build/bench_hccl_compat_sp 4 16M 512M 10 2 --devices=0,1,2,3

# 通过 HCCL 兼容层使用 PCIeCCL。
LD_PRELOAD=$PWD/build/libpccl.so ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  ./build/bench_hccl_compat_sp 4 16M 512M 10 2 \
  --init=root --rootinfo=zero --devices=0,1,2,3
```

PCIeCCL benchmark 路径请使用 `--init=root --rootinfo=zero`，因为兼容层实现的是
root-info communicator 初始化，而不是 `HcclCommInitAll`。

性能矩阵 runner 会在单进程和 MPI 模式下，对比原生 HCCL 与 PCIeCCL 在 2 / 4 /
8 卡上的 AllGather、ReduceScatter 和 AllReduce 性能：

```bash
PCCL_PERF_ITERS=10 PCCL_PERF_WARMUP=2 python3 ./run_pcieccl_perf_matrix.py
```

它会将原始日志写入 `perf_results/<timestamp>/logs`，生成解析后的
`results.json` 和带时间戳的报告，并更新 `PCIe_CCL_perf_report.md`。

## HCCL 兼容性

兼容层导出 `src/api/hccl_compat/hccl_hook.cpp` 中实现的 HCCL 符号：

- `HcclGetRootInfo`
- `HcclCommInitRootInfo`
- `HcclCommInitRootInfoConfig`
- `HcclCommDestroy`
- `HcclGetCommConfigCapability`
- `HcclAllReduce`
- `HcclAllGather`
- `HcclReduceScatter`
- `HcclAlltoAll`
- `HcclBroadcast`，`root=0`
- `HcclCommSetMemoryRange`
- `HcclCommUnsetMemoryRange`
- `HcclCommActivateCommMemory`
- `HcclCommDeactivateCommMemory`
- `HcclCommSymWinRegister`
- `HcclCommSymWinDeregister`

`HcclAllGather`、`HcclAlltoAll` 和 `HcclBroadcast(root=0)` 接受
`mapDataType` 实现的固定 count HCCL 数据类型映射：`int8`、`uint8`、
`int32`、`uint32`、`int64`、`uint64`、`fp16`、`fp32`、`fp64` 和 `bfp16`。
`HcclAlltoAll` 还要求 send/recv count 相等，并且 send/recv 类型相同。

兼容层未实现 `HcclCommInitAll`、`HcclReduce`、`HcclScatter`、
`HcclAllGatherV`、`HcclReduceScatterV`、`HcclAlltoAllV` 或
`HcclAlltoAllVC`。

可以使用 `LD_PRELOAD=build/libpccl.so`，也可以将 `build/libhccl.so` 放在原生
CANN HCCL 库之前的 `LD_LIBRARY_PATH` 中。

通过仓库内置 `hccl_test` 构建运行 MPI 的示例：

```bash
mpirun --allow-run-as-root -n 4 env \
  LD_PRELOAD=$PWD/build/libpccl.so \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  HCCL_TEST_USE_DEVS=0,1,2,3 \
  ./build/hccl_test/bin/all_reduce_test \
  -b 16M -e 512M -f 2 -d fp32 -o sum -p 4 -n 10 -w 2 -c 1
```

对于 `all_gather_test`，省略 `-o sum`。对于 `reduce_scatter_test`，保留
`-o sum`。对于 `alltoall_test`，使用相等的固定 count 和类型。对于
`broadcast_test`，使用 `-r 0`。
