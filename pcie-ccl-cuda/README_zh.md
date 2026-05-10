[English](README.md) | 中文

# PCCL — PCIe Collective Communication Library

PCCL 是一个针对 PCIe 互联优化的集合通信库，为多 GPU、多 NUMA 节点拓扑提供高性能集合操作。

核心特性：

- **IR 编程模型** — 通过 XML 中间表示声明式定义数据搬运模式，将算法设计与执行解耦
- **NUMA 感知执行** — 自动发现拓扑，按 NUMA 节点分配主机内存和工作线程，最小化跨 NUMA 流量
- **单进程/多进程双模式** — 支持单进程多线程和多进程（每进程一个 GPU）两种部署模式，通过 `pcclUniqueId` 自动切换
- **多后端支持** — 通过 HAL 抽象层同时支持 NVIDIA CUDA 和华为 Ascend CANN
- **vendor CCL 兼容 API** — 提供 NCCL / HCCL 兼容接口，可作为 drop-in 替换
- **Strided 集合操作** — 支持 MPCCL 多路径拆分场景（如 NVLink + PCIe 混合路径）

## 系统架构

PCCL 采用四层架构：

- **API 层** (`include/pccl/`, `src/api/`) — 对外接口，包括 PCCL 原生 C API 和 vendor CCL 兼容 API（NCCL / HCCL），提供集合操作和通信域管理
- **编排层** (`src/ir/`, `src/topology/`) — 加载 XML IR 程序，结合拓扑信息（NUMA 发现、GPU-NUMA 映射）编译为地址无关的 `CompiledGraph`，运行时再绑定真实 device/host 地址
- **执行层** (`src/executor/`, `src/memory/`) — 每个 rank 使用一个 `GraphExecutor` 轮询图节点并驱动 D2H / H2D / D2D CUDA 异步 lane；H2H 仍由常驻 NUMA worker pool 作为后端执行；通过 versioned progress 和 subchunk 级进度计数器实现流水线、读者消费游标和覆盖保护；管理主机内存池（单进程使用 `devMallocHost`，多进程使用共享 hugepage + `devHostRegister`）、跨 rank 共享状态（单进程使用堆内存，多进程使用 POSIX 共享内存）和 GPU 可见的完成标志。CUDA 后端使用 stream memory wait/write value 让用户 stream 等待 collective 完成，避免在等待路径上启动占用 SM 的自旋 kernel
- **HAL 层** (`src/hal/`) — 设备运行时抽象，通过 tag dispatch 适配 CUDA 和 Ascend CANN

执行流程：API 调用 → IR 加载 → 编译为 `CompiledGraph` → 运行时地址绑定 → `GraphExecutor` 轮询并提交可执行节点 → terminal 输出完成 → 流同步

## 多进程模式

PCCL 支持两种部署模式：

- **单进程模式（SP）** — 一个进程管理所有 GPU，通过多线程并行提交集合操作。传入全零的 `pcclUniqueId` 即自动使用此模式。主机内存通过 `devMallocHost` 分配，每个 NUMA 节点一块。
- **多进程模式（MP）** — 每个进程管理一个 GPU，通过共享内存协调。Rank 0 调用 `pcclGetUniqueId` 生成唯一 ID 后广播给所有 rank，PCCL 根据此 ID 建立跨进程共享。

### 多进程共享内存架构

多进程模式使用两层共享内存：

| 组件 | 用途 | 实现 |
|------|------|------|
| **SharedState** | 跨 rank 同步（进度计数器、barrier、chunk 状态） | POSIX `shm_open` + `mmap` |
| **HostMemoryPool** | 跨进程共享的主机 DMA 缓冲区 | `memfd_create` (MFD_HUGETLB) + `devHostRegister` |

Bootstrap 流程：
1. Creator（rank 0 所在进程）通过 `memfd_create` 分配 hugepage 内存，通过 bootstrap shm 共享 fd
2. Joiner（其他进程）通过 `/proc/<creator_pid>/fd/<N>` 打开 memfd，`mmap` 映射到本进程地址空间
3. 所有进程对共享缓冲区调用 `devHostRegister` 注册为 DMA 可访问
4. 通过 barrier 同步确保所有进程完成初始化后才开始通信

### Hugepage 配置

多进程模式使用 2MB hugepage 作为共享 DMA 缓冲区。需要在系统中预留足够的 hugepage：

```bash
# 查看当前 hugepage 状态
cat /proc/meminfo | grep Huge

# 预留 hugepage（每个 NUMA 节点需要 PCCL_HOST_BUFFER_SIZE，默认 4GB）
# 例如 4 NUMA 节点 × 4GB = 16GB = 8192 个 2MB hugepage
echo 8192 > /proc/sys/vm/nr_hugepages

# 或按 NUMA 节点分别预留（推荐，避免跨 NUMA 分配）
echo 2048 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 2048 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
echo 2048 > /sys/devices/system/node/node2/hugepages/hugepages-2048kB/nr_hugepages
echo 2048 > /sys/devices/system/node/node3/hugepages/hugepages-2048kB/nr_hugepages

# 持久化（重启后生效），在 /etc/sysctl.conf 中添加：
vm.nr_hugepages = 8192
```

若 hugepage 不足，`memfd_create` 会失败，多进程初始化将报错退出。

## 构建

### 前置依赖

- CMake >= 3.18
- C++14 编译器
- NVIDIA CUDA Toolkit 或 Ascend CANN（自动检测，也可手动指定）
- libnuma（必选，NUMA 感知内存分配和线程绑定的基础）
- librt（POSIX 共享内存，多进程模式需要）
- MPI（可选，仅多进程测试需要）

### 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

手动指定后端：

```bash
cmake .. -DPCCL_USE_CUDA=ON   # 强制使用 CUDA
cmake .. -DPCCL_USE_CANN=ON   # 强制使用 Ascend CANN
```

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PCCL_USE_CUDA` | OFF（自动检测） | 强制启用 CUDA 后端 |
| `PCCL_USE_CANN` | OFF（自动检测） | 强制启用 Ascend CANN 后端 |
| `PCCL_BUILD_NCCL_COMPAT` | ON | 构建 NCCL 兼容层 |
| `PCCL_TRACE` | OFF | 启用 Perfetto 兼容 Trace 输出（Chrome JSON 格式） |
| `PCCL_LOG_LEVEL` | 2 | 日志级别（0=DEBUG, 1=INFO, 2=WARN, 3=ERROR） |
| `PCCL_CUDA_ARCHITECTURES` | auto | CUDA kernel 架构；`auto` 会优先使用 CMake `native`，旧版 CMake 下尝试通过 `nvidia-smi` 检测，也可显式设为 `70;80`、`native`、`all`、`all-major` 或 `OFF` |
| `ASCEND_PRODUCT_TYPE` | Ascend310P1 | 昇腾芯片型号（Ascend910A, Ascend310P1, Ascend310B1, Ascend910B1, Ascend910B3） |
| `ASCEND_CORE_TYPE` | VectorCore | 昇腾核心类型（AiCore, VectorCore） |

### 安装

```bash
cmake --install build --prefix /usr/local
```

安装产物：
- 库文件 → `lib/`
- 头文件 → `include/pccl/`
- IR 文件 → `share/pccl/ir/`
- CMake 配置 → `lib/cmake/pccl/`

## 快速开始

**单进程模式**（一个进程管理所有 GPU，多线程并行提交）：

```cpp
#include <pccl/pccl.h>

devInit();

// 单进程模式：传入全零 UniqueId
pcclUniqueId id = {};
pcclComm_t comms[NRANKS];
devStream  streams[NRANKS];
for (int r = 0; r < NRANKS; ++r) {
    devSetDevice(r);
    pcclCommInitRank(&comms[r], NRANKS, id, r);
    devCreateStream(&streams[r]);
}

// 各线程并行提交集合操作
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

**多进程模式**（每个进程管理一个 GPU，通过 MPI 等方式启动）：

```cpp
#include <pccl/pccl.h>
#include <mpi.h>

MPI_Init(&argc, &argv);
int rank, nranks;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nranks);

devInit();
devSetDevice(rank);

// Rank 0 生成 UniqueId 并广播
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

完整示例参见 `tests/coll/` 目录下的测试程序（SP 版本和 `_mp` 后缀的 MP 版本）。

## API 参考

### 通信域管理

```c
// 生成唯一 ID（多进程模式下由 rank 0 调用后广播给所有 rank）
pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId);

// 创建通信域（首次调用时自动初始化进程级状态）
// uniqueId: 多进程模式传入 pcclGetUniqueId 生成的 ID；
//           单进程模式传入全零的 pcclUniqueId
pcclResult_t pcclCommInitRank(pcclComm_t* comm, int nranks, pcclUniqueId uniqueId, int rank);

// 销毁通信域
pcclResult_t pcclCommDestroy(pcclComm_t comm);

// 查询 rank 和 world size
pcclResult_t pcclCommRank(pcclComm_t comm, int* rank);
pcclResult_t pcclCommSize(pcclComm_t comm, int* size);

// 进程级清理（销毁所有共享资源）
pcclResult_t pcclFinalize(void);
```

### 集合操作

所有集合操作均为异步，提交到指定设备流后立即返回，通过流同步等待完成。

```c
// AllReduce: 对所有 rank 的数据求和，结果广播到所有 rank
// count — 元素个数
pcclResult_t pcclAllReduce(
    const void* sendbuff, void* recvbuff,
    size_t count, pcclComm_t comm, pcclStream_t stream);

// AllGather: 每个 rank 发送 sendcount 个元素，接收所有 rank 的拼接结果
// sendcount — 每个 rank 发送的元素个数
pcclResult_t pcclAllGather(
    const void* sendbuff, void* recvbuff,
    size_t sendcount, pcclComm_t comm, pcclStream_t stream);

// ReduceScatter: 对所有 rank 的数据求和后，将结果分散到各 rank
// recvcount — 每个 rank 接收的元素个数
pcclResult_t pcclReduceScatter(
    const void* sendbuff, void* recvbuff,
    size_t recvcount, pcclComm_t comm, pcclStream_t stream);

// AllToAll: 每个 rank 向其他所有 rank 发送不同的数据段
// count — 每个 rank 发送/接收的总元素个数
pcclResult_t pcclAllToAll(
    const void* sendbuff, void* recvbuff,
    size_t count, pcclComm_t comm, pcclStream_t stream);
```

### 类型

```c
// 多进程协调用的唯一标识（128 字节，值类型）
#define PCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[PCCL_UNIQUE_ID_BYTES]; } pcclUniqueId;
```

### 错误处理

```c
// 返回值类型
typedef enum {
    pcclSuccess          = 0,
    pcclInvalidArgument  = 1,
    pcclSystemError      = 2,
    pcclInternalError    = 3,
    pcclDeviceError      = 4,
    pcclNetworkError     = 5,
    pcclTimeout          = 6
} pcclResult_t;

// 错误码转字符串
const char* pcclGetErrorString(pcclResult_t result);
```

## IR 语义

### 概述

PCCL 使用 XML 格式的中间表示（IR）来声明式地描述集合操作的数据搬运编排。每个 IR 程序为每个 rank 定义一组有序指令，指定数据在设备内存、主机内存和 NUMA 节点之间的移动方式。

IR 程序文件位于 `ir/` 目录，按 `<集合操作>_<GPU数>gpu_<NUMA数>numa.xml` 命名。

### Segment 与 Chunk

IR 的数据模型围绕两个核心概念组织：

**Segment（段）** 是设备缓冲区中按 rank 划分的连续数据区域。不同集合操作的缓冲区有不同的 segment 结构：

| 集合操作 | sendbuf segment 数 | recvbuf segment 数 | 每个 segment 大小 |
|----------|-------------------|-------------------|------------------|
| AllReduce | 1 | 1 | `count` 个元素 |
| AllGather | 1 | nranks | `sendcount` 个元素 |
| ReduceScatter | nranks | 1 | `recvcount` 个元素 |
| AllToAll | nranks | nranks | `count / nranks` 个元素 |

**Chunk（块）** 是 IR 指令操作的数据单元，是 segment 的进一步细分。每条 IR 指令（D2H、H2D、H2H 等）搬运恰好一个 chunk 的数据。一个 segment 可以包含一个或多个 chunk，chunk 大小由编译器根据 segment 大小和 IR 程序中的 chunk 数量自动计算：`chunk_size = pcie_per_segment / chunks_per_segment`。

在设备侧，chunk 通过索引映射到设备缓冲区的物理地址。在主机侧，每个 chunk 由 `(numa_id, chunk_idx)` 二元组唯一标识，对应主机内存池中的一块临时缓冲区。主机 chunk 的数量通常多于设备 chunk，因为归约等操作需要额外的中间存储空间。

在执行时，每个 chunk 还会被进一步切分为 1MB 的 **subchunk**，以实现 D2H/H2D 传输与 H2H 计算的流水线重叠。

### 操作码

| 操作码 | 语义 |
|--------|------|
| `D2D` | 设备内部拷贝（如 GPU 内 D2D 传输） |
| `D2H` | 设备到主机传输（GPU → 主机，经 PCIe） |
| `H2D` | 主机到设备传输（主机 → GPU，经 PCIe） |
| `H2H` | 主机间内存拷贝（`dst = src`） |
| `REDUCE` | 三地址主机归约（`dst = src1 + src2`），结果写入新 chunk |

### 指令格式

每条指令包含以下字段：

| 字段 | 说明 |
|------|------|
| `op` | 操作码（见上表） |
| `src_chunk` / `src_numa` | 源 chunk 索引和 NUMA 节点 ID（H2H/H2D 使用） |
| `dst_chunk` / `dst_numa` | 目标 chunk 索引和 NUMA 节点 ID（D2H/H2H/REDUCE 使用） |
| `src1_chunk` / `src1_numa` | REDUCE 第一源操作数 |
| `src2_chunk` / `src2_numa` | REDUCE 第二源操作数 |

对于 D2H 指令，`src_chunk` 引用设备侧 chunk 索引；对于 H2D 指令，`dst_chunk` 引用设备侧 chunk 索引。主机侧 chunk 由 `(numa_id, chunk_idx)` 二元组唯一标识。

**单写者模型：** 每个主机 chunk 恰好有一个写入者。归约操作不做原地累加（`dst += src`），而是将结果写入新的 chunk（`dst = src1 + src2`）。多输入归约通过二叉树分解实现（例如 4 输入拆为两级：两次两两归约 + 一次合并）。这一设计消除了多写者流水线同步的复杂性，所有依赖通过 SpinWait 源 chunk 的进度计数器解决。

### XML 格式示例

以 2-GPU 单 NUMA 节点的 AllReduce 为例（`ir/allreduce_00.xml`）：

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

执行流程：
1. Rank 0 和 Rank 1 分别通过 D2H 将各自的数据传到主机 chunk 0 和 chunk 1
2. Rank 0 通过 REDUCE 将 chunk 0 和 chunk 1 归约到新的 chunk 2（`hc2 = hc0 + hc1`）
3. 两个 rank 通过 H2D 将归约结果从 chunk 2 写回各自的 GPU

### 编译流程

IR 程序的编译由 `Compiler` 完成（`src/ir/compiler.hpp`），流程如下：

1. **IR 加载** — `IRLoader` 根据集合操作名称和拓扑自动定位并解析 XML 文件
2. **地址映射** — 将 IR 中的 chunk 索引映射到物理设备地址和主机内存地址（支持 strided 寻址）
3. **子分块切片** — 将大块传输切分为 1MB 子块（subchunk），实现流水线传输
4. **生成执行图** — 输出地址无关的 `CompiledGraph`，节点之间由 versioned chunk progress 建立依赖；运行时根据本次 collective 的真实地址生成 `RuntimeBinding`

## Strided 集合操作

### 背景

当 MPCCL 将一个集合操作拆分为快速路径（如 NVLink）和 PCIe 路径时，快速路径处理每个 segment 的前 `stride` 个元素，PCCL 处理其余部分。由于 PCCL 需要在每个 segment 中跳过快速路径区域，这就产生了非连续的 strided 访问模式。

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

### 参数语义

- **`count`** — 含义因集合操作而异：
  - `AllGatherStrided`: per-segment 元素个数（= sendcount）
  - `ReduceScatterStrided`: 总元素个数（= recvcount × nranks），内部除以 nranks 得到 seg_count
  - `AllToAllStrided`: 总元素个数（= per-rank count × nranks），内部除以 nranks 得到 seg_count
- **`stride`** — 每个 segment 中由快速路径处理的前导元素个数。PCCL 操作 `(seg_count - stride)` 个元素。设为 0 等同于非 strided 版本。

per-segment 大小包含快速路径和 PCIe 两部分：`seg_count = count`（或 `count / nranks`）。

### 地址计算公式

```
segment_size   = seg_count * elem_size
stride_offset  = stride * elem_size
pcie_per_seg   = segment_size - stride_offset
chunks_per_seg = device_chunks / num_segments
chunk_size     = pcie_per_seg / chunks_per_seg

addr = base + seg * segment_size + stride_offset + chunk_in_seg * chunk_size
       其中 seg          = chunk_idx / chunks_per_seg
            chunk_in_seg = chunk_idx % chunks_per_seg
```

`INPUT` / `OUTPUT` / `SCRATCH` 使用上述 strided 地址映射。PCCL 内部临时 `INTERNAL` device buffer 是紧凑分配的 staging 区，不包含快速路径 hole，因此内部 chunk 使用连续地址映射；这对 ReduceScatterStrided 中的临时归约尤其重要。

### 示例：AllGatherStrided

场景：4 个 rank，`count = 64M`（元素），`stride = 48M`（元素），`elem_size = 4B`（float32）。

**基本参数推导：**

- `seg_count = count = 64M` 元素（AllGather 的 count 即 per-segment 元素数）
- `segment_size = seg_count × elem_size = 64M × 4B = 256MB`
- `stride_offset = stride × elem_size = 48M × 4B = 192MB`（快速路径区域）
- `pcie_per_seg = segment_size - stride_offset = 256MB - 192MB = 64MB`（PCCL 负责的区域）

**sendbuf（输入，1 个 segment）：**

sendbuf 存放当前 rank 自身的数据，只有 1 个 segment，共 64M 元素（256MB）。其中前 48M 元素（192MB）由快速路径处理，PCCL 从偏移 192MB 处读取后 16M 元素（64MB）。

**recvbuf（输出，4 个 segment）：**

recvbuf 存放所有 rank 的拼接结果，共 4 个 segment（对应 rank 0~3），总大小 1024MB。每个 segment 256MB，其中前 192MB 由快速路径写入，PCCL 负责写入后 64MB。

PCCL 写入 chunk k（对应 rank k）的地址为 `recvbuf + k × 256MB + 192MB`。例如 chunk 2 的写入地址为 `recvbuf + 512MB + 192MB = recvbuf + 704MB`。

### 测试覆盖

`test_allgather_strided`、`test_reducescatter_strided`、`test_alltoall_strided` 会覆盖以下配置：

| 配置 | `seg_count` | `stride` | PCCL 负责的数据比例 |
|------|-------------|----------|---------------------|
| `75% fast-path` | `32M` float | `24M` float | 25% |
| `50% fast-path` | `32M` float | `16M` float | 50% |
| `25% fast-path` | `32M` float | `8M` float | 75% |
| `contiguous` | `32M` float | `0` | 100% |

这些测试同时验证 fast-path 区域保持不变、PCCL 区域结果正确，并输出与普通集合通信相同格式的延迟、算法带宽和总线带宽。

## 性能追踪（Perfetto Trace）

PCCL 内置了基于 Chrome Trace Event Format 的性能追踪系统，输出可直接在 [Perfetto UI](https://ui.perfetto.dev) 中查看的 JSON 文件。

### 启用

```bash
cmake .. -DPCCL_TRACE=ON
make -j$(nproc)

# 运行时通过环境变量指定输出路径
PCCL_TRACE_FILE=trace.json ./test_allgather 4
```

### 追踪内容

生成的 Trace 包含以下信息，按独立的 Perfetto Track 分行展示：

| Track | 说明 | 计时方式 |
|-------|------|----------|
| `Rank N D2H` | D2H 执行器 CPU 线程 | 批次总时长 |
| `Rank N D2H Stream K` | GPU D2H DMA 传输 | Event Query + Per-Stream 链式计时 |
| `Rank N H2D` | H2D 执行器 CPU 线程 | 批次总时长 |
| `Rank N H2D Stream K` | GPU H2D DMA 传输 | Event Query + Per-Stream 链式计时 |
| `H2H NUMA M/#W` | H2H 工作线程 | 逐 subchunk 精确计时 |

每个 Track 的 tid 按固定规则分配，保证在 Perfetto 中按 Rank → D2H/H2D → H2H NUMA 的顺序排列。

### GPU DMA 计时原理

CPU 提交 DMA 后通过 `devRecordEvent` 在 GPU stream 上标记完成点，随后通过 `devQueryEvent` 轮询检测完成。对同一条 stream 上的多个 DMA，使用 **链式计时** 避免时间重叠：

```
begin = max(submit_ts, 同 stream 上前一个 DMA 的 complete_ts)
end   = devQueryEvent 返回 Complete 时的 CPU 时间
```

### H2H 事件详情

H2H Copy/Reduce 事件的 args 字段包含：

| 字段 | 说明 |
|------|------|
| `bw_gbps` | 该 subchunk 的即时带宽（GB/s） |
| `src1_numa` / `dst_numa` | 源/目标 NUMA 节点 |
| `bytes` | 传输字节数 |
| `sc_idx` | Subchunk 索引 |
| `cpu_us` | CPU 时间（微秒） |
| `cpu_before` / `cpu_after` | 执行前后的 CPU 核心编号 |

在 Perfetto 中点击任意 H2H 事件色块，底部面板的 Arguments 表格中可查看这些字段。

### 设计特点

- **零外部依赖** — 纯 C++14 实现，不依赖 Perfetto SDK 或 protobuf
- **线程局部缓冲** — 每个线程独立的事件 shard，热路径无锁（仅 `vector::push_back` + 时间戳读取）
- **统一后端** — 所有平台（CUDA / CANN）使用同一套追踪系统
- **运行时控制** — 编译时 `-DPCCL_TRACE=ON` 启用，运行时通过 `PCCL_TRACE_FILE` 环境变量控制是否实际输出（未设置则不产生任何开销）
- **多进程对齐** — 每个 Trace 文件记录 `wall_epoch_us`（墙钟时间戳），多进程模式下可在 Perfetto 中对齐不同 rank 的时间线

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PCCL_TRACE_FILE` | Trace 输出路径，设置后启用追踪 | 不输出 |
| `PCCL_GPU_NUMA_MAP` | GPU 到 NUMA 节点的映射，如 `"0,0,1,1"` | 自动检测 |
| `PCCL_HOST_BUFFER_SIZE` | 每个 NUMA 节点的主机缓冲区大小（字节） | 4 GB |
| `PCCL_IR_PATH` | IR XML 文件搜索路径 | `./ir`, `/usr/local/share/pccl/ir` |
| `PCCL_IR_DIR` | `PCCL_IR_PATH` 的兼容别名 | `./ir`, `/usr/local/share/pccl/ir` |
| `PCCL_NRANKS` | rank 总数 | — |
| `PCCL_BOOTSTRAP_ADDR` | 多节点引导地址，格式 `"host:port"` | 单节点模式 |
| `PCCL_D2D_PUSH` | remote D2D 使用 source rank 主动写入 destination rank；设为 `0` 时改为 destination rank 从 source rank 读取 | `1` |
| `PCCL_D2D_INPUT_FIRST` | 覆盖 remote D2D 图节点调度策略；设为 `1` 时优先调度读取 INPUT 的 D2D 节点 | collective 默认策略 |
| `PCCL_BARRIER_TIMEOUT_MS` | 共享状态 barrier 等待超时，超时后返回 `pcclTimeout` 并打印诊断 | `120000` |
| `PCCL_EXECUTOR_TIMEOUT_MS` | 图执行器等待空闲、CUDA event、H2H completion、runGraph 空转等诊断超时 | `120000` |
| `PCCL_REMOTE_BUFFER_TIMEOUT_MS` | 多进程 CUDA IPC 模式下等待远端 rank 发布 device buffer 的超时 | `120000` |
| `PCCL_CUDA_IPC_CLOSE_ON_FINALIZE` | finalize 时是否强制调用 `cudaIpcCloseMemHandle` 关闭远端 CUDA IPC 映射；非 `0` 表示开启 | `0` |
| `PCCL_NCCL_COMPAT_MODE` | NCCL 兼容层 SP/MP 模式选择：`auto`、`sp`/`single`、`mp`/`multi` | `auto` |
| `PCCL_NCCL_COMPAT_MODE_TIMEOUT_MS` | NCCL 兼容层自动探测 SP/MP 模式时等待 rendezvous 共享内存和 rank 状态的超时 | `120000` |
| `PCCL_NCCL_COMPAT_MODE_GRACE_MS` | NCCL 兼容层自动探测中，未观察到其他进程 PID 时选择单进程模式前的宽限时间 | `1000` |
| `PCCL_NCCL_COMPAT_HOST_FALLBACK_BYTES` | NCCL 兼容层 multi-process 小消息 host fallback 阈值 | 由兼容层默认值决定 |
| `PCCL_NCCL_COMPAT_HOST_COLL_TIMEOUT_MS` | NCCL 兼容层 host fallback 集合操作等待超时 | `120000` |
| `PCCL_NCCL_COMPAT_BCAST_STAGE_BYTES` | NCCL 兼容层 multi-process Broadcast host staging 分块大小 | `4194304` |
| `PCCL_NCCL_COMPAT_BCAST_TIMEOUT_MS` | NCCL 兼容层 multi-process Broadcast 等待超时 | `120000` |

### pcclConfig_t

```c
typedef struct {
    int nranks;                 // rank 总数
    int device_count;           // 本地 GPU/NPU 数量
    const int* gpu_ids;         // 设备 ID 列表（NULL = 自动）
    const char* bootstrap_addr; // 多节点引导地址（NULL = 单节点）
    size_t host_buffer_size;    // 每个 NUMA 节点的主机缓冲区大小（0 = 自动）
    const char* gpu_numa_map;   // GPU-NUMA 映射字符串（NULL = 自动检测）
    const char* ir_path;        // IR XML 搜索路径（NULL = 默认）
} pcclConfig_t;
```

## 测试

构建完成后，测试可执行文件位于 `build/` 目录：

```bash
cd build

# 单进程集合操作测试（参数为 rank 数）
./test_allgather 4
./test_allreduce 4
./test_reducescatter 4
./test_alltoall 4

# Strided 变体
./test_allgather_strided 4
./test_reducescatter_strided 4
./test_alltoall_strided 4

# 多进程集合操作测试（需要 MPI）
mpirun -np 4 ./test_allgather_mp
mpirun -np 4 ./test_allreduce_mp

# 带 Trace 输出
PCCL_TRACE_FILE=trace.json ./test_allgather 4
# 在 https://ui.perfetto.dev 打开 trace.json 查看
```

测试覆盖范围：

- **集合操作**: AllReduce, AllGather, ReduceScatter, AllToAll
- **部署模式**: 单进程多线程（SP）、多进程（MP，需 MPI）
- **GPU 配置**: 2 GPU, 4 GPU, 8 GPU
- **NUMA 拓扑**: 1 NUMA, 2 NUMA, 4 NUMA
- **Strided 变体**: AllGather / ReduceScatter / AllToAll 的 strided 版本
- **Strided 拆分比例**: 75% / 50% / 25% fast-path 以及 contiguous，对 AllGather / ReduceScatter / AllToAll 做正确性和带宽验证

每个测试执行正确性验证和带宽基准测试，报告算法带宽和总线带宽。

### 使用 nccl-tests 测试

PCCL 提供 NCCL 兼容层，可通过 `LD_PRELOAD` 直接使用 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 进行测试：

当前 NCCL 兼容层覆盖 Megatron / PyTorch ProcessGroupNCCL 常见调用路径：

- 通信域生命周期：`ncclGetUniqueId`、`ncclCommInitRank`、`ncclCommInitRankConfig`、`ncclCommInitAll`、`ncclCommSplit`、`ncclCommDestroy`、`ncclCommAbort`、`ncclCommFinalize`
- 查询和注册：`ncclCommCount`、`ncclCommUserRank`、`ncclCommCuDevice`、`ncclCommGetAsyncError`、`ncclCommRegister`、`ncclCommDeregister`、`ncclMemAlloc`、`ncclMemFree`
- group 语义：`ncclGroupStart`、`ncclGroupEnd`、`ncclGroupSimulateEnd`
- 集合通信：`ncclAllReduce`、`ncclAllGather`、`ncclReduceScatter`、`ncclBroadcast` / `ncclBcast`、`ncclReduce`
- dtype：`int8`、`uint8`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`float32`、`float64`、`bfloat16`
- reduce op：`sum`、`max`、`min`

copy-only 集合按字节搬运；规约集合按 dtype 和 reduce op 分发到 typed device reduce kernel。16-bit 浮点规约依赖 device reduce 路径；如果某个 IR 需要 host-side `H2HReduce`，兼容层会拒绝该 16-bit 路径。

注意：`ncclSend` / `ncclRecv` 仍未实现。`ncclCommSplit` 当前用于创建可用的新通信域，尚不完整覆盖 NCCL 的所有 `color` / `key` 排序语义。

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

# MPI AllGather（每进程 1 GPU）
mpirun --mca coll_hcoll_enable 0 -np 4 env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PCCL_IR_PATH=/workspace/pcieccl/ir PCCL_LOG_LEVEL=3 \
  LD_PRELOAD=/workspace/pcieccl/build/libpccl.so \
  /workspace/nccl-tests/build_mpi/all_gather_perf_mpi -b 4M -e 256M -f 2 -g 1 -p 1 -w 1 -n 5
```

**环境变量说明：**

- `PCCL_IR_PATH` — 指向 IR XML 文件目录
- `PCCL_LOG_LEVEL=3` — 设为 ERROR 级别，减少输出噪音
- `LD_PRELOAD` — 通过预加载 PCCL 库拦截 NCCL API 调用

**nccl-tests 参数说明：**

| 参数 | 说明 |
|------|------|
| `-b 4M -e 256M -f 2` | 数据量从 4M 到 256M，每步乘 2 |
| `-g 2` | 使用 2 个 GPU |
| `-p 1` | 并行初始化（必须启用，由于 nccl-tests 本身的 Bug） |
| `-w 1 -n 5` | 1 次 warmup，5 次测量 |

### Megatron 端到端 Benchmark

Megatron 端到端测试用于验证 PCCL 的 NCCL 兼容层在 PyTorch `ProcessGroupNCCL` 路径下是否能跑通，并观察真实训练迭代耗时。当前推荐的大消息 benchmark 配置使用 `/workspace/megatron`，4 个进程分别绑定 GPU `0,2,4,6`，TP=4、PP=1、mock data、fp16、`transformer_engine`。

该配置主要覆盖 Megatron TP 中的大消息 AllGather / ReduceScatter / AllReduce 路径；不要使用 `torch_dist_smoke/run_megatron.sh` 作为正式性能结论来源，那个脚本是较小模型的 smoke test，默认还包含 `--sequence-parallel`。

**公共配置：**

| 项 | 值 |
|----|----|
| Megatron 路径 | `/workspace/megatron` |
| GPU | `CUDA_VISIBLE_DEVICES=0,2,4,6` |
| 进程数 | `torch.distributed.run --nproc_per_node=4` |
| 并行 | `--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1` |
| 模型 | `--num-layers 4 --hidden-size 8192 --ffn-hidden-size 28672 --num-attention-heads 64` |
| 序列 | `--seq-length 4096 --max-position-embeddings 4096` |
| batch | `--micro-batch-size 1 --global-batch-size 4` |
| 精度 | `--fp16` |
| Transformer 实现 | `--transformer-impl transformer_engine` |
| 数据 | `--mock-data --tokenizer-type NullTokenizer --vocab-size 8192` |
| 迭代 | `--train-iters 12 --log-interval 1` |
| 后端参数 | `--distributed-backend nccl` |
| 不使用 | 不加 `--sequence-parallel` |
| CUDA connection | `CUDA_DEVICE_MAX_CONNECTIONS=8` |

`CUDA_DEVICE_MAX_CONNECTIONS=8` 是当前 PCCL Megatron benchmark 的推荐设置。`CUDA_DEVICE_MAX_CONNECTIONS=1` 会限制硬件连接数，和 PCCL 这种 CPU 编排、多 CUDA stream 的执行方式不匹配；Megatron / TransformerEngine 仍可能打印建议设置为 1 的 warning，可以忽略。

**构建待测库：**

```bash
cd /workspace/pcieccl-noctx
cmake --build build_quiet -j$(nproc)
```

如需和干净 HEAD 对比，可单独建一个 worktree，避免当前工作树的未提交改动影响 HEAD 基线：

```bash
HEAD_ROOT=/workspace/pcieccl_no_fastpath_head_$(git rev-parse --short HEAD)
git worktree add "$HEAD_ROOT" HEAD
cmake -S "$HEAD_ROOT" \
      -B "$HEAD_ROOT/build_quiet" \
      -DPCCL_USE_CUDA=ON -DPCCL_BUILD_NCCL_COMPAT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$HEAD_ROOT/build_quiet" -j$(nproc)
```

**推荐启动器：**

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

**运行当前版本、关闭 fastpath、HEAD 或原生 NCCL：**

```bash
# 当前工作树，small-message fastpath 默认开启
run_megatron_bench current_fastpath /workspace/pcieccl-noctx build_quiet on \
  /workspace/megatron/bench_results/megatron_0246

# 当前工作树，显式关闭 small-message fastpath，用于确认 fastpath 以外的大消息路径
run_megatron_bench current_fastpath_disabled /workspace/pcieccl-noctx build_quiet off \
  /workspace/megatron/bench_results/megatron_0246

# 干净 HEAD worktree；如果 fastpath 修改尚未提交，它就是 no-fastpath 基线
HEAD_ROOT=${HEAD_ROOT:-/workspace/pcieccl_no_fastpath_head_1d2de024}
run_megatron_bench head_clean "$HEAD_ROOT" build_quiet on \
  /workspace/megatron/bench_results/megatron_0246

# 原生 NCCL，不 preload PCCL
run_megatron_bench native_nccl /workspace/pcieccl-noctx build_quiet nccl \
  /workspace/megatron/bench_results/megatron_0246
```

如果使用的 HEAD worktree 目录不是 `/workspace/pcieccl_no_fastpath_head_1d2de024`，把 `HEAD_ROOT` 换成实际 worktree 路径即可。对比 current fastpath-disabled 和干净 HEAD 时，二者都应该走大消息 IR/GraphExecutor 路径；`PCCL_SMALL_MSG_FASTPATH=0` 只用于关闭当前版本的小消息 fastpath 初始化和路由。

**结果汇总：**

Megatron 的耗时行格式为 `elapsed time per iteration (ms): ...`。通常跳过第 1 个 warmup iteration，统计 iteration 2-12：

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

正确性至少检查最后一轮的 `lm loss` 和 `grad norm` 是否一致，并确认进程退出码为 0：

```bash
rg "iteration +12/ +12|lm loss|grad norm|Traceback|RuntimeError|internal error|timed out" \
  /workspace/megatron/bench_results/megatron_0246/*.log
cat /workspace/megatron/bench_results/megatron_0246/*.status
```

在当前 0246 配置下，稳定性判断建议看多次交替运行后的 p50 和 avg。单次 run 的 tail iteration 可能带来约 1% 内的波动；如果需要判断小幅回归，按 `HEAD -> current disabled -> current fastpath` 的顺序交替跑 3-5 组再比较。
