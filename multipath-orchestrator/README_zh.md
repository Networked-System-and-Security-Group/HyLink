[English](README.md) | 中文

# Adaptive-CCL 设计说明

本文描述自适应多路径集合通信层（AMP-CCL）的完整架构、调用链，以及与 PCIeCCL（pcieccl / PCCL）的集成方式。

**流同步与双路径专述**（用户流 / `pcie_stream`、混合与纯路径、AllGather 分支与 V 原语）：见 **[STREAM_SYNC.md](STREAM_SYNC.md)**。

---

## 1. 概述与目标

AMP-CCL 通过 **LD_PRELOAD** **透明地**拦截 NCCL / HCCL 集合通信 API，并将每次集合操作拆成两条并行路径：

- **快路径（Fast Backend）**：走 NCCL / HCCL 原生链路（如 NVLink、IB、HCCS 等）。
- **PCIe 路径（PCIe Backend）**：走 PCIe 集合库（当前实现为 **PCIeCCL / PCCL**）。

目标：

- 对应用 **透明**：无需改代码，仅通过 LD_PRELOAD 注入。
- 按通信域（拓扑 + 通信器） **自适应拆分**；反馈控制调节快路径与 PCIe 路径之间的字节比例（alpha）。
- **多卡参数全局一致**：所有 rank 看到相同参数表；仅 Rank 0 根据 **全局**计时（经共享内存聚合）更新参数。
- PCIe 不划算时回退为仅快路径。

---

## 2. 架构

```
用户应用
    |
NCCL / HCCL API（集合、流同步等）
    |
LD_PRELOAD 注入
    |
AMP-CCL
├── Hook 层（nccl_hook / hccl_hook）
│   └── 拦截 CommInit、CommDestroy、AllReduce、AllGather、SynchronizeStream 等
├── 虚拟集合层（VirtualCollective）
│   ├── 参数来源：单卡 → 本地 ParamCache；多卡 → 共享内存 ShmParamStore
│   ├── Planner 生成分拆方案（fast_bytes / pcie_bytes）
│   ├── 快路径：FastBackend → 原始 NCCL/HCCL
│   └── PCIe 路径：PCIeBackend → PCIeCCL（pcclCommInitRank、pcclSubmit、Strided 集合等）
├── 流收尾（OnStreamSynchronized，经 host callback）
│   └── 集合返回前在 user stream 上 `StreamWaitEvent`(PCIe) + `LaunchHostFunc`；回调内消费 Pending、读 `ElapsedSeconds`、写 shm 或本地 Update（见 [STREAM_SYNC.md](STREAM_SYNC.md)）
└── 通信域管理（DomainManager + CommDomain）
    └── 每个 raw comm 一个 CommDomain；LogicalCommKey 用于 node-scope 与共享参数 shm
```

- **通信器的创建与销毁** 仍由 NCCL / HCCL 完成；我们把 **raw_comm → CommDomain** 映射到本地运行态，并在域内维护参数表与 PCIe 资源（pcie_comm、pcie_stream 等）。跨 rank 共享状态不复用整个 CommDomain，而是通过 **LogicalCommKey** 连接 node-scope shm 与参数 shm。

---

## 3. 模块与目录结构

```
libampccl/
├── hook/
│   ├── nccl_hook.cc      # 拦截 NCCL 集合与 Comm 符号（不导出 cudaStreamSynchronize）
│   └── hccl_hook.cc      # 拦截 HCCL 集合与 Comm 符号（不导出 aclrtSynchronizeStream）
├── core/
│   ├── domain_key.h      # CommDomainKey：本地 domain 标识，可包含当前 rank
│   ├── logical_comm_key.h # LogicalCommKey：全 rank 一致的 communicator 共享标识
│   ├── node_scope.h/cc   # 本节点 rank 注册与 intra-node communicator 检测
│   ├── domain.h          # CommDomain（key、logical_key、param_cache、controller、PCIe 状态、计时器、ShmParamStore）
│   ├── domain_manager.h  # DomainManager：raw_comm↔CommDomain、stream↔PendingCollective
│   ├── comm_init.h/cc    # BuildKey/BuildLogicalKey、InitPCIeForDomain（pcclCommInitRank+pcclUniqueId、设备流创建）
│   ├── virtual_collective.h  # VirtualCollective：AllReduce/AllGather 等入口与拆分执行
│   ├── planner.h         # Planner：由 alpha、use_pcie 生成 Plan（fast_bytes、pcie_bytes）
│   ├── stream_sync.h/cc  # OnStreamSynchronized：消费 Pending、同步计时器、写统计或 Update
│   ├── stream_host_callback.h/cc  # EnqueueCollectiveCompletionCallback：LaunchHostFunc / aclrtLaunchHostFunc
│   └── shm_store.h/cc    # ShmParamStore：shm 布局、Attach、WriteMyStat、ReadParams、WriteParams
├── controller/
│   ├── controller.h      # AdaptiveController（SuggestAlpha、Update）
│   ├── algo_base.h       # AdaptiveAlgo 接口
│   ├── algo_factory.h    # 按配置选择算法（inverse / split_ratio / STATIC）
│   ├── algo_inverse_proportion.h
│   ├── algo_split_ratio_control.h
├── backend/
│   ├── backend_base.h    # BackendResult、模板 BackendBase
│   ├── fast_backend.h/cc # FastBackendImpl：直接调 NCCL/HCCL
│   ├── memory/           # 面向 NCCL/HCCL 的内存接口
│   ├── dtype/            # 编译期数据类型特征（NCCL/HCCL）
│   └── pcie_backend.h/cc # PCIeBackendImpl：PCCL Strided + pcclSubmit（域内 pcie_comm、pcie_stream）
├── cache/
│   └── param_cache.h     # ParamCache（OpKey → ParamValue），GetAll/SetFrom 供 shm 序列化
├── telemetry/
│   ├── timer.h           # Timer（CUDA/ACL 事件或 CPU），Start(stream)、Stop(stream)、Synchronize()
│   └── stats.h           # ExecStat（fast_time、pcie_time、bytes、success）
├── common/
│   ├── op_key.h          # OpKey（op、bytes、datatype）、CollectiveType
│   ├── config.h          # AMPCCL_ENABLE、AMPCCL_ALGO、AMPCCL_MIN_*、AMPCCL_ENABLE_PCIE 等
│   └── log.h             # 日志级别与 AMPCCL_LOG
```

---

## 4. 通信域（LogicalCommKey、CommDomainKey、CommDomain、DomainManager）

### 4.1 通信域标识：LogicalCommKey 与 CommDomainKey

当前实现把「共享 identity」和「本地运行态 identity」分开：

- **LogicalCommKey**：communicator 级共享标识，所有属于同一个逻辑 communicator 的 rank 得到相同 `world_size` 与 `comm_hash`。它用于 **node-scope shm** 与 **参数 shm** 命名。
  - `ncclCommInitRank`：由 `ncclUniqueId` 派生。
  - `HcclCommInitRootInfo`：由 `HcclRootInfo` 派生。
  - `CommInitAll`：没有跨进程 bootstrap，使用本进程内递增 nonce，保证同一次 InitAll 的本地 ranks 共享 key，同时避免同规模重复 InitAll 撞 key。
- **CommDomainKey**：本地 domain 标识，可包含当前 rank；主要用于日志与本地语义，不用于跨 rank shm 命名。

**我们不管理 NCCL/HCCL 的通信器列表**；通信器的创建与销毁由 NCCL/HCCL 完成。AMP-CCL 只维护 raw comm 到本地 CommDomain 的映射。

### 4.2 CommDomain（per raw comm）

每个进程对每个在用的 raw communicator 有一个独立 **CommDomain**。这点很重要：CommDomain 内有 `pcie_comm`、`pcie_stream`、timer、临时缓冲等 rank/comm 私有运行态，不能按 logical key 在不同 raw comm 之间复用。

- **key**：本地域标识。
- **logical_key**：共享 communicator 标识，用于 node-scope 和参数 shm。
- **is_intra_node / allow_ampccl**：CommInit 时缓存的 node-scope 决策。
- **param_cache**：参数表；同一个 communicator 生命周期内多个 iter 复用。多卡时通过 LogicalCommKey 对应的共享内存同步。
- **controller**：自适应算法（SuggestAlpha、Update）；多卡时仅 Rank 0 更新参数。
- **PCIe（CommInit 后由 InitPCIeForDomain 设置）**：
  - `pcie_comm`：PCCL 的 `pcclComm_t`（来自 `pcclCommInitRank`）。
  - `pcie_rank`、`pcie_nranks`：本 rank 与总 rank 数。
  - `pcie_stream`：PCCL 的 `pcclStream_t`（Ascend 上为 **`aclrtCreateStream`** 创建后作为 `pcclStream_t` 传入 PCCL），仅 PCIe 路径使用，在域内管理。
- **计时器**：`timer_fast`、`timer_pcie`，分别挂在用户流与 `pcie_stream` 上；在 **host callback** 里在确有对应段时先对 **`timer_fast` / `timer_pcie` 分别 `Synchronize()`**（仅同步各计时器的 **end event**，再读 **`ElapsedSeconds()`**），**不对用户流** **`aclrtSynchronizeStream`/`cudaStreamSynchronize`**（见 §7 / [STREAM_SYNC.md](STREAM_SYNC.md)）。
- **ShmParamStore**：多卡共享内存；各 rank 写自己的统计；Rank 0 聚合后写回参数表。

### 4.3 DomainManager

- **raw_comm → CommDomain**：每个 raw communicator 拥有独立 CommDomain，避免不同 comm 覆盖 `pcie_rank / pcie_comm / pcie_stream / timer`。
- **stream → PendingCollective（每流 FIFO 队列）**：同一 `stream` 上多次集合会依次 `push_back`；在 **host callback** 里 `TakeStreamPending` 弹出队首，用于计时与写统计（见 §7）。

CommInit 时：构建 CommDomainKey 与 LogicalCommKey，RegisterRawComm(raw_comm, key, logical_key)，执行 node-scope 检测；仅当 `domain->allow_ampccl()` 为真时才 **InitPCIeForDomain(domain, rank, nranks, pccl_bootstrap, len)**。NCCL 传入 **`ncclUniqueId` 字节**；HCCL **`HcclCommInitRootInfo`** 传入 **`HcclRootInfo` 字节**（过长则库内哈希为固定长度 **pccl** id）；**`HcclCommInitAll` / `ncclCommInitAll`** 传入 **空 bootstrap**（pccl 全零 id，单进程 pccl 语义）。可选 **`AMPCCL_PCCL_UNIQUE_ID`** 覆盖、`AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1` 强制全零。
CommDestroy 时：**DestroyPCIeForDomain**，再 UnregisterRawComm(raw_comm)，释放本地 CommDomain。

### 4.4 节点内 communicator 检测

AMP-CCL 默认只接管纯节点内 communicator。CommInit 后，每个 rank 用 **LogicalCommKey** 在本机 POSIX shm 中登记自己的 communicator rank；如果当前节点能看到的 rank 数 `local_seen == nranks`，则判定为 intra-node，并允许 AMP-CCL 初始化 PCCL。否则保守旁路到原始 NCCL/HCCL。

- 单节点 communicator：所有 ranks 在同一个节点注册，`allow_ampccl=true`。
- 多节点 communicator：每个节点只能看到本地部分 ranks，`local_seen < nranks`，`allow_ampccl=false`。
- `CommInitAll`：先登记本进程内所有 ranks，再统一 resolve，避免顺序初始化时 rank 0 等待阻塞后续 rank 注册。

---

## 5. 多卡与共享内存（ShmParamStore）

为保证 **所有 rank 看到同一张参数表**，且 **仅由全局集合时间（例如各 rank 上的 max）驱动调参**，使用共享内存：

- **ShmParamStore** 按 LogicalCommKey 的哈希命名（如 `/ampccl_<hex>`）；同一 logical communicator 的 ranks 挂接到同一段。
- **布局**：头部（magic、nranks、param_version）+ 每 rank 一个 **StatSlot**（op、bytes、datatype、fast_time、pcie_time、fast_bytes、pcie_bytes、success、valid）+ 当前 **OpKey** 对应的 **param 区**（每次集合合并为单条）。
- **单卡（nranks==1）**：不挂 shm；ParamCache 与 Controller Update 均为本地。
- **生命周期**：ShmParamStore 析构时只 `munmap`/`close`，不自动 `shm_unlink`，避免某个 rank 提前销毁 communicator 时删除其他 rank 或 late attach 仍需要的 shm 名称；旧段通过 header 校验/重置处理。
- **多卡（nranks>1）**：
  - **在集合完成后的 host callback（OnStreamSynchronized）**：各 rank 把 ExecStat 写入自己的 StatSlot（WriteMyStat）；**本进程不**在此调用 controller->Update。
  - **下一次集合入口**（如调用 AllReduce/AllGather 时）：
    - **Rank 0**：从 shm ReadAllStatsAndAggregate（各 rank 的 fast_time、pcie_time 取 max 等），得到全局 ExecStat，controller->Update(agg_op_key, global_stat, param_cache)，再把 **当前 OpKey** 的参数 **WriteParams** 写回 shm。
    - **所有 rank**：**ReadParams** 把 shm 中当前 OpKey 的条目合并进本地 **ParamCache**（不是每次清空整张表）。
- 结果：测到的是 **全局**时间（跨 rank 聚合）；所有 rank 读同一张参数表；**只有 Rank 0** 修改它。
- **Rank 0 聚合日志（ReadAllStatsAndAggregate）**：按 **集合类型** 打印与 `tests/HCCL_TEST` 一致的 **十进制 GB/s**（`1e9` 字节/秒）：`algbw = moved_bytes / time`，`busbw = algbw × bus_factor`，其中 `moved_bytes = fast_bytes + pcie_bytes`，`bus_factor` 为 **AllReduce：`2×(n−1)/n`**，**AllGather / ReduceScatter / AllToAll：`(n−1)/n`**。另保留一行 **`Total bw is … GB/s`**，数值等于 **busbw**，供 `tests/nccl_test/allgather/process_log_and_plot.py` 等脚本解析。

---

## 6. Hook 层与调用链

### 6.1 拦截的符号（HCCL / NCCL）

- **HCCL（由 LD_PRELOAD 导出）**：`HcclCommInitRootInfo`、`HcclCommInitAll`、`HcclCommDestroy`、`HcclAllReduce`、`HcclAllGather`、`HcclReduceScatter`、`HcclAllToAll`。**不**导出 **`HcclGetUniqueId` / `HcclCommInitRank`**（常见 HCCL 使用 RootInfo；多进程 bootstrap 与 **pccl** 对齐见上节）。**不**从本库导出 **aclrtSynchronizeStream**（应用仍使用 Ascend 运行时库中的实现）。**此处不导出 `HcclBroadcast`**；应用仍按惯例从 **libhccl.so** 解析该符号。
- **HCCL 的 `HcclAllGatherV` / `HcclReduceScatterV` / `HcclAlltoAllV`**：**不**从 AMP-CCL 导出。在 hook 初始化里通过 **dlsym(libhccl.so)** 取函数指针并注册，仅供 **VirtualCollective** 做混合拆分。**混合 AllGather** 在 **`HasHcclAllGatherV()`** 为真时走 **`HcclAllGatherV`**（无符号则回退 staging + **`HcclAllGather`** + 回调内同步 D2D merge，见 [STREAM_SYNC.md](STREAM_SYNC.md) §6）。应用内 **直接**调用这些 V 接口仍进入 **libhccl.so**，**不**经过自适应路由。
- **NCCL**：对应的 nccl 符号（在链接存在时包括 **ncclBroadcast**）。**不**从本库导出 **cudaStreamSynchronize**（应用仍使用 libcudart 中的实现）。

被 hook 的符号的原始实现通过 **dlopen/dlsym** 从 libhccl.so / libnccl.so、libascendcl.so 或 libacl.so、libcudart.so 等加载。关闭自适应时（例如 `AMPCCL_ENABLE!=1`），调用直接转给原始 API。

### 6.2 调用链概览

1. **CommInit**
   → 调原始 CommInit → 构建 CommDomainKey / LogicalCommKey → RegisterRawComm → node-scope 检测 → 若 `allow_ampccl=true`，InitPCIeForDomain（**pcclCommInitRank(..., pcclUniqueId, rank)**、**aclrtCreateStream/cudaStreamCreate** 作为 pcie_stream；设置域内 pcie_comm、pcie_rank、pcie_nranks、pcie_stream）。

2. **AllReduce / AllGather / ReduceScatter / AllToAll**
   → 由 raw_comm 取 CommDomain；若 domain 不存在或 `allow_ampccl=false`，直接调用原始 NCCL/HCCL → 多卡则 EnsureShmAttached → 若 Rank 0：从 shm 聚合、Update、WriteParams → 所有 rank ReadParams 刷新 param_cache → ParamCache Lookup、Controller SuggestAlpha、Planner CreatePlan → 在 **用户流**上记 timer_fast、启动快路径、停 timer_fast；在 **pcie_stream** 上记 timer_pcie、启动 PCIe 路径、停 timer_pcie → 若启用 PCIe：在用户流上 **StreamWaitEvent**（等待 `timer_pcie` 的 end event 完成）→ **RegisterStreamPending** → **EnqueueCollectiveCompletionCallback**（`cudaLaunchHostFunc` / `aclrtLaunchHostFunc`），回调内执行 **OnStreamSynchronized**。

3. **SynchronizeStream（aclrtSynchronizeStream / cudaStreamSynchronize）**
   → 不经 LD_PRELOAD 拦截；集合收尾与统计已在 **LaunchHostFunc** 回调中完成（与 `StreamWaitEvent` 一起保证 PCIe 与 fast 侧在用户 sync 前完成）。

4. **CommDestroy**
   → **DestroyPCIeForDomain**（pcclCommDestroy；引用计数到 0 则 **pcclFinalize**）→ UnregisterRawComm；**不删除**域。

---

## 7. 虚拟集合层、流同步与计时

更完整的分支说明（混合 / 纯快 / 纯 PCIe、AllGather V 与非 V）见 **[STREAM_SYNC.md](STREAM_SYNC.md)**。

### 7.0 三层分工（与「仅用户流 + pcie_stream」两流模型）

| 层级 | 内容 |
|------|------|
| **L1 设备图顺序** | `timer_pcie.Stop(pcie_stream)` 在 PCIe 工作之后 record end event；**`WaitEventOnStream(user_stream)`**（`PcieWaitOnUserStreamIfNeeded`）使用户流在设备序上等待该 end。**不必**在 host 回调里再 `aclrtSynchronizeStream(pcie_stream)` 来对齐本次 collective。 |
| **L2 用户可见同步** | 用户调用 **`aclrtSynchronizeStream(user_stream)`**（不经 hook）时，队列中已包含 **StreamWaitEvent** 与 **LaunchHostFunc**；sync 返回前 PCIe 与回调收尾已按序完成。 |
| **L3 主机收尾** | **`OnStreamSynchronized`** 内：**RuntimeDeviceGuard**；若 **`fast_bytes>0`** 对 **`timer_fast`**、若 **`use_pcie && pcie_bytes>0`** 对 **`timer_pcie`** 各 **`Synchronize()`**（仅计时 end event）再读 **`ElapsedSeconds()`**。**不得**对 **LaunchHostFunc 所在用户流** **`SynchronizeStream`**。merge、**WriteMyStat** / **Update** 见 §7.2。 |

### 7.1 VirtualCollective 执行（以 AllReduce 为例）

1. 由 count、datatype 与通信器身份构造 **OpKey**（op、bytes、datatype、nranks、topology_hash）。
2. **多卡**：EnsureShmAttached；Rank 0 从 shm ReadAllStatsAndAggregate → Update → WriteParams；所有 rank ReadParams 覆盖 param_cache。
3. ParamCache **Lookup(op_key)**，Controller **SuggestAlpha**，Planner **CreatePlan(op_key.bytes, alpha, use_pcie)** → 得到 Plan（fast_bytes、pcie_bytes、use_pcie）。
4. **启动快路径**：在 **用户流**上：`timer_fast.Start(stream)` → FastBackendImpl::AllReduce(..., stream) → `timer_fast.Stop(stream)`。
5. **启动 PCIe 路径**（若 plan.use_pcie 且 plan.pcie_bytes>0）：在 **domain->pcie_stream()** 上：`timer_pcie.Start(pcie_stream)` → PCIeBackendImpl::AllReduce(domain, ..., pcie_stream) → `timer_pcie.Stop(pcie_stream)`。
   - 集合内部不做同步，以保持透明。
6. **PCIe 时**：`timer_pcie().WaitEventOnStream(user_stream)`（用户流等待 PCIe 计时 end event）→ **RegisterStreamPending** → **EnqueueCollectiveCompletionCallback**；然后返回。

### 7.2 OnStreamSynchronized（host 回调内）

在用户流上 **LaunchHostFunc** 触发，与 `cudaStreamSynchronize` 解耦。

1. **TakeStreamPending(stream)**；若无 pending 则返回。
2. **RuntimeDeviceGuard**：在回调线程上 **`aclrtSetDevice`/`cudaSetDevice`** 到创建 `pcie_stream` 时的设备（见 `domain->pcie_device_id()`），避免 **`aclrtGetDevice == -1`** 导致同步/计时落在错误设备上。
3. **不对用户流** **`SyncDeviceStreamRaw(stream)`**（会与 **LaunchHostFunc** 死锁）。若 **`fast_bytes>0`** / **`use_pcie && pcie_bytes>0`**，分别对 **`timer_fast`** / **`timer_pcie`** **`Synchronize()`**（仅各计时 end event），再读 **`ElapsedSeconds()`**。**不要**对整段 **`pcie_stream`** 做流级排空（与同步计时 event 不同）。
4. **AllGather / ReduceScatter** 若需 merge **tmp → user recv**：**不得**在回调里对 **`LaunchHostFunc` 所在的用户流** 调用 **`aclrtSynchronizeStream`**（会与回调自身死锁）；应使用 **`aclrtMemcpy` 同步 D2D** 等不依赖「同流排空」的完成方式。
5. 从计时器取 **ElapsedSeconds()** 作为 fast_time、pcie_time；与 plan 字节及 pending 成功与否组合成 **ExecStat**。
6. **多卡且已挂 shm**：**WriteMyStat**(my_rank, op_key, stat)；**不要**在此调用 controller->Update。
   **单卡或未挂 shm**：**domain->controller->Update**(op_key, stat, domain->param_cache)。

因此收尾与计时在 **host 回调**中执行（排在同一 user stream 上、位于 **`StreamWaitEvent` 之后**）。集合 API 通常仅 **入队**后即返回；用户 **SynchronizeStream** 会等待队列中的回调跑完，再得到与旧实现一致的可见性与统计时机。

### 7.3 回归检查清单（Ascend / 容器内自行执行）

在已构建 **`build/libampccl_hccl.so`** 的前提下，用 **`tests/HCCL_TEST/run.sh`** 通过 **`LD_PRELOAD`** 跑集成用例；仓内单进程多线程测试仍默认 **`HCCL_TEST_SYNC_EACH_ITER=1`**，便于逐轮验证反馈统计。官方 CANN **`tools/hccl_test`** 的 batch 计时模式应使用 **`AMPCCL_HCCL_TEST_MODE=1`**，此模式固定使用 **`AMPCCL_STATIC_ALPHA`**，并跳过 shm 参数同步和反馈更新。建议覆盖：

- **混合 fast + PCIe**：多消息尺寸、多 rank（**`-p`** 与可用 NPU 数一致）；观察日志中 **`StreamSync`** 后进程不再长时间无响应（旧版回调内 **`timer`/`pcie_stream` 阻塞同步** 易触发卡住）。
- **纯 PCIe / 纯 fast**：通过 **`AMPCCL_MIN_MSG_SIZE`** 等调参使 Planner 偏向一侧（见 §10），各跑一轮冒烟。
- **多卡**：**`ShmParamStore`** 路径下多 rank 统计写入正常。
- **官方 HCCL_TEST batch**：用 **`scripts/run_official_hccl_test.sh -n 4 ./bin/all_reduce_test -b 8K -e 64M -f 2 -p 4`** 从 **`/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test`** 运行；确认日志中 alpha 恒定，且无 shm aggregate/update 反馈环节。

**Fast / PCIe** 段计时均在回调内对对应 **`timer_fast` / `timer_pcie`** **`Synchronize()`** 后再读（仍**不**对用户流 **`SynchronizeStream`**）。在 **`AMPCCL_HCCL_TEST_MODE=1`** 下，回调只保留必要的 pending 清理 / merge 和 timer 对象生命周期，跳过计时查询与参数更新。

---

## 8. PCIe 后端与 PCIeCCL 使用

### 8.1 依赖与构建

- 启用 PCIe 路径时定义 **AMPCCL_ENABLE_PCIE**，并链接 pcieccl 的头文件与库（**`include/pccl/pccl.h`** 与 **`libpccl.so`**；旧版若同目录存在 **`libpccl_kernels.so`** 会自动追加链接；默认在 pcieccl 的 CMake 构建目录根下，见 `BUILD.md`）。
- 在 **InitPCIeForDomain**（comm_init.cc）中：
  - 调用 **`pcclCommInitRank(&pcie_comm, nranks, pcclUniqueId, rank)`**（`pcclUniqueId` 来自 NCCL **`ncclUniqueId`**、HCCL **`HcclRootInfo` 的哈希**或全零）；存入 domain->set_pcie_comm(...)。
  - 使用 **`aclrtCreateStream` / `cudaStreamCreate`** 创建专用流，作为 **`pcclStream_t`** 传入 PCCL；存入 domain->set_pcie_stream(...)。
  每个 CommDomain 在 CommInit 时由 AMP-CCL 创建各自的 **pcie_comm** 与 **pcie_stream**，生命周期内复用。销毁时 **pcclCommDestroy**；最后一个 PCCL 使用者释放后 **pcclFinalize**。

### 8.2 PCIeBackend 如何调用 PCIeCCL

- **入口**：VirtualCollective 调用
  `PCIeBackendImpl::AllReduce(domain, pcie_send, pcie_recv, count, datatype, op, pcie_stream)`
  其中 `pcie_stream = domain->pcie_stream()`；send/recv 指针按 plan 在缓冲区上的偏移选取。
- **AllReduce / AllGather / ReduceScatter / AllToAll**：使用域内 **pcie_comm**、**pcie_rank**、**pcie_stream**；按拆分后的 **Plan** 调用 PCCL 的 **Strided** 集合 API（必要时非 Strided）；**不要**在后端内部调用 **pcclSynchronizeStream**；用户流通过 **StreamWaitEvent**（等待 `timer_pcie` 的 end）与 **LaunchHostFunc** 回调与 PCIe 对齐；**不在** host 回调里对 `pcie_stream` 再做整流同步（见 §7.0）。
- **Broadcast**：HCCL hook **不**包装 `HcclBroadcast`。**NCCL** 构建仍可通过 `nccl_hook.cc` 包装 **ncclBroadcast**。PCIeBackend 提供 Broadcast 桩函数以保持 API 完整；VirtualCollective 在主集合路径上不使用它。

### 8.3 小结

- **通信器与流**：AMP-CCL 在 **InitPCIeForDomain** 中调用 **pcclCommInitRank**（四参数）与 **设备运行时建流**（Ascend/CUDA），并挂到 **CommDomain**。
- **执行**：PCIe 路径使用 **domain->pcie_stream()**，经 **pcclSubmit** 提交；用户流通过 **StreamWaitEvent**（`PcieWaitOnUserStreamIfNeeded`）与 **LaunchHostFunc** 回调与 PCIe 对齐；**不在** **OnStreamSynchronized** 内调用 **pcclSynchronizeStream** / **aclrtSynchronizeStream(pcie_stream)**（见 §7.0）。

---

## 9. Planner 与 Controller

- **Planner**：由 total_bytes、alpha、use_pcie_hint 生成 **Plan**（fast_bytes、pcie_bytes、use_pcie）。做最小消息/块大小与对齐检查；若不应使用 PCIe（消息过小或 use_pcie 为假），则 plan 为仅快路径。
- **Controller / ParamCache**：ParamCache 存 (OpKey → ParamValue)（alpha、use_pcie、fast_bw、pcie_bw）。参数复用不再要求消息 size 精确相等，而是按 **size bucket** 匹配；真实执行、拆分和统计仍使用原始 bytes。拆分用 Controller 的 SuggestAlpha；Update 根据 ExecStat 更新算法状态并写回 ParamValue；多卡时仅 Rank 0 执行 Update，并通过 ShmParamStore 写入共享内存。

### 9.1 Size Bucket 参数复用

现实训练/推理负载中，集合通信 size 往往难以精确重复。AMP-CCL 因此将 `OpKey` 中用于参数复用的 size 从精确 `bytes` 映射为几何分桶后的 `bucket_bytes`：

- `bytes`：保留真实消息大小，用于 Planner 计算 `fast_bytes / pcie_bytes`、执行集合通信、统计带宽与日志。
- `bucket_bytes`：用于 ParamCache 与 ShmParamStore 中的参数匹配；同一通信域、原语、datatype、nranks、topology 下，落入同一 bucket 的不同 size 共享 alpha / use_pcie 等参数。
- 低于 bucket 起点的消息不使用可复用 bucket，参数 key 回退为精确 bytes；同时默认 `AMPCCL_MIN_MSG_SIZE=4MB`，小消息通常直接走纯 fast。

默认分桶从 **4MB** 开始，`AMPCCL_SIZE_BUCKETS_PER_OCTAVE=2` 表示每个 2 倍 size 区间切成 2 个 bucket，因此相邻 bucket 边界倍率为 `2^(1/2)=sqrt(2)`。例如默认边界为：`4MB -> 5.657MB -> 8MB -> 11.314MB -> 16MB -> ...`。

---

## 10. 配置与日志

- **config.h /环境变量**：AMPCCL_ENABLE、**AMPCCL_FORCE_DISABLE**、**AMPCCL_INTRA_NODE_ONLY**、**AMPCCL_NODE_SCOPE_WAIT_MS**、**AMPCCL_FORCE_ENABLE_INTER_NODE**、**AMPCCL_HCCL_TEST_MODE=1**（官方 `tools/hccl_test` batch 兼容模式：强制 static、禁用反馈与 shm 参数同步）、**AMPCCL_FEEDBACK=0**（仅禁用反馈）、**AMPCCL_ALGO**（`inverse` / `split_ratio` / `static`）、**AMPCCL_STATIC_ALPHA**（`static` 固定 PCIe 分流比例，默认 `0.8`）、AMPCCL_MIN_MSG_SIZE、AMPCCL_MIN_CHUNK_SIZE、AMPCCL_SIZE_BUCKET_BASE、AMPCCL_SIZE_BUCKETS_PER_OCTAVE、AMPCCL_ENABLE_PCIE 等；**split_ratio** 可调 **AMPCCL_SPLIT_RATIO_SMOOTH**、**AMPCCL_SPLIT_RATIO_ETA**（见 [adaptive_split_ratio_control.md](adaptive_split_ratio_control.md)）。**pccl**：可选 **AMPCCL_PCCL_UNIQUE_ID**（覆盖 hook 推导的 `pcclUniqueId` 前缀字节）、**AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1**（强制单进程 pccl id）。多进程 **pccl** 需满足 pcieccl 文档的 hugepage 等要求。
- **log.h**：AMPCCL_LOG(level, ...)；级别由环境变量或 SetLogLevel 控制；用于调试拆分、计时、shm 与 PCIe 调用。

---

## 11. 总表

| 项 | 说明 |
|------|-------------|
| 透明性 | LD_PRELOAD 拦截 NCCL/HCCL 与流同步 API；同步与计时发生在用户的 SynchronizeStream。 |
| 通信域 | LogicalCommKey 标识共享 communicator；每个 raw comm 一个 CommDomain，维护本地参数表与 PCIe 资源；通信器由 NCCL/HCCL 管理。 |
| 多卡一致性 | ShmParamStore 共享「每 rank 统计」与「单份参数表」；Rank 0 聚合后写回；所有 rank 读相同参数。 |
| PCIeCCL 使用 | InitPCIeForDomain：pcclCommInitRank、pcclCreateStream；销毁：pcclCommDestroy / pcclFinalize；PCIeBackend 使用 Strided PCCL API + 域内流；跨流对齐靠 **StreamWaitEvent** + host 回调收尾（§7.0），**不在**回调内对整段 `pcie_stream` 再 sync。 |
| 流同步专述 | [STREAM_SYNC.md](STREAM_SYNC.md)：用户流 / `pcie_stream`、L1–L3、AllGather 分支（**`HasHcclAllGatherV()`** 决定 V 与非 V）。 |

以上为完整设计架构及 PCIeCCL 集成与使用说明。
