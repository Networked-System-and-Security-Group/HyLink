English | [中文](README_zh.md)

# Adaptive-CCL Design Notes

This document describes the complete architecture and call chain of the Adaptive Multi-Path Collective Communication Layer (AMP-CCL), as well as how it integrates with PCIeCCL (pcieccl / PCCL).

**Dedicated notes on stream synchronization and dual paths** (user stream / `pcie_stream`, hybrid and pure paths, AllGather branches and V primitives): see **[STREAM_SYNC.md](STREAM_SYNC.md)**.

---

## 1. Overview and Goals

AMP-CCL **transparently** intercepts NCCL / HCCL collective communication APIs via **LD_PRELOAD** and splits each collective operation into two parallel paths:

- **Fast path (Fast Backend)**: uses the native NCCL / HCCL links (e.g., NVLink, IB, HCCS).
- **PCIe path (PCIe Backend)**: uses a PCIe collective library (currently implemented with **PCIeCCL / PCCL**).

Goals:

- **Transparent** to applications: no code changes required; injected solely via LD_PRELOAD.
- **Adaptive splitting** per communication domain (topology + communicator); feedback control tunes the byte ratio (alpha) between the fast path and the PCIe path.
- **Globally consistent parameters across devices**: all ranks see the same parameter table; only Rank 0 updates parameters, based on **global** timings (aggregated via shared memory).
- Fall back to the fast path only when PCIe is not worthwhile.

---

## 2. Architecture

```
User application
    |
NCCL / HCCL API (collectives, stream synchronization, etc.)
    |
LD_PRELOAD injection
    |
AMP-CCL
├── Hook layer (nccl_hook / hccl_hook)
│   └── Intercepts CommInit, CommDestroy, AllReduce, AllGather, SynchronizeStream, etc.
├── Virtual collective layer (VirtualCollective)
│   ├── Parameter source: single device → local ParamCache; multi-device → shared-memory ShmParamStore
│   ├── Planner generates the split plan (fast_bytes / pcie_bytes)
│   ├── Fast path: FastBackend → original NCCL/HCCL
│   └── PCIe path: PCIeBackend → PCIeCCL (pcclCommInitRank, pcclSubmit, Strided collectives, etc.)
├── Stream finalization (OnStreamSynchronized, via host callback)
│   └── Before the collective returns, `StreamWaitEvent`(PCIe) + `LaunchHostFunc` on the user stream; the callback consumes the Pending entry, reads `ElapsedSeconds`, and writes to shm or performs a local Update (see [STREAM_SYNC.md](STREAM_SYNC.md))
└── Communication domain management (DomainManager + CommDomain)
    └── One CommDomain per raw comm; LogicalCommKey is used for node-scope and shared-parameter shm
```

- **Creation and destruction of communicators** are still handled by NCCL / HCCL; we map **raw_comm → CommDomain** to local runtime state and maintain the parameter table and PCIe resources (pcie_comm, pcie_stream, etc.) within the domain. Cross-rank shared state does not reuse the entire CommDomain; instead, **LogicalCommKey** links the node-scope shm and the parameter shm.

---

## 3. Modules and Directory Structure

```
libampccl/
├── hook/
│   ├── nccl_hook.cc      # Intercepts NCCL collective and Comm symbols (does not export cudaStreamSynchronize)
│   └── hccl_hook.cc      # Intercepts HCCL collective and Comm symbols (does not export aclrtSynchronizeStream)
├── core/
│   ├── domain_key.h      # CommDomainKey: local domain identifier, may include the current rank
│   ├── logical_comm_key.h # LogicalCommKey: communicator shared identifier, consistent across all ranks
│   ├── node_scope.h/cc   # Registration of ranks on this node and intra-node communicator detection
│   ├── domain.h          # CommDomain (key, logical_key, param_cache, controller, PCIe state, timers, ShmParamStore)
│   ├── domain_manager.h  # DomainManager: raw_comm↔CommDomain, stream↔PendingCollective
│   ├── comm_init.h/cc    # BuildKey/BuildLogicalKey, InitPCIeForDomain (pcclCommInitRank+pcclUniqueId, device stream creation)
│   ├── virtual_collective.h  # VirtualCollective: entry points for AllReduce/AllGather etc. and split execution
│   ├── planner.h         # Planner: generates a Plan (fast_bytes, pcie_bytes) from alpha and use_pcie
│   ├── stream_sync.h/cc  # OnStreamSynchronized: consumes Pending, synchronizes timers, writes stats or Update
│   ├── stream_host_callback.h/cc  # EnqueueCollectiveCompletionCallback: LaunchHostFunc / aclrtLaunchHostFunc
│   └── shm_store.h/cc    # ShmParamStore: shm layout, Attach, WriteMyStat, ReadParams, WriteParams
├── controller/
│   ├── controller.h      # AdaptiveController (SuggestAlpha, Update)
│   ├── algo_base.h       # AdaptiveAlgo interface
│   ├── algo_factory.h    # Selects the algorithm by configuration (inverse / split_ratio / STATIC)
│   ├── algo_inverse_proportion.h
│   ├── algo_split_ratio_control.h
├── backend/
│   ├── backend_base.h    # BackendResult, template BackendBase
│   ├── fast_backend.h/cc # FastBackendImpl: calls NCCL/HCCL directly
│   ├── memory/           # Memory interfaces for NCCL/HCCL
│   ├── dtype/            # Compile-time data type traits (NCCL/HCCL)
│   └── pcie_backend.h/cc # PCIeBackendImpl: PCCL Strided + pcclSubmit (per-domain pcie_comm, pcie_stream)
├── cache/
│   └── param_cache.h     # ParamCache (OpKey → ParamValue), GetAll/SetFrom for shm serialization
├── telemetry/
│   ├── timer.h           # Timer (CUDA/ACL events or CPU), Start(stream), Stop(stream), Synchronize()
│   └── stats.h           # ExecStat (fast_time, pcie_time, bytes, success)
├── common/
│   ├── op_key.h          # OpKey (op, bytes, datatype), CollectiveType
│   ├── config.h          # AMPCCL_ENABLE, AMPCCL_ALGO, AMPCCL_MIN_*, AMPCCL_ENABLE_PCIE, etc.
│   └── log.h             # Log levels and AMPCCL_LOG
```

---

## 4. Communication Domains (LogicalCommKey, CommDomainKey, CommDomain, DomainManager)

### 4.1 Domain Identifiers: LogicalCommKey and CommDomainKey

The current implementation separates the "shared identity" from the "local runtime identity":

- **LogicalCommKey**: a communicator-level shared identifier; all ranks belonging to the same logical communicator obtain the same `world_size` and `comm_hash`. It is used to name the **node-scope shm** and the **parameter shm**.
  - `ncclCommInitRank`: derived from `ncclUniqueId`.
  - `HcclCommInitRootInfo`: derived from `HcclRootInfo`.
  - `CommInitAll`: there is no cross-process bootstrap, so a per-process incrementing nonce is used. This ensures that local ranks from the same InitAll share a key, while preventing repeated InitAll calls of the same size from colliding on the key.
- **CommDomainKey**: a local domain identifier that may include the current rank; mainly used for logging and local semantics, not for cross-rank shm naming.

**We do not manage NCCL/HCCL's list of communicators**; communicator creation and destruction are handled by NCCL/HCCL. AMP-CCL only maintains the mapping from raw comm to local CommDomain.

### 4.2 CommDomain (per raw comm)

Each process has an independent **CommDomain** for each raw communicator in use. This is important: a CommDomain holds rank/comm-private runtime state such as `pcie_comm`, `pcie_stream`, timers, and temporary buffers, which must not be shared across different raw comms by logical key.

- **key**: local domain identifier.
- **logical_key**: shared communicator identifier, used for node-scope and parameter shm.
- **is_intra_node / allow_ampccl**: node-scope decision cached at CommInit.
- **param_cache**: the parameter table, reused across iterations within the lifetime of a communicator. In the multi-device case, it is synchronized through the shared memory corresponding to the LogicalCommKey.
- **controller**: the adaptive algorithm (SuggestAlpha, Update); in the multi-device case, only Rank 0 updates parameters.
- **PCIe (set by InitPCIeForDomain after CommInit)**:
  - `pcie_comm`: PCCL's `pcclComm_t` (from `pcclCommInitRank`).
  - `pcie_rank`, `pcie_nranks`: this rank and the total number of ranks.
  - `pcie_stream`: PCCL's `pcclStream_t` (on Ascend, created with **`aclrtCreateStream`** and then passed into PCCL as a `pcclStream_t`); used only by the PCIe path and managed within the domain.
- **Timers**: `timer_fast` and `timer_pcie`, attached to the user stream and `pcie_stream` respectively. In the **host callback**, when the corresponding segment actually exists, **`Synchronize()` is called on `timer_fast` / `timer_pcie` separately** (only synchronizing each timer's **end event**, then reading **`ElapsedSeconds()`**); **the user stream is not** **`aclrtSynchronizeStream`/`cudaStreamSynchronize`**-ed (see §7 / [STREAM_SYNC.md](STREAM_SYNC.md)).
- **ShmParamStore**: multi-device shared memory; each rank writes its own stats; Rank 0 aggregates them and writes back the parameter table.

### 4.3 DomainManager

- **raw_comm → CommDomain**: each raw communicator owns an independent CommDomain, preventing different comms from overwriting `pcie_rank / pcie_comm / pcie_stream / timer`.
- **stream → PendingCollective (per-stream FIFO queue)**: multiple collectives on the same `stream` are `push_back`-ed in order; in the **host callback**, `TakeStreamPending` pops the head of the queue for timing and stat writing (see §7).

At CommInit: build the CommDomainKey and LogicalCommKey, call RegisterRawComm(raw_comm, key, logical_key), and run node-scope detection; **InitPCIeForDomain(domain, rank, nranks, pccl_bootstrap, len)** is called only when `domain->allow_ampccl()` is true. NCCL passes the **`ncclUniqueId` bytes**; HCCL **`HcclCommInitRootInfo`** passes the **`HcclRootInfo` bytes** (if too long, the library hashes them into a fixed-length **pccl** id); **`HcclCommInitAll` / `ncclCommInitAll`** pass an **empty bootstrap** (all-zero pccl id, single-process pccl semantics). Optionally, **`AMPCCL_PCCL_UNIQUE_ID`** overrides it, and `AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1` forces an all-zero id.
At CommDestroy: **DestroyPCIeForDomain**, then UnregisterRawComm(raw_comm), releasing the local CommDomain.

### 4.4 Intra-Node Communicator Detection

By default, AMP-CCL only takes over purely intra-node communicators. After CommInit, each rank registers its communicator rank in the local POSIX shm using the **LogicalCommKey**; if the number of ranks visible on the current node `local_seen == nranks`, the communicator is deemed intra-node and AMP-CCL is allowed to initialize PCCL. Otherwise, it conservatively bypasses to the original NCCL/HCCL.

- Single-node communicator: all ranks register on the same node, `allow_ampccl=true`.
- Multi-node communicator: each node only sees its local subset of ranks, `local_seen < nranks`, `allow_ampccl=false`.
- `CommInitAll`: first registers all ranks within this process, then resolves them together, preventing rank 0 from blocking while waiting and thus stalling registration of subsequent ranks during sequential initialization.

---

## 5. Multi-Device and Shared Memory (ShmParamStore)

To ensure that **all ranks see the same parameter table**, and that **tuning is driven only by global collective time (e.g., the max across ranks)**, shared memory is used:

- **ShmParamStore** is named by the hash of the LogicalCommKey (e.g., `/ampccl_<hex>`); ranks of the same logical communicator attach to the same segment.
- **Layout**: header (magic, nranks, param_version) + one **StatSlot** per rank (op, bytes, datatype, fast_time, pcie_time, fast_bytes, pcie_bytes, success, valid) + a **param region** for the current **OpKey** (merged into a single entry per collective).
- **Single device (nranks==1)**: no shm is attached; ParamCache and Controller Update are both local.
- **Lifetime**: the ShmParamStore destructor only calls `munmap`/`close` and does not automatically `shm_unlink`, so that a rank destroying its communicator early does not delete an shm name still needed by other ranks or late attachers; stale segments are handled via header validation/reset.
- **Multi-device (nranks>1)**:
  - **In the host callback after the collective completes (OnStreamSynchronized)**: each rank writes its ExecStat into its own StatSlot (WriteMyStat); **this process does not** call controller->Update here.
  - **At the entry of the next collective** (e.g., when AllReduce/AllGather is called):
    - **Rank 0**: calls ReadAllStatsAndAggregate from shm (taking the max of each rank's fast_time, pcie_time, etc.) to obtain a global ExecStat, calls controller->Update(agg_op_key, global_stat, param_cache), then **WriteParams** the parameters for the **current OpKey** back to shm.
    - **All ranks**: **ReadParams** merges the current OpKey's entry from shm into the local **ParamCache** (rather than clearing the whole table each time).
- Result: the measured time is **global** (aggregated across ranks); all ranks read the same parameter table; **only Rank 0** modifies it.
- **Rank 0 aggregation log (ReadAllStatsAndAggregate)**: prints, per **collective type**, **decimal GB/s** (`1e9` bytes/second) consistent with `tests/HCCL_TEST`: `algbw = moved_bytes / time`, `busbw = algbw × bus_factor`, where `moved_bytes = fast_bytes + pcie_bytes` and `bus_factor` is **AllReduce: `2×(n−1)/n`**, **AllGather / ReduceScatter / AllToAll: `(n−1)/n`**. A **`Total bw is … GB/s`** line is also retained, whose value equals **busbw**, for parsing by scripts such as `tests/nccl_test/allgather/process_log_and_plot.py`.

---

## 6. Hook Layer and Call Chain

### 6.1 Intercepted Symbols (HCCL / NCCL)

- **HCCL (exported via LD_PRELOAD)**: `HcclCommInitRootInfo`, `HcclCommInitAll`, `HcclCommDestroy`, `HcclAllReduce`, `HcclAllGather`, `HcclReduceScatter`, `HcclAllToAll`. **`HcclGetUniqueId` / `HcclCommInitRank`** are **not** exported (typical HCCL usage relies on RootInfo; for aligning multi-process bootstrap with **pccl**, see the previous section). **aclrtSynchronizeStream** is **not** exported from this library (applications still use the implementation in the Ascend runtime library). **`HcclBroadcast` is not exported here**; applications still resolve this symbol from **libhccl.so** as usual.
- **HCCL's `HcclAllGatherV` / `HcclReduceScatterV` / `HcclAlltoAllV`**: **not** exported from AMP-CCL. During hook initialization, their function pointers are obtained via **dlsym(libhccl.so)** and registered, solely for **VirtualCollective** to use in hybrid splitting. **Hybrid AllGather** uses **`HcclAllGatherV`** when **`HasHcclAllGatherV()`** is true (if the symbol is absent, it falls back to staging + **`HcclAllGather`** + synchronous D2D merge in the callback; see [STREAM_SYNC.md](STREAM_SYNC.md) §6). Calls made **directly** to these V interfaces within the application still go into **libhccl.so** and do **not** pass through adaptive routing.
- **NCCL**: the corresponding nccl symbols (including **ncclBroadcast** when present at link time). **cudaStreamSynchronize** is **not** exported from this library (applications still use the implementation in libcudart).

The original implementations of hooked symbols are loaded via **dlopen/dlsym** from libhccl.so / libnccl.so, libascendcl.so or libacl.so, libcudart.so, etc. When adaptivity is disabled (e.g., `AMPCCL_ENABLE!=1`), calls are forwarded directly to the original API.

### 6.2 Call Chain Overview

1. **CommInit**
   → Call the original CommInit → build CommDomainKey / LogicalCommKey → RegisterRawComm → node-scope detection → if `allow_ampccl=true`, InitPCIeForDomain (**pcclCommInitRank(..., pcclUniqueId, rank)**, **aclrtCreateStream/cudaStreamCreate** as pcie_stream; set the domain's pcie_comm, pcie_rank, pcie_nranks, pcie_stream).

2. **AllReduce / AllGather / ReduceScatter / AllToAll**
   → Look up the CommDomain by raw_comm; if the domain does not exist or `allow_ampccl=false`, call the original NCCL/HCCL directly → in the multi-device case, EnsureShmAttached → if Rank 0: aggregate from shm, Update, WriteParams → all ranks ReadParams to refresh param_cache → ParamCache Lookup, Controller SuggestAlpha, Planner CreatePlan → on the **user stream**, start timer_fast, launch the fast path, stop timer_fast; on **pcie_stream**, start timer_pcie, launch the PCIe path, stop timer_pcie → if PCIe is enabled: **StreamWaitEvent** on the user stream (waiting for `timer_pcie`'s end event to complete) → **RegisterStreamPending** → **EnqueueCollectiveCompletionCallback** (`cudaLaunchHostFunc` / `aclrtLaunchHostFunc`), with **OnStreamSynchronized** executed inside the callback.

3. **SynchronizeStream (aclrtSynchronizeStream / cudaStreamSynchronize)**
   → Not intercepted via LD_PRELOAD; collective finalization and stats are already done in the **LaunchHostFunc** callback (together with `StreamWaitEvent`, this guarantees that both the PCIe and fast sides complete before the user's sync).

4. **CommDestroy**
   → **DestroyPCIeForDomain** (pcclCommDestroy; when the reference count reaches 0, **pcclFinalize**) → UnregisterRawComm; the domain is **not deleted**.

---

## 7. Virtual Collective Layer, Stream Synchronization, and Timing

For a more complete description of the branches (hybrid / pure fast / pure PCIe, AllGather V vs. non-V), see **[STREAM_SYNC.md](STREAM_SYNC.md)**.

### 7.0 Three-Layer Division of Responsibilities (with the two-stream model of "user stream + pcie_stream only")

| Layer | Content |
|------|------|
| **L1 Device graph ordering** | `timer_pcie.Stop(pcie_stream)` records the end event after the PCIe work; **`WaitEventOnStream(user_stream)`** (`PcieWaitOnUserStreamIfNeeded`) makes the user stream wait on that end event in device order. There is **no need** to additionally call `aclrtSynchronizeStream(pcie_stream)` in the host callback to align this collective. |
| **L2 User-visible synchronization** | When the user calls **`aclrtSynchronizeStream(user_stream)`** (not hooked), the queue already contains **StreamWaitEvent** and **LaunchHostFunc**; before the sync returns, PCIe and the callback finalization have completed in order. |
| **L3 Host finalization** | Inside **`OnStreamSynchronized`**: **RuntimeDeviceGuard**; if **`fast_bytes>0`**, call **`Synchronize()`** on **`timer_fast`**, and if **`use_pcie && pcie_bytes>0`**, on **`timer_pcie`** (timing end events only), then read **`ElapsedSeconds()`**. **Must not** call **`SynchronizeStream`** on **the user stream hosting the LaunchHostFunc**. For merge, **WriteMyStat** / **Update**, see §7.2. |

### 7.1 VirtualCollective Execution (AllReduce as an Example)

1. Construct the **OpKey** (op, bytes, datatype, nranks, topology_hash) from count, datatype, and communicator identity.
2. **Multi-device**: EnsureShmAttached; Rank 0 calls ReadAllStatsAndAggregate from shm → Update → WriteParams; all ranks ReadParams to overwrite param_cache.
3. ParamCache **Lookup(op_key)**, Controller **SuggestAlpha**, Planner **CreatePlan(op_key.bytes, alpha, use_pcie)** → obtain a Plan (fast_bytes, pcie_bytes, use_pcie).
4. **Launch the fast path**: on the **user stream**: `timer_fast.Start(stream)` → FastBackendImpl::AllReduce(..., stream) → `timer_fast.Stop(stream)`.
5. **Launch the PCIe path** (if plan.use_pcie and plan.pcie_bytes>0): on **domain->pcie_stream()**: `timer_pcie.Start(pcie_stream)` → PCIeBackendImpl::AllReduce(domain, ..., pcie_stream) → `timer_pcie.Stop(pcie_stream)`.
   - No synchronization is performed inside the collective, to preserve transparency.
6. **When PCIe is used**: `timer_pcie().WaitEventOnStream(user_stream)` (the user stream waits on the PCIe timer's end event) → **RegisterStreamPending** → **EnqueueCollectiveCompletionCallback**; then return.

### 7.2 OnStreamSynchronized (Inside the Host Callback)

Triggered by **LaunchHostFunc** on the user stream, decoupled from `cudaStreamSynchronize`.

1. **TakeStreamPending(stream)**; return if there is no pending entry.
2. **RuntimeDeviceGuard**: on the callback thread, **`aclrtSetDevice`/`cudaSetDevice`** to the device on which `pcie_stream` was created (see `domain->pcie_device_id()`), avoiding **`aclrtGetDevice == -1`** causing synchronization/timing to land on the wrong device.
3. **Do not** call **`SyncDeviceStreamRaw(stream)`** on **the user stream** (it would deadlock with **LaunchHostFunc**). If **`fast_bytes>0`** / **`use_pcie && pcie_bytes>0`**, call **`Synchronize()`** on **`timer_fast`** / **`timer_pcie`** respectively (only each timer's end event), then read **`ElapsedSeconds()`**. **Do not** perform stream-level draining on the whole **`pcie_stream`** (which differs from synchronizing the timer events).
4. If **AllGather / ReduceScatter** needs to merge **tmp → user recv**: **must not** call **`aclrtSynchronizeStream`** in the callback on **the user stream hosting the `LaunchHostFunc`** (it would deadlock with the callback itself); instead use a completion method that does not rely on "draining the same stream", such as **synchronous D2D via `aclrtMemcpy`**.
5. Take **ElapsedSeconds()** from the timers as fast_time and pcie_time; combine them with the plan's byte counts and the pending entry's success status into an **ExecStat**.
6. **Multi-device with shm attached**: **WriteMyStat**(my_rank, op_key, stat); **do not** call controller->Update here.
   **Single device or shm not attached**: **domain->controller->Update**(op_key, stat, domain->param_cache).

Thus, finalization and timing are performed in the **host callback** (queued on the same user stream, **after `StreamWaitEvent`**). The collective API typically just **enqueues** and returns; the user's **SynchronizeStream** waits for the queued callback to finish, yielding the same visibility and statistics timing as the old implementation.

### 7.3 Regression Checklist (Run Yourself on Ascend / Inside a Container)

With **`build/libampccl_hccl.so`** already built, use **`tests/HCCL_TEST/run.sh`** to run the integration cases via **`LD_PRELOAD`**; the in-repo single-process multi-threaded tests still default to **`HCCL_TEST_SYNC_EACH_ITER=1`**, which makes it easy to verify the feedback stats round by round. The batch timing mode of the official CANN **`tools/hccl_test`** should use **`AMPCCL_HCCL_TEST_MODE=1`**; this mode always uses **`AMPCCL_STATIC_ALPHA`** and skips shm parameter synchronization and feedback updates. Recommended coverage:

- **Hybrid fast + PCIe**: multiple message sizes, multiple ranks (**`-p`** matching the number of available NPUs); check in the logs that the process no longer hangs for a long time after **`StreamSync`** (the old implementation's **blocking synchronization of `timer`/`pcie_stream`** inside the callback could easily cause hangs).
- **Pure PCIe / pure fast**: tune via **`AMPCCL_MIN_MSG_SIZE`** etc. to bias the Planner toward one side (see §10), and run one smoke test for each.
- **Multi-device**: stats from multiple ranks are written correctly along the **`ShmParamStore`** path.
- **Official HCCL_TEST batch**: run **`scripts/run_official_hccl_test.sh -n 4 ./bin/all_reduce_test -b 8K -e 64M -f 2 -p 4`** from **`/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test`**; confirm in the logs that alpha stays constant and that there is no shm aggregate/update feedback step.

Both **Fast / PCIe** segment timings are read in the callback after calling **`Synchronize()`** on the corresponding **`timer_fast` / `timer_pcie`** (still **without** **`SynchronizeStream`** on the user stream). Under **`AMPCCL_HCCL_TEST_MODE=1`**, the callback keeps only the necessary pending cleanup / merge and timer object lifetime management, skipping timing queries and parameter updates.

---

## 8. PCIe Backend and PCIeCCL Usage

### 8.1 Dependencies and Build

- When the PCIe path is enabled, define **AMPCCL_ENABLE_PCIE** and link against pcieccl's headers and library (**`include/pccl/pccl.h`** and **`libpccl.so`**; with older versions, if **`libpccl_kernels.so`** exists in the same directory, it is linked automatically as well; by default these are located at the root of pcieccl's CMake build directory, see `BUILD.md`).
- In **InitPCIeForDomain** (comm_init.cc):
  - Call **`pcclCommInitRank(&pcie_comm, nranks, pcclUniqueId, rank)`** (`pcclUniqueId` comes from NCCL's **`ncclUniqueId`**, the **hash of HCCL's `HcclRootInfo`**, or all zeros); store it via domain->set_pcie_comm(...).
  - Create a dedicated stream with **`aclrtCreateStream` / `cudaStreamCreate`** and pass it into PCCL as a **`pcclStream_t`**; store it via domain->set_pcie_stream(...).
  For each CommDomain, AMP-CCL creates its own **pcie_comm** and **pcie_stream** at CommInit, reused throughout its lifetime. On destruction, **pcclCommDestroy** is called; after the last PCCL user is released, **pcclFinalize** is called.

### 8.2 How PCIeBackend Calls PCIeCCL

- **Entry point**: VirtualCollective calls
  `PCIeBackendImpl::AllReduce(domain, pcie_send, pcie_recv, count, datatype, op, pcie_stream)`
  where `pcie_stream = domain->pcie_stream()`; the send/recv pointers are chosen by the plan's offsets within the buffers.
- **AllReduce / AllGather / ReduceScatter / AllToAll**: use the domain's **pcie_comm**, **pcie_rank**, and **pcie_stream**; call PCCL's **Strided** collective APIs (non-Strided when necessary) according to the split **Plan**; **do not** call **pcclSynchronizeStream** inside the backend; the user stream is aligned with PCIe via **StreamWaitEvent** (waiting for `timer_pcie`'s end) and the **LaunchHostFunc** callback; **no** additional full-stream synchronization of `pcie_stream` is performed in the host callback (see §7.0).
- **Broadcast**: the HCCL hook does **not** wrap `HcclBroadcast`. **NCCL** builds can still wrap **ncclBroadcast** via `nccl_hook.cc`. PCIeBackend provides a Broadcast stub to keep the API complete; VirtualCollective does not use it on the main collective path.

### 8.3 Summary

- **Communicators and streams**: in **InitPCIeForDomain**, AMP-CCL calls **pcclCommInitRank** (four parameters) and **creates a stream via the device runtime** (Ascend/CUDA), attaching both to the **CommDomain**.
- **Execution**: the PCIe path uses **domain->pcie_stream()** and submits via **pcclSubmit**; the user stream is aligned with PCIe via **StreamWaitEvent** (`PcieWaitOnUserStreamIfNeeded`) and the **LaunchHostFunc** callback; **pcclSynchronizeStream** / **aclrtSynchronizeStream(pcie_stream)** are **not** called inside **OnStreamSynchronized** (see §7.0).

---

## 9. Planner and Controller

- **Planner**: generates a **Plan** (fast_bytes, pcie_bytes, use_pcie) from total_bytes, alpha, and use_pcie_hint. It performs minimum message/chunk size and alignment checks; if PCIe should not be used (message too small or use_pcie is false), the plan is fast-path only.
- **Controller / ParamCache**: ParamCache stores (OpKey → ParamValue) (alpha, use_pcie, fast_bw, pcie_bw). Parameter reuse no longer requires exact message-size equality; instead, entries are matched by **size bucket**, while actual execution, splitting, and statistics still use the original bytes. Splitting uses the Controller's SuggestAlpha; Update updates the algorithm state based on ExecStat and writes back the ParamValue; in the multi-device case, only Rank 0 runs Update and writes to shared memory via ShmParamStore.

### 9.1 Size Bucket Parameter Reuse

In real training/inference workloads, collective communication sizes rarely repeat exactly. AMP-CCL therefore maps the size used for parameter reuse in `OpKey` from the exact `bytes` to a geometrically bucketed `bucket_bytes`:

- `bytes`: retains the real message size, used by the Planner to compute `fast_bytes / pcie_bytes`, to execute the collective, and for bandwidth statistics and logs.
- `bucket_bytes`: used for parameter matching in ParamCache and ShmParamStore; within the same communication domain, primitive, datatype, nranks, and topology, different sizes falling into the same bucket share parameters such as alpha / use_pcie.
- Messages below the bucket starting point do not use reusable buckets, and the parameter key falls back to the exact bytes; in addition, the default `AMPCCL_MIN_MSG_SIZE=4MB` means small messages usually go purely through the fast path.

By default, bucketing starts at **4MB**, and `AMPCCL_SIZE_BUCKETS_PER_OCTAVE=2` means each 2x size interval is split into 2 buckets, so the ratio between adjacent bucket boundaries is `2^(1/2)=sqrt(2)`. For example, the default boundaries are: `4MB -> 5.657MB -> 8MB -> 11.314MB -> 16MB -> ...`.

---

## 10. Configuration and Logging

- **config.h / environment variables**: AMPCCL_ENABLE, **AMPCCL_FORCE_DISABLE**, **AMPCCL_INTRA_NODE_ONLY**, **AMPCCL_NODE_SCOPE_WAIT_MS**, **AMPCCL_FORCE_ENABLE_INTER_NODE**, **AMPCCL_HCCL_TEST_MODE=1** (batch compatibility mode for the official `tools/hccl_test`: forces static, disables feedback and shm parameter synchronization), **AMPCCL_FEEDBACK=0** (disables feedback only), **AMPCCL_ALGO** (`inverse` / `split_ratio` / `static`), **AMPCCL_STATIC_ALPHA** (the fixed PCIe split ratio for `static`, default `0.8`), AMPCCL_MIN_MSG_SIZE, AMPCCL_MIN_CHUNK_SIZE, AMPCCL_SIZE_BUCKET_BASE, AMPCCL_SIZE_BUCKETS_PER_OCTAVE, AMPCCL_ENABLE_PCIE, etc.; **split_ratio** can be tuned with **AMPCCL_SPLIT_RATIO_SMOOTH** and **AMPCCL_SPLIT_RATIO_ETA** (see [adaptive_split_ratio_control.md](adaptive_split_ratio_control.md)). **pccl**: optional **AMPCCL_PCCL_UNIQUE_ID** (overrides the prefix bytes of the `pcclUniqueId` derived by the hook) and **AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1** (forces a single-process pccl id). Multi-process **pccl** must satisfy the hugepage and other requirements in the pcieccl documentation.
- **log.h**: AMPCCL_LOG(level, ...); the level is controlled by an environment variable or SetLogLevel; used for debugging splitting, timing, shm, and PCIe calls.

---

## 11. Summary Table

| Item | Description |
|------|-------------|
| Transparency | LD_PRELOAD intercepts NCCL/HCCL and stream synchronization APIs; synchronization and timing happen at the user's SynchronizeStream. |
| Communication domains | LogicalCommKey identifies the shared communicator; one CommDomain per raw comm, maintaining the local parameter table and PCIe resources; communicators are managed by NCCL/HCCL. |
| Multi-device consistency | ShmParamStore shares "per-rank stats" and "a single parameter table"; Rank 0 aggregates and writes back; all ranks read the same parameters. |
| PCIeCCL usage | InitPCIeForDomain: pcclCommInitRank, pcclCreateStream; destruction: pcclCommDestroy / pcclFinalize; PCIeBackend uses the Strided PCCL APIs + the per-domain stream; cross-stream alignment relies on **StreamWaitEvent** + host-callback finalization (§7.0), with **no** additional sync of the whole `pcie_stream` inside the callback. |
| Stream sync notes | [STREAM_SYNC.md](STREAM_SYNC.md): user stream / `pcie_stream`, L1–L3, AllGather branches (**`HasHcclAllGatherV()`** decides V vs. non-V). |

The above is the complete design architecture together with the PCIeCCL integration and usage notes.
