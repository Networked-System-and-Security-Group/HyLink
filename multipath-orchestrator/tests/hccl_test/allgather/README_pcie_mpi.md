# AllGather + PCIe 干扰：MPI 版

`allgather_with_pcie.cc` 已改为单节点、一卡一个 MPI 进程。每个进程只有主通信线程及一个后台干扰线程；后台使用独立 ACL stream 做 `aclrtMemcpyAsync(D2H)`，每次同步完成后再提交下一次，最多一个在途 copy。

构建与运行：

```bash
cd tests/hccl_test/allgather
bash build.sh allgather_with_pcie.cc hccl_allgather_with_pcie
NP=4 AMPCCL_ALGO=split_ratio AMPCCL_SPLIT_RATIO_INITIAL_ALPHA=0.16 \
  bash run.sh ./hccl_allgather_with_pcie \
  --devices=4 --send-count=67108864 --iters=80 --warmup=5 \
  --interference-ranges=20-39,55-64 --interference-size=134217728
```

- `--send-count` 是 FP32 元素数，67108864 = 256 MiB input/rank；output/rank 为 1 GiB。
- 保留 NVIDIA 参考的**闭区间**：20-39 表示 iter 20..39。与 `mpi_primitives` 的半开区间不同，不要直接照抄右端点。
- `NP` 控制进程数；`--devices` 只检查进程数，不会再创建多个设备线程。设备由同节点 local rank 指定。当前只支持单节点。
- `run.sh` 仅对 `hccl_allgather_with_pcie` 转发 MPI launcher；目录内其他旧线程程序的入口不变。自定义二进制名请直接使用 `bash run_mpi_with_pcie.sh ./自定义名字 ...`。
- `BACKEND=native` 移除 LD_PRELOAD，作为原生 HCCL 对照；默认 `BACKEND=ampccl`。算法用 `AMPCCL_ALGO=static|split_ratio|tcp`；static 比例用 `AMPCCL_STATIC_ALPHA`。
- launcher 强制 `AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0`，由 rank 0 取得并通过 MPI 广播真实 HCCL rootInfo。`AMPCCL_SHM_PARAMS` 默认 1，不再使用线程测试的单进程 PCIeCCL 身份。
- 可覆盖 ACL/Ascend visible devices、库路径 `AMP_CCL_PATH`；默认 timeout 600 秒，可通过 `TIMEOUT_SECONDS` 修改。

## 计时与观测

输入 CPU 填充及 H2D 后、事件计时前执行 MPI_Barrier；预热结束也对齐。所有 MPI 调用只在主线程执行（MPI_THREAD_FUNNELED）。干扰按每个进程的本地 formal iter 激活，不在 warmup 启动。

主输出 `iter=... event_ms=... bw_GBps=...` 使用跨 rank 最大事件耗时与 **output bytes / time / 1e9**（十进制 GB/s），不是 NVIDIA 参考原有的 input GiB/s。原始路径时间和框架反馈仍由 preload 库输出。

各进程使用 rank 0 广播的同一个 steady-clock epoch。结束后顺序输出各卡 collective 和 interference 时间线及 copy 计数，避免 copy 热路径 printf。`copy_interval_GBps` 是完成字节数除以 Host copy/synchronize 区间之和，不是 PCIe 引擎 busy 或固定时间窗的负载强度。

当前保留 NVIDIA 的整段连续干扰语义，包含输入准备、输出 D2H 和校验阶段；没有增加切换 ACK、限速或固定时间窗强度监控。区间切换时一个在途 copy 可能越界，不能把 iter 标签当作精确 DMA 重叠时间。
