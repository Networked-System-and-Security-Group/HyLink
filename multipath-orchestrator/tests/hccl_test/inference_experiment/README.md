# 2 卡多进程 AllGather：TCP 与 Static

运行：`bash run.sh`；只查看配置：`bash run.sh --dry-run`。

仅重跑 Static 并复用现有 TCP：`bash run.sh --static-only results/<原时间>`。创建新结果目录，复制 TCP 日志及来源元数据，不覆盖旧结果。

- 2 个 MPI 进程，设备 0、1，每卡输入 512MiB（FP32 134217728 元素），每卡接收 1GiB。
- 10 次预热、1001 次正式迭代，分别串行运行 TCP 和 Static。
- Static 固定 alpha=0.39；TCP 从 alpha=0.075 初始化，预热期间也允许反馈更新，正式第 0 轮不强制重置。图例从各自原始日志读取比例，不硬编码。
- 沿用参考脚本的 `mpi_primitives/hccl_mpi_primitives.cc`，每进程一个背景线程持续同步 D2H，每次 256MiB，区间 `[100,200)`、`[300,600)`、`[700,900)`。
- 启用 `--reuse-input`：只在预热前填充并 H2D 一次，所有迭代复用固定输入；全部正式轮结束、干扰线程退出后，仅 D2H 并校验最终输出一次。不逐轮校验中间结果。
- 不启用 msprof 或新增采样；使用默认 build 的两个库，保留既有性能优化。

每次结果位于 `results/<UTC时间>/`：`tcp.log`、`static.log`、`build.log`、`metadata.txt`（配置、环境、库哈希、HEAD）、`bandwidth.csv` 和 `summary.json`。

处理：`python3 process_results.py results/<时间>`，检查两卡正确性及完整 1001 轮。

两个独立绘图脚本：

- `python3 plot_bandwidth.py results/<时间>`：`bandwidth_raw.png`，逐 iter 原始带宽，不去噪、不平滑。
- `python3 plot_variance_envelope.py results/<时间>`：`bandwidth_variance_envelope.png` 与 `variance_envelope.csv`。11-iter 居中窗口的均值 ±1 个总体标准差；保留所有尖峰，窗口不跨阶段边界，边缘截短。这里“方差包络”以标准差表示上下界，CSV 同时记录方差。

带宽口径为 **1GiB / 两 rank 最大 event 耗时**，单位 GiB/s。测试程序原始 `bw_GBps` 字段只计算 512MiB 输入，处理脚本忽略该字段，统一从 event_ms 重算聚合带宽。耗时原始输出精度为 0.001ms。

测试：`python3 -m pytest -q test_experiment.py`。TCP 初值回归可在仓库根目录运行：

干扰有效性检查：`bash check_interference.sh results/<已有TCP结果时间>`。先用链接器包装验证生产后台 worker 的连续循环、设备选择/拷贝错误传播和预热禁用，再用两进程独立运行同一同步 D2H worker 3 秒，最后仅重跑 static=0.39 的 1001 轮。独立探针在结束后校验完整 256MiB Host 缓冲区由 0x5a 变为 0xa5；日志保存在 `diagnostics/`。

每个干扰区间结束后统一输出 `pcie_background`：尝试/成功次数、API状态、总字节、拷贝调用累计时间和区间时长、吞吐。没有逐拷贝 trace，也没有逐 iter 大缓冲区校验。后台错误会使测试失败，不再被静默忽略。`copy_GBps` 是十进制 GB/s，图中的集合通信带宽是 GiB/s，两者不能直接混比。计数按拷贝启动时所在区间归属，最后一笔拷贝可能跨结束边界；Host 调用时间占比不是 DMA 硬件利用率。

汇总审计：`python3 analyze_interference.py results/<新时间> diagnostics/<包含verified=1的独立探针日志>`，生成 `interference_audit.json`。历史 TCP 被复用，不含新增后台统计，不能当作同轮诊断数据。

方向消融：`bash run_direction_ablation.sh`。同期串行运行 D2H、H2D、双向三个 static=0.39 实验，每个 1001 iter，各自先进行 3 秒双进程独立后台探针并校验数据。双向使用两条独立后台线程与独立缓冲区，每方向 256MiB（每卡合计 512MiB），并非单线程交替。结果包含各方向原始日志、后台审计、summary.json、原始带宽图与 11 点均值±标准差包络图；包络不跨阶段边界，原始曲线不平滑。

H2D 下重新比较 TCP 与 static：`PCIE_DIRECTION=h2d bash run.sh`。两种算法均重新运行，不复用历史日志；不设置该变量时仍默认 D2H。绘图标题从日志读取实际干扰方向。

TCP 保持区间：通信库支持 `AMPCCL_TCP_HOLD_RATE_LOW`、`AMPCCL_TCP_HOLD_RATE_HIGH`，默认 `(0.95,1.10)`，与原算法一致。两端均为开区间；下界及以下仍减半，上界及以上仍执行原来的比例增长。参数在每个算法实例创建时读取；非正数、非有限数、格式错误或下界不小于上界时，整对参数回退到默认值。不改变初始比例、上下限、错误回退或通信组隔离。

本目录的 `run.sh` 清理继承的 AMPCCL 环境后会重新设置参数，因此调用它时使用 `TCP_HOLD_RATE_LOW=0.85 TCP_HOLD_RATE_HIGH=1.10`，不要仅设置 `AMPCCL_` 变量。`run_tcp_holdband_sweep.sh` 固定两卡、每卡输入 512MiB、双向每方向 128MiB、1001 iter，跑默认基线一轮和 0.90/0.85/0.80 各两轮。`analyze_tcp_holdband.py <结果目录>` 批量验证日志、汇总带宽及比例跳变，生成原始曲线、包络和划分比例图；路径反馈来自上一轮，脚本已按此对齐，最后一轮没有后续路径统计时不虚构比例。

```bash
g++ -std=c++17 -Ilibampccl tests/hccl_test/inference_experiment/test_tcp_initial_alpha.cc -pthread -o /tmp/test_tcp_initial_alpha
/tmp/test_tcp_initial_alpha
```
