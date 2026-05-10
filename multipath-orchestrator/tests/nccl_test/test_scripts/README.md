# NCCL 集合通信性能对比实验指南

本目录包含两套完整的 baseline(原生 NCCL)vs adaptive(Adaptive-CCL 多路径)对比实验流程:

| 实验 | 运行模式 | sweep 脚本 | 日志目录 | 结果目录 |
|------|----------|-----------|----------|----------|
| 实验一 | 单进程多线程(每卡一线程) | `compare_multi_case.sh` | `logs/compare/` | `csv/` |
| 实验二 | 多进程 MPI(每卡一进程) | `mpi_compare_multi_case.sh` | `logs/mpi_compare/` | `csv_mpi/` |

默认实验矩阵:`allgather / allreduce / reducescatter` × `2 / 4 / 8` 卡 × `1KB ~ 1GB`(×4 递增)× `baseline / adaptive`,每个 case 30 次迭代。
(注:单进程脚本默认还包含 alltoall,用 `COLLECTIVES` 限定为三个原语,见下文。)

---

## 第 0 步:前置条件检查(只需一次)

依赖路径(脚本已默认指向本机,均可用环境变量覆盖):

- NCCL:`/root/sxb/nccl/build`(`NCCL_HOME`)
- pcieccl:`/root/sxb/pcieccl`(`PCIECCL_ROOT`,IR 文件在其 `ir/` 下,脚本自动推导 `PCCL_IR_PATH`)
- Adaptive-CCL hook:`/root/sxb/Adaptive-CCL/build/libampccl_nccl.so`(需已构建)
- `mpirun` 为 MPICH/Hydra(本机 `/usr/bin/mpirun`),脚本已使用 `-genvall` 传递环境变量

## 第 1 步:编译测试二进制(只需一次,改代码后重编)

```bash
# 单进程测试程序(allgather 通常已编好,可一并重编)
cd /root/sxb/Adaptive-CCL/tests/nccl_test/allgather      && ./build.sh allgather_param.cc nccl_allgather_param
cd /root/sxb/Adaptive-CCL/tests/nccl_test/allreduce      && ./build.sh allreduce_param.cc nccl_allreduce_param
cd /root/sxb/Adaptive-CCL/tests/nccl_test/reducescatter  && ./build.sh reducescatter_param.cc nccl_reducescatter_param

# 多进程 MPI 测试程序(源码新增了 "Warmup done" 标记,必须重编一次)
cd /root/sxb/Adaptive-CCL/tests/nccl_test/mpi_primitives && ./build.sh
```

## 第 2 步:冒烟测试(建议,几分钟)

先用小矩阵确认两条链路都能跑通,再投全量:

```bash
cd /root/sxb/Adaptive-CCL
bash tests/nccl_test/test_scripts/compare_multi_case.sh --quick
bash tests/nccl_test/test_scripts/mpi_compare_multi_case.sh --quick
```

`--quick` = 2 卡 × 4 个尺寸 × 5 次迭代。结束时输出 `Failed cases: 0` 即通过。

## 第 3 步:实验一 —— 单进程多线程全量 sweep

```bash
cd /root/sxb/Adaptive-CCL
COLLECTIVES="allgather allreduce reducescatter" \
  bash tests/nccl_test/test_scripts/compare_multi_case.sh
```

## 第 4 步:实验二 —— 多进程 MPI 全量 sweep

```bash
cd /root/sxb/Adaptive-CCL
bash tests/nccl_test/test_scripts/mpi_compare_multi_case.sh
```

(该脚本默认就是 allgather/allreduce/reducescatter 三个原语,无需指定。)

## 第 5 步:解析日志、生成 CSV 和图表

```bash
cd /root/sxb/Adaptive-CCL

# 实验一:logs/compare → csv/{raw,summary,standard}_bandwidth.csv + csv/plots/*.png
bash tests/nccl_test/test_scripts/run_full_pipeline.sh

# 实验二:logs/mpi_compare → csv_mpi/... + csv_mpi/plots/*.png
bash tests/nccl_test/test_scripts/run_full_pipeline.sh --mpi
```

两套结果目录完全隔离,跑完后直接对比 `csv/standard_bw.csv` 与 `csv_mpi/standard_bw.csv`
(或对比 `csv/plots/` 与 `csv_mpi/plots/` 下的同名图)即可看两种运行模式的性能差异。

---

## 常用参数与环境变量

sweep 脚本(两套通用):

| 变量 | 默认 | 说明 |
|------|------|------|
| `COLLECTIVES` | 见上 | 空格分隔的原语列表 |
| `CARD_COUNTS` | `2 4 8` | 卡数(MPI 模式即进程数) |
| `SIZES_BYTES` | 1KB..1GB ×4 | 消息尺寸列表(字节) |
| `ITERS` | 30 | 每 case 测量迭代次数 |
| `RUN_BASELINE` / `RUN_ADAPTIVE` | 1 / 1 | 单独跑某一模式(或用 `--baseline-only` / `--adaptive-only`) |
| `CUDA_VISIBLE_DEVICES` | 不设 | 限定可见 GPU |

后处理 `run_full_pipeline.sh`:

- `--trim-n N`:去掉每组样本最高/最低各 N 个再取均值(默认 8)
- `--bw-column algbw_gbps`:画 algbw 而非默认的 busbw
- `--skip-plots`:只出 CSV 不画图

示例(只补跑 8 卡 allreduce 的 adaptive):

```bash
CARD_COUNTS="8" COLLECTIVES="allreduce" RUN_BASELINE=0 \
  bash tests/nccl_test/test_scripts/mpi_compare_multi_case.sh
```

## 注意事项

1. **内存峰值**:8 卡 × 1GB 的 allgather/reducescatter,每 rank 约需 8GB 显存 + 16GB 锁页主机内存,MPI 模式 8 进程合计锁页内存峰值约 128GB。机器内存不足时把 8 卡的 `SIZES_BYTES` 截到 256MB(去掉最后一项 `1073741824`)。
2. **单进程 vs 多进程的关键环境差异**:单进程 sweep 设 `AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1`;多进程 sweep 必须 unset 该变量(脚本内已处理),并依赖 `AMPCCL_SHM_PARAMS=1` 做跨 rank 参数聚合。不要手动给 MPI 模式设置 `FORCE_ZERO_UNIQUE_ID=1`。
3. **带宽口径**:解析阶段已把单进程 baseline 的 GiB/s 统一换算为十进制 GB/s(×1.0737),与 AMPCCL aggregate 行、MPI 程序口径一致;MPI `result:` 行的 payload 口径也已换算为项目的 per-rank op_bytes 口径。因此新跑出的 baseline 数值会比旧版 CSV 高约 7.4%,属于单位修正。
4. **失败重跑**:sweep 单个 case 失败不会中断整轮(结尾汇总 `Failed cases`),可按上面的示例用环境变量定点补跑;同一时间戳的日志按 `<N>gpu_<mode>_<时间>.log` 命名,解析器会扫描目录下全部日志。
5. 重新跑全量前如需清空旧数据:`rm -rf tests/nccl_test/test_scripts/logs/compare tests/nccl_test/test_scripts/csv`(MPI 同理换成 `logs/mpi_compare`、`csv_mpi`)。
