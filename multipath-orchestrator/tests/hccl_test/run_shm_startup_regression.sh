#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/../.." && pwd)"
log_root="${LOG_ROOT:-/logs/allgather_4card_256mb/shm_startup_fixed_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${log_root}"
export NP=4 OP=ag SIZE=67108864 ITERS=4 WARMUP=5
export ACL_VISIBLE_DEVICES=0,1,2,3 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_SHM_PARAMS=1 AMPCCL_LOG_LEVEL=off
export AMPCCL_SPLIT_RATIO_INITIAL_ALPHA=0.15 AMPCCL_SPLIT_RATIO_ETA=0.10
export AMPCCL_SPLIT_RATIO_SMOOTH=0.40 AMPCCL_SPLIT_RATIO_MIN_ALPHA=0.05
export AMPCCL_SPLIT_RATIO_WARMUP_STEPS=5 PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_DISABLE_STAGING_LEASE=0 AMP_CCL_PATH="${repo}/build/libampccl_hccl.so"
export PCCL_IR_PATH="${repo}/../pcieccl/ir"
unset AMPCCL_TARGET_TP_SIZE BENCHMARK_DEBUG_TIMING PCCL_DEBUG_TIMING
sha256sum "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so" > "${log_root}/libraries.sha256"
for algo in split_ratio tcp; do
    export AMPCCL_ALGO="${algo}"
    for repeat in 1 2 3; do
        log="${log_root}/${algo}_${repeat}.log"
        echo "Starting ${algo} launch ${repeat}: ${log}"
        timeout --signal=TERM --kill-after=10s 60s \
            bash "${script_dir}/mpi_primitives/run.sh" \
            --pcie-ranges=1-3 --pcie-size=134217728 > "${log}" 2>&1
        [[ "$(rg -c '^rank [0-3]: PASSED' "${log}")" == 4 ]]
        rg '^Warmup done|^result: PASSED.*global_mismatches=0' "${log}"
        [[ "$(rg -c '^iter=[0-3] event_ms=' "${log}")" == 4 ]]
    done
done
echo "PASS: six fresh 4-card launches; 5 warmup + 4 formal iterations each; logs=${log_root}"
