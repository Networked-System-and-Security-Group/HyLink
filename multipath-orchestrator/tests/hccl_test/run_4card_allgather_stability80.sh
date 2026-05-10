#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-/logs/allgather_4card_256mb/stability80_${STAMP}}"
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/images/stability80_${STAMP}}"
mkdir -p "${LOG_ROOT}" "${IMAGE_DIR}"

export NP=4 OP=ag SIZE=67108864 ITERS=80 WARMUP=5
export ACL_VISIBLE_DEVICES=0,1,2,3 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_SHM_PARAMS=1 AMPCCL_LOG_LEVEL=off
export AMPCCL_SPLIT_RATIO_INITIAL_ALPHA=0.15 AMPCCL_SPLIT_RATIO_ETA=0.10
export AMPCCL_SPLIT_RATIO_SMOOTH=0.40 AMPCCL_SPLIT_RATIO_MIN_ALPHA=0.05
export AMPCCL_SPLIT_RATIO_WARMUP_STEPS=5 PCCL_H2H_WORKERS_PER_NUMA=8
# Test the normal staging lease path, including its barrier-slot fix.
export PCCL_DISABLE_STAGING_LEASE=0
unset AMPCCL_TARGET_TP_SIZE BENCHMARK_DEBUG_TIMING PCCL_DEBUG_TIMING
export AMP_CCL_PATH="${REPO}/build/libampccl_hccl.so"
export PCIECCL_ROOT="$(cd "${REPO}/../pcieccl" && pwd)"
export PCCL_IR_PATH="${PCIECCL_ROOT}/ir"

{
    date -u
    echo 'np=4 input_bytes_per_rank=268435456 output_bytes_per_rank=1073741824 dtype=fp32'
    echo 'iters=80 warmup=5 pcie_d2h_bytes_per_rank=134217728 ranges=[20,40),[55,65)'
    echo 'split_ratio: initial_alpha=0.15 eta=0.10 smooth=0.40 min_alpha=0.05 warmup_steps=5'
    echo 'PCCL_H2H_WORKERS_PER_NUMA=8 PCCL_DISABLE_STAGING_LEASE=0'
    git -C "${REPO}" rev-parse HEAD
    git -C "${PCIECCL_ROOT}" rev-parse HEAD
    git -C "${REPO}" diff --stat
    git -C "${PCIECCL_ROOT}" diff --stat
    sha256sum "${AMP_CCL_PATH}" "${PCIECCL_ROOT}/build/libpccl.so" "${SCRIPT_DIR}/mpi_primitives/hccl_mpi_primitives"
    ldd "${AMP_CCL_PATH}"
    env | sort | rg '^(AMPCCL|PCCL|ACL_VISIBLE|ASCEND_RT_VISIBLE|LD_PRELOAD|LD_LIBRARY_PATH)'
} > "${LOG_ROOT}/metadata.txt"
git -C "${REPO}" diff > "${LOG_ROOT}/adaptive_worktree.patch"
git -C "${PCIECCL_ROOT}" diff > "${LOG_ROOT}/pcieccl_worktree.patch"

for algorithm in split_ratio tcp; do
    export AMPCCL_ALGO="${algorithm}"
    echo "Running ${algorithm}: ${LOG_ROOT}/${algorithm}.log"
    timeout --signal=TERM --kill-after=30s 600s \
        bash "${SCRIPT_DIR}/mpi_primitives/run.sh" \
        --pcie-ranges=20-40,55-65 --pcie-size=134217728 \
        > "${LOG_ROOT}/${algorithm}.log" 2>&1
    rg '^result:|^rank [0-3]: PASSED' "${LOG_ROOT}/${algorithm}.log"
done
python3 "${SCRIPT_DIR}/analyze_allgather_stability80.py" \
    --log-root "${LOG_ROOT}" --image-dir "${IMAGE_DIR}"
echo "logs=${LOG_ROOT}"
echo "images=${IMAGE_DIR}"
