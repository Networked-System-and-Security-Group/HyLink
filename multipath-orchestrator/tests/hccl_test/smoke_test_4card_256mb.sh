#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"
SMOKE_LOG_DIR="/logs/smoke_4card_256mb"
mkdir -p "${SMOKE_LOG_DIR}"

NP=4
OP=ag
SIZE=67108864
ITERS=40
WARMUP=5
PCIE_RANGES="15-30"
PCIE_SIZE_BYTES=67108864

export NP OP SIZE ITERS WARMUP
export ACL_VISIBLE_DEVICES="0,1,2,3"
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_SHM_PARAMS=1
export AMPCCL_LOG_LEVEL=off
export PCCL_H2H_WORKERS=8
export PCCL_H2H_WORKERS_PER_NUMA=8

run_test() {
    local name="$1"
    local algo="$2"
    local static_a="$3"
    local init_a="$4"
    local smooth="$5"
    local eta="$6"
    local log_path="${SMOKE_LOG_DIR}/${name}.log"

    echo "=== Running ${name}: algo=${algo} static_a=${static_a} init_a=${init_a} smooth=${smooth} eta=${eta} ==="
    export AMPCCL_ALGO="${algo}"
    export AMPCCL_STATIC_ALPHA="${static_a}"
    export AMPCCL_SPLIT_RATIO_INITIAL_ALPHA="${init_a}"
    export AMPCCL_SPLIT_RATIO_SMOOTH="${smooth}"
    export AMPCCL_SPLIT_RATIO_ETA="${eta}"
    export AMPCCL_SPLIT_RATIO_MIN_ALPHA="0.03"

    "${RUNNER}" --pcie-ranges="${PCIE_RANGES}" --pcie-size="${PCIE_SIZE_BYTES}" > "${log_path}" 2>&1
    echo "Finished ${name}, log written to ${log_path}"
}

run_test "static_0.15" "static" "0.15" "0.15" "0.5" "0.1"
run_test "static_0.12" "static" "0.12" "0.12" "0.5" "0.1"
run_test "static_0.10" "static" "0.10" "0.10" "0.5" "0.1"
run_test "split_ratio_s0.5_e0.10" "split_ratio" "0.15" "0.15" "0.5" "0.10"
run_test "split_ratio_s0.4_e0.12" "split_ratio" "0.15" "0.15" "0.4" "0.12"

echo "ALL SMOKE TESTS FINISHED"
