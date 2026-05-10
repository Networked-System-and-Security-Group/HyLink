#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"

# Multi-path configuration with split_ratio adaptive algorithm
NP=8
OP=ag
SIZE=$((128 * 1024 * 1024 / 4))
ITERS=40
WARMUP=10
ENABLE_PCIE=0

usage() {
    cat <<'USAGE'
Usage:
  ./run_allgather_split_ratio.sh

Runs 8-process, 1GiB AllGather with split_ratio adaptive algorithm.
The algorithm dynamically adjusts PCIe/fast split based on measured latencies.
USAGE
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

[[ -x "${RUNNER}" ]] || {
    echo "Runner not found or not executable: ${RUNNER}" >&2
    exit 1
}

export NP OP SIZE ITERS WARMUP
export AMPCCL_ALGO=split_ratio
export AMPCCL_ENABLE_PCIE="${ENABLE_PCIE}"
export AMPCCL_LOG_LEVEL=off
export AMPCCL_SHM_PARAMS=1
export ACL_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PCCL_H2H_WORKERS_PER_NUMA=8

echo "Starting split_ratio adaptive AllGather: np=${NP}, total=1GiB, per-rank=128MiB, " \
     "iters=${ITERS}, warmup=${WARMUP}"
exec "${RUNNER}"
