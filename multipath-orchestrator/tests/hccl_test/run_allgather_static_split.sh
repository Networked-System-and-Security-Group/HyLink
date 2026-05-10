#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"

# Multi-path configuration: 8 MPI processes, AllGather, 1GiB aggregate buffer,
# 128MiB per rank, 40 formal iterations and 10 warmup iterations.
# PCIe path ratio controlled by AMPCCL_STATIC_ALPHA (0.1 = 10% PCIe, 90% fast).
NP=8
OP=ag
SIZE=$((128 * 1024 * 1024 / 4))
ITERS=40
WARMUP=10
STATIC_ALPHA="${1:-0}"  # Default 0.1 (10% PCIe, 90% fast)
ENABLE_PCIE=1

usage() {
    cat <<'USAGE'
Usage:
  ./run_allgather_static_split.sh [ALPHA]

Runs an 8-process, 1GiB aggregate-buffer AllGather with static split.
ALPHA: PCIe path ratio (default 0.1 = 10% PCIe, 90% fast path)
Examples:
  ./run_allgather_static_split.sh 0.1   # 10% PCIe, 90% fast
  ./run_allgather_static_split.sh 0.5   # 50% PCIe, 50% fast
USAGE
}

case "${STATIC_ALPHA}" in
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
export AMPCCL_ALGO=static
export AMPCCL_STATIC_ALPHA="${STATIC_ALPHA}"
export AMPCCL_ENABLE_PCIE="${ENABLE_PCIE}"
export AMPCCL_LOG_LEVEL=off
export ACL_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export AMPCCL_FAST_ONLY_BW_LOG=1

echo "Starting static split AllGather: np=${NP}, total=1GiB, per-rank=128MiB, " \
     "iters=${ITERS}, warmup=${WARMUP}, pcie_ratio=${STATIC_ALPHA}"
exec "${RUNNER}"
