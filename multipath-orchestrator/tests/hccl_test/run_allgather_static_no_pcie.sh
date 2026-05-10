#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"

# Fixed experiment configuration:
#   8 MPI processes, AllGather, 1GiB aggregate buffer,
#   128MiB per rank, 40 formal iterations and 10 warmup iterations.
NP=8
OP=ag
SIZE=$((128 * 1024 * 1024 / 4))
ITERS=40
WARMUP=10
STATIC_ALPHA=1
ENABLE_PCIE=1

usage() {
    cat <<'USAGE'
Usage:
  ./run_allgather_static_no_pcie.sh
  ./run_allgather_static_no_pcie.sh --dry-run

Runs an 8-process, 1GiB aggregate-buffer AllGather with the static
algorithm at alpha=0.08 and no PCIe interference.
USAGE
}

case "${1:-}" in
    "") ;;
    --dry-run)
        echo "config: np=${NP} op=${OP} size=${SIZE} iters=${ITERS} warmup=${WARMUP}"
        echo "static_alpha=${STATIC_ALPHA} enable_pcie=${ENABLE_PCIE}"
        echo "ag_total_bytes=$((SIZE * 4 * NP)) ag_per_rank_bytes=$((SIZE * 4))"
        exit 0
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
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

echo "Starting no-PCIe AllGather: np=${NP}, total=1GiB, per-rank=128MiB, " \
     "iters=${ITERS}, warmup=${WARMUP}, static=${STATIC_ALPHA}"
exec "${RUNNER}"
