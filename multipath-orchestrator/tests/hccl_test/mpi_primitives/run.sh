#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
AMP_CCL_PATH="${AMP_CCL_PATH:-${ADAPTIVE_CCL_ROOT}/build/libampccl_hccl.so}"

BIN="${BIN:-${SCRIPT_DIR}/hccl_mpi_primitives}"
NP="${NP:-8}"
OP="${OP:-ar}"
SIZE="${SIZE:-16777216}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-6}"

PCIECCL_ROOT="${PCIECCL_ROOT:-$(cd "${ADAPTIVE_CCL_ROOT}/.." 2>/dev/null && pwd)/pcieccl}"
if [ -z "${PCCL_IR_PATH:-}" ] && [ -d "${PCIECCL_ROOT}/ir" ]; then
  export PCCL_IR_PATH="${PCIECCL_ROOT}/ir"
fi

export ACL_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export AMPCCL_ENABLE="${AMPCCL_ENABLE:-1}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-off}"
export AMPCCL_ALGO="${AMPCCL_ALGO:-split_ratio}"
export AMPCCL_STATIC_ALPHA="${AMPCCL_STATIC_ALPHA:-0.0625}"
export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-1}"
export AMPCCL_MIN_CHUNK_SIZE="${AMPCCL_MIN_CHUNK_SIZE:-20}"
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK="${AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK:-0}"

# Multi-process MPI must use a real HCCL rootInfo/pcclUniqueId.
export AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID="${AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID:-0}"

if [ ! -x "${BIN}" ]; then
  echo "Binary not found: ${BIN}" >&2
  echo "Build first: cd ${SCRIPT_DIR} && ./build.sh" >&2
  exit 1
fi
if [ ! -f "${AMP_CCL_PATH}" ]; then
  echo "Error: ${AMP_CCL_PATH} not found." >&2
  echo "Build it first: cd ${ADAPTIVE_CCL_ROOT} && mkdir -p build && cd build && cmake .. -DHCCL_ONLY=ON && make" >&2
  exit 1
fi

export LD_PRELOAD="${AMP_CCL_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

# MPICH/Hydra mpirun: -genvall propagates the exported environment.
exec mpirun -np "${NP}" -genvall \
  "${BIN}" --op="${OP}" --size="${SIZE}" --iters="${ITERS}" --warmup="${WARMUP}" "$@"
