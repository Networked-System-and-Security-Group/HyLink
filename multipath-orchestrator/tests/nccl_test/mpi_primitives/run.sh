#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${BIN:-${SCRIPT_DIR}/nccl_mpi_primitives}"
NP="${NP:-4}"
OP="${OP:-allreduce}"
COUNT="${COUNT:-1048576}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-3}"
AMP_SO="${AMP_SO:-${ROOT}/build/libampccl_nccl.so}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found: ${BIN}" >&2
  echo "Build first: cd ${SCRIPT_DIR} && ./build.sh" >&2
  exit 1
fi
if [[ ! -f "${AMP_SO}" ]]; then
  echo "AMP-CCL NCCL hook not found: ${AMP_SO}" >&2
  exit 1
fi

export LD_PRELOAD="${AMP_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
export AMPCCL_ENABLE="${AMPCCL_ENABLE:-1}"
export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-1}"
export AMPCCL_ALGO="${AMPCCL_ALGO:-split_ratio}"
export AMPCCL_SHM_PARAMS="${AMPCCL_SHM_PARAMS:-1}"
export AMPCCL_INTRA_NODE_ONLY="${AMPCCL_INTRA_NODE_ONLY:-1}"
export AMPCCL_NODE_SCOPE_WAIT_MS="${AMPCCL_NODE_SCOPE_WAIT_MS:-1000}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-3}"
unset AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID

# MPICH/Hydra mpirun: -genvall propagates the full (exported) environment,
# replacing OpenMPI-style "-x VAR" which Hydra does not understand.
exec mpirun -np "${NP}" -genvall \
  "${BIN}" --op="${OP}" --count="${COUNT}" --iters="${ITERS}" --warmup="${WARMUP}" "$@"
