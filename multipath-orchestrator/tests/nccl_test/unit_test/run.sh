#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BIN="${BIN:-${SCRIPT_DIR}/nccl_commsplit_allgather_mpi}"
MODE="${MODE:-adaptive}"
NP="${NP:-4}"
SEND_COUNT="${SEND_COUNT:-1024}"
ITERS="${ITERS:-1}"
CHILD_SIZE="${CHILD_SIZE:-4}"
AMP_SO="${AMP_SO:-${ROOT}/build/libampccl_nccl.so}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"

if [[ -n "${NCCL_HOME}" && -d "${NCCL_HOME}/lib" ]]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found: ${BIN}" >&2
  echo "Build first: cd ${SCRIPT_DIR} && ./build.sh" >&2
  exit 1
fi

if [[ "${MODE}" == "baseline" ]]; then
  exec env -u LD_PRELOAD mpirun -np "${NP}" \
    "${BIN}" "${NP}" "${SEND_COUNT}" "${ITERS}" "${CHILD_SIZE}" "$@"
fi

[[ -f "${AMP_SO}" ]] || { echo "AMP so not found: ${AMP_SO}" >&2; exit 1; }

export LD_PRELOAD="${AMP_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
export AMPCCL_ENABLE="${AMPCCL_ENABLE:-1}"
export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-1}"
export AMPCCL_ALGO="${AMPCCL_ALGO:-static}"
export AMPCCL_STATIC_ALPHA="${AMPCCL_STATIC_ALPHA:-0.3}"
export AMPCCL_SHM_PARAMS="${AMPCCL_SHM_PARAMS:-1}"
export AMPCCL_ENABLE_NCCL_V="${AMPCCL_ENABLE_NCCL_V:-0}"
export AMPCCL_INTRA_NODE_ONLY="${AMPCCL_INTRA_NODE_ONLY:-1}"
export AMPCCL_NODE_SCOPE_WAIT_MS="${AMPCCL_NODE_SCOPE_WAIT_MS:-1000}"
export AMPCCL_MAX_PCIE_RECVBUF_BYTES="${AMPCCL_MAX_PCIE_RECVBUF_BYTES:-$((4 * 1024 * 1024 * 1024))}"
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK="${AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK:-1}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-info}"
unset AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID

exec mpirun -np "${NP}" \
  -x LD_PRELOAD \
  -x LD_LIBRARY_PATH \
  -x AMPCCL_ENABLE \
  -x AMPCCL_ENABLE_PCIE \
  -x AMPCCL_ALGO \
  -x AMPCCL_STATIC_ALPHA \
  -x AMPCCL_SHM_PARAMS \
  -x AMPCCL_ENABLE_NCCL_V \
  -x AMPCCL_INTRA_NODE_ONLY \
  -x AMPCCL_NODE_SCOPE_WAIT_MS \
  -x AMPCCL_MAX_PCIE_RECVBUF_BYTES \
  -x AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK \
  -x AMPCCL_LOG_LEVEL \
  "${BIN}" "${NP}" "${SEND_COUNT}" "${ITERS}" "${CHILD_SIZE}" "$@"
