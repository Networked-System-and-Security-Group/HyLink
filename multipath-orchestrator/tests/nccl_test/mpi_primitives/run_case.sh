#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${BIN:-${SCRIPT_DIR}/nccl_mpi_primitives}"
AMP_SO="${AMP_SO:-${ROOT}/build/libampccl_nccl.so}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

MODE="${MODE:-adaptive}"          # baseline | adaptive
NP="${NP:-4}"
OP="${OP:-allgather}"             # allgather | allreduce | reducescatter | alltoall
SIZE_BYTES="${SIZE_BYTES:-$((1024 * 1024 * 256))}"
COUNT="${COUNT:-}"
ITERS="${ITERS:-30}"
WARMUP="${WARMUP:-5}"
LOG_DIR="${LOG_DIR:-}"


export PCCL_FORCE_NVLINK=1
export AMPCCL_MAX_PCIE_RECVBUF_BYTES=$((4 * 1024 * 1024 * 1024))
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK=1

usage() {
  cat <<USAGE
Usage:
  MODE=adaptive OP=allgather SIZE_BYTES=16777216 NP=4 ITERS=20 WARMUP=5 $0
  MODE=baseline OP=allreduce COUNT=4194304 NP=4 $0

Environment:
  MODE          baseline | adaptive, default adaptive
  OP            allgather | allreduce | reducescatter | alltoall, default allgather
  NP            MPI process count / NCCL ranks, default 4
  SIZE_BYTES    message bytes per rank/peer, default 16777216. Ignored when COUNT is set
  COUNT         float element count. If unset, COUNT=SIZE_BYTES/4
  ITERS         measured iterations, default 20
  WARMUP        warmup iterations, default 5
  BIN           benchmark binary, default ./nccl_mpi_primitives
  AMP_SO        Adaptive-CCL hook, default ../../../build/libampccl_nccl.so
  LOG_DIR       optional directory to tee one log file into

Adaptive env defaults:
  AMPCCL_ENABLE=1
  AMPCCL_ENABLE_PCIE=1
  AMPCCL_ALGO=split_ratio
  AMPCCL_SHM_PARAMS=1
  AMPCCL_ENABLE_NCCL_V=1
  AMPCCL_INTRA_NODE_ONLY=1
  AMPCCL_NODE_SCOPE_WAIT_MS=1000
  AMPCCL_LOG_LEVEL=off
  AMPCCL_MIN_MSG_SIZE is not set here, so code default is 4MB unless you export it.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  baseline|adaptive) ;;
  *) echo "ERROR: MODE must be baseline or adaptive, got ${MODE}" >&2; exit 1 ;;
esac
case "${OP}" in
  allgather|allreduce|reducescatter|alltoall) ;;
  *) echo "ERROR: unsupported OP=${OP}" >&2; exit 1 ;;
esac

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found: ${BIN}" >&2
  echo "Build first: cd ${SCRIPT_DIR} && ./build.sh" >&2
  exit 1
fi

if [[ -z "${COUNT}" ]]; then
  if (( SIZE_BYTES <= 0 || SIZE_BYTES % 4 != 0 )); then
    echo "ERROR: SIZE_BYTES must be positive and divisible by 4, got ${SIZE_BYTES}" >&2
    exit 1
  fi
  COUNT=$((SIZE_BYTES / 4))
fi

run_cmd=()
if [[ "${MODE}" == "baseline" ]]; then
  run_cmd=(env -u LD_PRELOAD mpirun -np "${NP}" "${BIN}" --op="${OP}" --count="${COUNT}" --iters="${ITERS}" --warmup="${WARMUP}")
else
  [[ -f "${AMP_SO}" ]] || { echo "AMP so not found: ${AMP_SO}" >&2; exit 1; }
  export LD_PRELOAD="${AMP_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
  export AMPCCL_ENABLE=1
  export AMPCCL_ENABLE_PCIE=1
  export AMPCCL_ALGO=split_ratio #tcp / static
  export AMPCCL_SHM_PARAMS=1
  export AMPCCL_ENABLE_NCCL_V=0
  export AMPCCL_INTRA_NODE_ONLY=1
  export AMPCCL_NODE_SCOPE_WAIT_MS=1000
  export AMPCCL_LOG_LEVEL=off #info / debug
  unset AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID
  # MPICH/Hydra mpirun: -genvall propagates the full (exported) environment,
  # replacing OpenMPI-style "-x VAR" which Hydra does not understand.
  run_cmd=(mpirun -np "${NP}" -genvall \
    "${BIN}" --op="${OP}" --count="${COUNT}" --iters="${ITERS}" --warmup="${WARMUP}")
fi

echo "=== mode=${MODE} op=${OP} np=${NP} size_bytes=${SIZE_BYTES} count=${COUNT} iters=${ITERS} warmup=${WARMUP} ==="
echo "Command: ${run_cmd[*]}"

if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
  log_file="${LOG_DIR}/${OP}_${NP}gpu_${MODE}_${SIZE_BYTES}B_$(date +%Y%m%d_%H%M%S).log"
  "${run_cmd[@]}" 2>&1 | tee "${log_file}"
  echo "Log: ${log_file}"
else
  exec "${run_cmd[@]}"
fi
