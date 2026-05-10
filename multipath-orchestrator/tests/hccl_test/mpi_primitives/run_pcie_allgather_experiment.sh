#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
BIN="${BIN:-${SCRIPT_DIR}/hccl_mpi_primitives}"
AMP_CCL_PATH="${AMP_CCL_PATH:-${ADAPTIVE_CCL_ROOT}/build/libampccl_hccl.so}"
NP="${NP:-8}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"
AG_TOTAL_BYTES="${AG_TOTAL_BYTES:-$((1024 * 1024 * 1024))}"
PCIE_SIZE_BYTES="${PCIE_SIZE_BYTES:-$((16 * 1024 * 1024))}"
PCIE_RANGES="${PCIE_RANGES:-10-20,30-40}"
MODE="${MODE:-smoke}"
STATIC_ALPHA="${STATIC_ALPHA:-0.08}"
DATA_ROOT="${DATA_ROOT:-/data/hccl_mpi_pcie}"
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/../images}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

usage() {
    cat <<'USAGE'
Usage:
  ./run_pcie_allgather_experiment.sh [--dry-run]

Defaults are the smoke experiment:
  NP=8 ITERS=50 WARMUP=10
  AG_TOTAL_BYTES=1GiB PCIE_SIZE_BYTES=16MiB
  PCIE_RANGES=10-20,30-40 (left-closed/right-open)
  STATIC_ALPHA=0.08 (override to compare a different static alpha)

For the requested full experiment, override for example:
  MODE=full ITERS=1001 WARMUP=10 \
  PCIE_RANGES=100-200,300-600,700-900 \
  ./run_pcie_allgather_experiment.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
else
    DRY_RUN=0
fi
if [[ "$#" -gt 1 || ( "$#" -eq 1 && "${1}" != "--dry-run" && "${1}" != "-h" && "${1}" != "--help" ) ]]; then
    usage >&2
    exit 2
fi

die() {
    echo "ERROR: $*" >&2
    exit 1
}

bytes_label() {
    local value="$1"
    if (( value % (1024 * 1024 * 1024) == 0 )); then
        echo "$((value / (1024 * 1024 * 1024)))GiB"
    elif (( value % (1024 * 1024) == 0 )); then
        echo "$((value / (1024 * 1024)))MiB"
    elif (( value % 1024 == 0 )); then
        echo "$((value / 1024))KiB"
    else
        echo "${value}B"
    fi
}

validate_ranges() {
    local item start end
    IFS=',' read -r -a range_array <<< "${PCIE_RANGES}"
    for item in "${range_array[@]}"; do
        [[ "${item}" =~ ^([0-9]+)-([0-9]+)$ ]] || die "invalid PCIe range: ${item}"
        start="${BASH_REMATCH[1]}"
        end="${BASH_REMATCH[2]}"
        (( end > start )) || die "PCIe ranges must be left-closed/right-open with start < end: ${item}"
        (( end <= ITERS )) || die "PCIe range ${item} exceeds ITERS=${ITERS}"
    done
    return 0
}

[[ "${NP}" =~ ^[1-9][0-9]*$ ]] || die "NP must be positive"
[[ "${ITERS}" =~ ^[1-9][0-9]*$ ]] || die "ITERS must be positive"
[[ "${WARMUP}" =~ ^[0-9]+$ ]] || die "WARMUP must be non-negative"
(( AG_TOTAL_BYTES > 0 && AG_TOTAL_BYTES % NP == 0 )) || die "AG_TOTAL_BYTES must be divisible by NP"
(( AG_TOTAL_BYTES / NP > 0 && (AG_TOTAL_BYTES / NP) % 4 == 0 )) || die "per-rank AllGather size must be divisible by 4"
(( PCIE_SIZE_BYTES > 0 )) || die "PCIE_SIZE_BYTES must be positive"
[[ "${STATIC_ALPHA}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "STATIC_ALPHA must be a non-negative decimal"
validate_ranges

AG_PER_RANK_BYTES=$((AG_TOTAL_BYTES / NP))
SEND_COUNT=$((AG_PER_RANK_BYTES / 4))
AG_LABEL="$(bytes_label "${AG_TOTAL_BYTES}")"
PCIE_LABEL="$(bytes_label "${PCIE_SIZE_BYTES}")"
RUN_LABEL="${MODE}_continuous"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/ag_${AG_LABEL}_pcie_${PCIE_LABEL}_alpha_${STATIC_ALPHA}_${RUN_LABEL}_${STAMP}}"
IMAGE_PREFIX="${IMAGE_PREFIX:-${IMAGE_DIR}/allgather_${AG_LABEL}_pcie_${PCIE_LABEL}_alpha_${STATIC_ALPHA}_${RUN_LABEL}}"

echo "mode=${MODE} np=${NP} ag_total=${AG_LABEL} ag_per_rank=$(bytes_label "${AG_PER_RANK_BYTES}") " \
     "send_count=${SEND_COUNT} iters=${ITERS} warmup=${WARMUP}"
echo "pcie_size=${PCIE_LABEL} pcie_ranges=${PCIE_RANGES} (left-closed/right-open)"
echo "pcie_mode=continuous static_alpha=${STATIC_ALPHA} algorithms=static:${STATIC_ALPHA},tcp"
echo "logs=${LOG_ROOT}"
echo "image=${IMAGE_PREFIX}.png"

if (( DRY_RUN == 1 )); then
    exit 0
fi

[[ -x "${BIN}" ]] || die "binary not found: ${BIN}; build with ./build.sh"
[[ -f "${AMP_CCL_PATH}" ]] || die "Adaptive-CCL library not found: ${AMP_CCL_PATH}"
command -v mpirun >/dev/null 2>&1 || die "mpirun not found"

mkdir -p "${LOG_ROOT}" "${IMAGE_DIR}"

export ACL_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_LOG_LEVEL=off
export AMPCCL_STATIC_ALPHA="${STATIC_ALPHA}"
export AMPCCL_MIN_CHUNK_SIZE="${AMPCCL_MIN_CHUNK_SIZE:-20}"
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK="${AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK:-0}"
unset AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID

PCIECCL_ROOT="${PCIECCL_ROOT:-$(cd "${ADAPTIVE_CCL_ROOT}/.." 2>/dev/null && pwd)/pcieccl}"
if [[ -z "${PCCL_IR_PATH:-}" && -d "${PCIECCL_ROOT}/ir" ]]; then
    export PCCL_IR_PATH="${PCIECCL_ROOT}/ir"
fi

run_algorithm() {
    local algorithm="$1"
    local log_path="${LOG_ROOT}/${algorithm}.log"
    export AMPCCL_ALGO="${algorithm}"
    echo "=== algorithm=${algorithm} log=${log_path} ==="
    {
        echo "config: algorithm=${algorithm} static_alpha=${STATIC_ALPHA} np=${NP} ag_total_bytes=${AG_TOTAL_BYTES} " \
             "ag_per_rank_bytes=${AG_PER_RANK_BYTES} iters=${ITERS} warmup=${WARMUP} " \
             "pcie_size_bytes=${PCIE_SIZE_BYTES} pcie_ranges=${PCIE_RANGES} pcie_mode=continuous"
        echo "command: mpirun -np ${NP} -genvall ${BIN} --op=ag --size=${SEND_COUNT} " \
             "--iters=${ITERS} --warmup=${WARMUP} --pcie-ranges=${PCIE_RANGES} --pcie-size=${PCIE_SIZE_BYTES}"
        LD_PRELOAD="${AMP_CCL_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}" \
            mpirun -np "${NP}" -genvall \
            "${BIN}" --op=ag --size="${SEND_COUNT}" --iters="${ITERS}" --warmup="${WARMUP}" \
            --pcie-ranges="${PCIE_RANGES}" --pcie-size="${PCIE_SIZE_BYTES}"
    } 2>&1 | tee "${log_path}"
}

run_algorithm static
run_algorithm tcp

python3 "${SCRIPT_DIR}/plot_pcie_allgather.py" \
    --static-log "${LOG_ROOT}/static.log" \
    --tcp-log "${LOG_ROOT}/tcp.log" \
    --ranges "${PCIE_RANGES}" \
    --size-bytes "${AG_PER_RANK_BYTES}" \
    --ranks "${NP}" \
    --output "${IMAGE_PREFIX}.png" \
    --csv-output "${LOG_ROOT}/algbw.csv" \
    --static-alpha "${STATIC_ALPHA}" \
    --title "AllGather ${AG_LABEL} with ${PCIE_LABEL} PCIe Interference"

echo "experiment_done logs=${LOG_ROOT} image=${IMAGE_PREFIX}.png"
