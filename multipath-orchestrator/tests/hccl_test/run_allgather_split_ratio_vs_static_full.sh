#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"
PLOTTER="${SCRIPT_DIR}/mpi_primitives/plot_pcie_allgather.py"

# Fixed experiment configuration.
NP=4
OP=ag
AG_PER_RANK_BYTES=$((256 * 1024 * 1024))
AG_TOTAL_BYTES=$((AG_PER_RANK_BYTES * NP))
SIZE=$((AG_PER_RANK_BYTES / 4))
ITERS=1001
WARMUP=10
STATIC_ALPHA=0.18
SPLIT_RATIO_ETA=0.10
SPLIT_RATIO_SMOOTH=0.40
SPLIT_RATIO_INITIAL_ALPHA=0.155
SPLIT_RATIO_MIN_ALPHA=0.03

# Three continuous PCIe-contention stages, all left-closed/right-open.
PCIE_SIZE_BYTES=$((64 * 1024 * 1024))
PCIE_RANGES="100-200,300-600,700-900"

DATA_ROOT="${DATA_ROOT:-/logs/allgather_4card_256mb}"
IMAGE_DIR="${IMAGE_DIR:-${DATA_ROOT}/images}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/ag_4card_256MB_pcie_64MiB_split_ratio_eta${SPLIT_RATIO_ETA}_smooth${SPLIT_RATIO_SMOOTH}_static${STATIC_ALPHA}_full1001_${STAMP}}"
IMAGE_PREFIX="${IMAGE_PREFIX:-${IMAGE_DIR}/allgather_4card_256MB_split_ratio_vs_static${STATIC_ALPHA}_pcie64MiB_full1001}"

usage() {
    cat <<'USAGE'
Usage:
  ./run_allgather_split_ratio_vs_static_full.sh
  ./run_allgather_split_ratio_vs_static_full.sh --dry-run

Runs split_ratio and static(alpha=0.075) on an 8-process, 1GiB
AllGather for 1001 formal iterations with 10 warmup iterations.
PCIe contention is 64MiB in ranges [100,200), [300,600), [700,900).
USAGE
}

case "${1:-}" in
    "") ;;
    --dry-run)
        echo "config: np=${NP} op=${OP} size=${SIZE} iters=${ITERS} warmup=${WARMUP}"
        echo "ag_total_bytes=${AG_TOTAL_BYTES} ag_per_rank_bytes=${AG_PER_RANK_BYTES}"
        echo "static_alpha=${STATIC_ALPHA} split_ratio_eta=${SPLIT_RATIO_ETA} split_ratio_smooth=${SPLIT_RATIO_SMOOTH}"
        echo "enable_pcie=1 pccl_h2h_workers=8 pcie_size_bytes=${PCIE_SIZE_BYTES}"
        echo "pcie_ranges=${PCIE_RANGES} (left-closed/right-open)"
        echo "log_root=${LOG_ROOT}"
        echo "image_prefix=${IMAGE_PREFIX}"
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
[[ -f "${PLOTTER}" ]] || {
    echo "Plotter not found: ${PLOTTER}" >&2
    exit 1
}

command -v mpirun >/dev/null 2>&1 || {
    echo "mpirun not found" >&2
    exit 1
}

export NP OP SIZE ITERS WARMUP
export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_ALGO=split_ratio
export AMPCCL_STATIC_ALPHA="${STATIC_ALPHA}"
export AMPCCL_SPLIT_RATIO_INITIAL_ALPHA="${SPLIT_RATIO_INITIAL_ALPHA}"
export AMPCCL_SPLIT_RATIO_ETA="${SPLIT_RATIO_ETA}"
export AMPCCL_SPLIT_RATIO_SMOOTH="${SPLIT_RATIO_SMOOTH}"
export AMPCCL_SPLIT_RATIO_MIN_ALPHA="${SPLIT_RATIO_MIN_ALPHA}"
export AMPCCL_SHM_PARAMS=1
export AMPCCL_LOG_LEVEL=off
export PCCL_H2H_WORKERS=8
export PCCL_H2H_WORKERS_PER_NUMA=8
export ACL_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "${LOG_ROOT}" "${IMAGE_DIR}"

run_algorithm() {
    local algorithm="$1"
    local log_path="${LOG_ROOT}/${algorithm}.log"
    export AMPCCL_ALGO="${algorithm}"

    echo "Starting ${algorithm}: log=${log_path}"
    {
        echo "config: algorithm=${algorithm} static_alpha=${STATIC_ALPHA} " \
             "split_ratio_eta=${SPLIT_RATIO_ETA} split_ratio_smooth=${SPLIT_RATIO_SMOOTH} " \
             "np=${NP} op=${OP} size=${SIZE} ag_total_bytes=${AG_TOTAL_BYTES} " \
             "ag_per_rank_bytes=${AG_PER_RANK_BYTES} iters=${ITERS} warmup=${WARMUP} " \
             "enable_pcie=1 pccl_h2h_workers=8 pcie_size_bytes=${PCIE_SIZE_BYTES} " \
             "pcie_ranges=${PCIE_RANGES} pcie_mode=continuous"
        echo "command: ${RUNNER} --pcie-ranges=${PCIE_RANGES} --pcie-size=${PCIE_SIZE_BYTES}"
        "${RUNNER}" \
            --pcie-ranges="${PCIE_RANGES}" \
            --pcie-size="${PCIE_SIZE_BYTES}"
    } 2>&1 | tee "${log_path}"
}

run_algorithm split_ratio
run_algorithm static

COMMON_PLOT_ARGS=(
    --static-log "${LOG_ROOT}/static.log"
    --tcp-log "${LOG_ROOT}/split_ratio.log"
    --ranges "${PCIE_RANGES}"
    --size-bytes "${AG_PER_RANK_BYTES}"
    --ranks "${NP}"
    --static-alpha "${STATIC_ALPHA}"
    --y-min 50
    --y-max 95
    --paper
    --line-width 0.8
    --reference-style
    --title ""
)

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output "${IMAGE_PREFIX}_bandwidth.png" \
    --csv-output "${LOG_ROOT}/algbw.csv"

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --spike-threshold 20 \
    --output "${IMAGE_PREFIX}_bandwidth_smooth20.png" \
    --denoised-csv-output "${LOG_ROOT}/algbw_smooth20.csv"

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --spike-threshold "20,14" \
    --output "${IMAGE_PREFIX}_bandwidth_smooth20_14.png" \
    --denoised-csv-output "${LOG_ROOT}/algbw_smooth20_14.csv"

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --spike-threshold "20,14" \
    --window-envelope \
    --envelope-window 11 \
    --envelope-sigma 1.0 \
    --output "${IMAGE_PREFIX}_window_envelope_w11_1sigma.png"

mkdir -p "${SCRIPT_DIR}/images"
cp -f "${IMAGE_PREFIX}"* "${SCRIPT_DIR}/images/" 2>/dev/null || true

echo "experiment_done logs=${LOG_ROOT}"
echo "bandwidth_plot=${IMAGE_PREFIX}_bandwidth.png"
echo "bandwidth_smooth_plot=${IMAGE_PREFIX}_bandwidth_smooth20.png"
echo "envelope_plot=${IMAGE_PREFIX}_window_envelope_w11_1sigma.png"
