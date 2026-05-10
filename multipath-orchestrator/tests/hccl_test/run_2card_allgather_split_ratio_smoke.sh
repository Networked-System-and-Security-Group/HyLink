#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"
PLOTTER="${SCRIPT_DIR}/mpi_primitives/plot_pcie_allgather.py"

# Experiment configuration: 2 cards, AllGather, 256MB sendBuffer per card
NP=2
OP=ag
AG_PER_RANK_BYTES=$((256 * 1024 * 1024))
AG_TOTAL_BYTES=$((AG_PER_RANK_BYTES * NP))
SIZE=$((AG_PER_RANK_BYTES / 4)) # 67,108,864 float elements
ITERS=80
WARMUP=5

# PCIe contention: 128MB in iters [20, 40) and [50, 60)
PCIE_SIZE_BYTES=$((128 * 1024 * 1024))
PCIE_RANGES="20-40,50-60"

SPLIT_RATIO_ETA=0.10
SPLIT_RATIO_SMOOTH=0.50
SPLIT_RATIO_INITIAL_ALPHA=0.20
SPLIT_RATIO_MIN_ALPHA=0.03

DATA_ROOT="${DATA_ROOT:-/logs/allgather_2card_256mb}"
IMAGE_DIR="${SCRIPT_DIR}/images"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/ag_2card_256MB_pcie128MiB_smoke80_${STAMP}}"
IMAGE_PREFIX="${IMAGE_DIR}/allgather_2card_256MB_split_ratio_smoke80_${STAMP}"
ARTIFACT_DIR="/root/.gemini/antigravity-cli/brain/5d7caaef-ee8a-4f9b-8629-bcc4639c6268"

mkdir -p "${LOG_ROOT}" "${IMAGE_DIR}"

[[ -x "${RUNNER}" ]] || { echo "Runner not found: ${RUNNER}" >&2; exit 1; }
[[ -f "${PLOTTER}" ]] || { echo "Plotter not found: ${PLOTTER}" >&2; exit 1; }

export NP OP SIZE ITERS WARMUP
export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_SHM_PARAMS=1
export AMPCCL_LOG_LEVEL=off
export PCCL_H2H_WORKERS=8
export PCCL_H2H_WORKERS_PER_NUMA=8
export ACL_VISIBLE_DEVICES="0,1"
export ASCEND_RT_VISIBLE_DEVICES="0,1"

echo "=========================================================="
echo "Phase 1: Running split_ratio (80 iters, 2-card 256MB AllGather)"
echo "  Contention: 128MB PCIe in ranges ${PCIE_RANGES}"
echo "=========================================================="

SPLIT_LOG="${LOG_ROOT}/split_ratio.log"
export AMPCCL_ALGO="split_ratio"
export AMPCCL_SPLIT_RATIO_INITIAL_ALPHA="${SPLIT_RATIO_INITIAL_ALPHA}"
export AMPCCL_SPLIT_RATIO_ETA="${SPLIT_RATIO_ETA}"
export AMPCCL_SPLIT_RATIO_SMOOTH="${SPLIT_RATIO_SMOOTH}"
export AMPCCL_SPLIT_RATIO_MIN_ALPHA="${SPLIT_RATIO_MIN_ALPHA}"

{
    echo "config: algorithm=split_ratio np=${NP} op=${OP} size=${SIZE} iters=${ITERS} warmup=${WARMUP}"
    echo "command: ${RUNNER} --pcie-ranges=${PCIE_RANGES} --pcie-size=${PCIE_SIZE_BYTES}"
    "${RUNNER}" --pcie-ranges="${PCIE_RANGES}" --pcie-size="${PCIE_SIZE_BYTES}"
} 2>&1 | tee "${SPLIT_LOG}"

echo ""
echo "=========================================================="
echo "Phase 2: Analyzing un-interfered iterations in split_ratio"
echo "=========================================================="

BEST_STATIC_ALPHA=$(python3 - <<'EOF'
import re, sys
from pathlib import Path

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("split_ratio.log")
clean_alphas = []
clean_bws = []
current_iter = None

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m_it = re.search(r"\[rank0\]\s+begin formal\s+(\d+)\s+iters", line)
        if m_it:
            current_iter = int(m_it.group(1))
            continue
        m_agg = re.search(r"aggregate\s+op=AllGather.*?\bsplit_pcie=([0-9.]+)\b.*?\balgbw\(GB/s,decimal\)=([0-9.eE+-]+)", line)
        if m_agg and current_iter is not None:
            alpha = float(m_agg.group(1))
            bw = float(m_agg.group(2))
            is_contention = (20 <= current_iter < 40) or (50 <= current_iter < 60)
            # Un-interfered stable iterations: iter 5..19, 43..49, 63..79
            is_stable_clean = (5 <= current_iter < 20) or (43 <= current_iter < 50) or (63 <= current_iter < 80)
            if is_stable_clean:
                clean_alphas.append(alpha)
                clean_bws.append((bw, alpha))
            current_iter = None

if not clean_alphas:
    print("0.20")
else:
    # Sort by bandwidth descending, take mean alpha of top 5 bandwidth points
    clean_bws.sort(key=lambda x: x[0], reverse=True)
    top_alphas = [a for bw, a in clean_bws[:max(3, len(clean_bws)//4)]]
    best_alpha = round(sum(top_alphas) / len(top_alphas), 2)
    print(f"{best_alpha:.2f}")
EOF
"${SPLIT_LOG}")

echo "Extracted Best Static Alpha Candidate: ${BEST_STATIC_ALPHA}"
echo ""

echo "=========================================================="
echo "Phase 3: Running static (alpha=${BEST_STATIC_ALPHA}) for comparison"
echo "=========================================================="

STATIC_LOG="${LOG_ROOT}/static.log"
export AMPCCL_ALGO="static"
export AMPCCL_STATIC_ALPHA="${BEST_STATIC_ALPHA}"

{
    echo "config: algorithm=static static_alpha=${BEST_STATIC_ALPHA} np=${NP} op=${OP} size=${SIZE} iters=${ITERS} warmup=${WARMUP}"
    echo "command: ${RUNNER} --pcie-ranges=${PCIE_RANGES} --pcie-size=${PCIE_SIZE_BYTES}"
    "${RUNNER}" --pcie-ranges="${PCIE_RANGES}" --pcie-size="${PCIE_SIZE_BYTES}"
} 2>&1 | tee "${STATIC_LOG}"

echo ""
echo "=========================================================="
echo "Phase 4: Plotting with mpi_primitives/plot_pcie_allgather.py"
echo "=========================================================="

COMMON_PLOT_ARGS=(
    --static-log "${STATIC_LOG}"
    --tcp-log "${SPLIT_LOG}"
    --ranges "${PCIE_RANGES}"
    --size-bytes "${AG_PER_RANK_BYTES}"
    --ranks "${NP}"
    --static-alpha "${BEST_STATIC_ALPHA}"
    --paper
    --line-width 0.8
    --reference-style
    --title "AllGather 2-Card 256MB with 128MB PCIe Interference"
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

# Also generate path timings and breakdown for split_ratio
python3 "${PLOTTER}" \
    --path-log "${SPLIT_LOG}" \
    --output "${IMAGE_PREFIX}_split_ratio_path_bandwidth.png" \
    --time-output "${IMAGE_PREFIX}_split_ratio_path_time.png" \
    --title "2-Card 256MB AllGather: split_ratio Path Bandwidth" 2>/dev/null || true

# Copy generated images to standard images/ and artifact dir
cp -f "${IMAGE_PREFIX}"* "${IMAGE_DIR}/" 2>/dev/null || true
if [[ -d "${ARTIFACT_DIR}" ]]; then
    cp -f "${IMAGE_PREFIX}"* "${ARTIFACT_DIR}/" 2>/dev/null || true
fi

echo ""
echo "=========================================================="
echo "Experiment completed successfully!"
echo "Logs saved in: ${LOG_ROOT}"
echo "Images saved in: ${IMAGE_DIR}"
echo "Prefix: ${IMAGE_PREFIX}"
echo "=========================================================="
