#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/mpi_primitives/run.sh"
PLOTTER="${SCRIPT_DIR}/mpi_primitives/plot_pcie_allgather.py"
REGULATION_PLOTTER="${SCRIPT_DIR}/plot_4card_allgather_regulation.py"

# Experiment configuration: 4 cards, AllGather, 256MB sendBuffer per card
NP=4
OP=ag
AG_PER_RANK_BYTES=$((256 * 1024 * 1024))
AG_TOTAL_BYTES=$((AG_PER_RANK_BYTES * NP))
SIZE=$((AG_PER_RANK_BYTES / 4)) # 67,108,864 float elements
ITERS=80
WARMUP=5

# PCIe contention: 128MB in iters [20, 40) and [55, 65)
PCIE_SIZE_BYTES=$((128 * 1024 * 1024))
PCIE_RANGES="20-40,55-65"

SPLIT_RATIO_ETA=0.10
SPLIT_RATIO_SMOOTH=0.40
SPLIT_RATIO_INITIAL_ALPHA=0.15
SPLIT_RATIO_MIN_ALPHA=0.03

DATA_ROOT="${DATA_ROOT:-/logs/allgather_4card_256mb}"
IMAGE_DIR="${SCRIPT_DIR}/images"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/ag_4card_256MB_pcie128MiB_smoke80_${STAMP}}"
IMAGE_PREFIX="${IMAGE_DIR}/allgather_4card_256MB_split_ratio_vs_tcp_smoke80_${STAMP}"
ARTIFACT_DIR="/root/.gemini/antigravity-cli/brain/4b4c7764-3a7c-447e-bd3f-12b560c00703"

mkdir -p "${LOG_ROOT}" "${IMAGE_DIR}"

[[ -x "${RUNNER}" ]] || { echo "Runner not found: ${RUNNER}" >&2; exit 1; }
[[ -f "${PLOTTER}" ]] || { echo "Plotter not found: ${PLOTTER}" >&2; exit 1; }
[[ -f "${REGULATION_PLOTTER}" ]] || { echo "Regulation Plotter not found: ${REGULATION_PLOTTER}" >&2; exit 1; }

export NP OP SIZE ITERS WARMUP
export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_SHM_PARAMS=1
export AMPCCL_LOG_LEVEL=off
export PCCL_H2H_WORKERS=8
export PCCL_H2H_WORKERS_PER_NUMA=8
export ACL_VISIBLE_DEVICES="0,1,2,3"
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"

echo "=========================================================="
echo "Phase 1: Running split_ratio (80 iters, 4-card 256MB AllGather)"
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
echo "Phase 2: Running tcp (AIMD) (80 iters, 4-card 256MB AllGather)"
echo "  Contention: 128MB PCIe in ranges ${PCIE_RANGES}"
echo "=========================================================="

TCP_LOG="${LOG_ROOT}/tcp.log"
export AMPCCL_ALGO="tcp"

{
    echo "config: algorithm=tcp np=${NP} op=${OP} size=${SIZE} iters=${ITERS} warmup=${WARMUP}"
    echo "command: ${RUNNER} --pcie-ranges=${PCIE_RANGES} --pcie-size=${PCIE_SIZE_BYTES}"
    "${RUNNER}" --pcie-ranges="${PCIE_RANGES}" --pcie-size="${PCIE_SIZE_BYTES}"
} 2>&1 | tee "${TCP_LOG}"

echo ""
echo "=========================================================="
echo "Phase 3: Plotting Results into tests/hccl_test/images/"
echo "=========================================================="

# 1. Comparison between split_ratio and tcp using plot_pcie_allgather.py
COMMON_PLOT_ARGS=(
    --static-log "${TCP_LOG}"
    --tcp-log "${SPLIT_LOG}"
    --ranges "${PCIE_RANGES}"
    --size-bytes "${AG_PER_RANK_BYTES}"
    --ranks "${NP}"
    --second-label "split_ratio"
    --paper
    --line-width 0.8
    --reference-style
    --title "4-Card 256MB AllGather: split_ratio vs tcp (128MB PCIe Contention)"
)

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output "${IMAGE_PREFIX}_bandwidth.png" \
    --csv-output "${LOG_ROOT}/algbw_comparison.csv" 2>/dev/null || true

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --spike-threshold 20 \
    --output "${IMAGE_PREFIX}_bandwidth_smooth20.png" \
    --denoised-csv-output "${LOG_ROOT}/algbw_smooth20.csv" 2>/dev/null || true

python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --spike-threshold "20,14" \
    --window-envelope \
    --envelope-window 11 \
    --envelope-sigma 1.0 \
    --output "${IMAGE_PREFIX}_window_envelope_w11_1sigma.png" 2>/dev/null || true

# 2. Path timings and breakdown for split_ratio
python3 "${PLOTTER}" \
    --path-log "${SPLIT_LOG}" \
    --output "${IMAGE_PREFIX}_split_ratio_path_bandwidth.png" \
    --time-output "${IMAGE_PREFIX}_split_ratio_path_time.png" \
    --title "4-Card 256MB AllGather: split_ratio Path Bandwidth" 2>/dev/null || true

# 3. Path timings and breakdown for tcp
python3 "${PLOTTER}" \
    --path-log "${TCP_LOG}" \
    --output "${IMAGE_PREFIX}_tcp_path_bandwidth.png" \
    --time-output "${IMAGE_PREFIX}_tcp_path_time.png" \
    --title "4-Card 256MB AllGather: tcp Path Bandwidth" 2>/dev/null || true

# 4. Multi-panel feedback regulation comparison
python3 "${REGULATION_PLOTTER}" \
    --split-log "${SPLIT_LOG}" \
    --tcp-log "${TCP_LOG}" \
    --ranges "${PCIE_RANGES}" \
    --output "${IMAGE_PREFIX}_regulation_3panel.png"

# Copy generated images to standard images/ and artifact dir
cp -f "${IMAGE_PREFIX}"* "${IMAGE_DIR}/" 2>/dev/null || true
if [[ -d "${ARTIFACT_DIR}" ]]; then
    cp -f "${IMAGE_PREFIX}"* "${ARTIFACT_DIR}/" 2>/dev/null || true
fi

echo ""
echo "=========================================================="
echo "Phase 4: Statistical Evaluation & Quantitative Summary"
echo "=========================================================="

python3 - <<'EOF'
import re, sys
from pathlib import Path
import statistics

def analyze_log(log_path, name):
    records = {}
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
                records[current_iter] = (bw, alpha)
                current_iter = None

    if not records:
        print(f"[{name}] No records found in {log_path}")
        return

    # Ranges: [20, 40) and [55, 65)
    normal_iters = [it for it in records if not ((20 <= it < 40) or (55 <= it < 65))]
    contention1_iters = [it for it in records if 20 <= it < 40]
    contention2_iters = [it for it in records if 55 <= it < 65]
    all_contention = contention1_iters + contention2_iters

    def stats(iters):
        if not iters:
            return 0.0, 0.0, 0.0, 0.0
        bws = [records[it][0] for it in iters]
        alphas = [records[it][1] for it in iters]
        mean_bw = statistics.mean(bws)
        std_bw = statistics.stdev(bws) if len(bws) > 1 else 0.0
        cv_bw = (std_bw / mean_bw * 100) if mean_bw > 0 else 0.0
        mean_a = statistics.mean(alphas)
        return mean_bw, std_bw, cv_bw, mean_a

    norm_bw, norm_std, norm_cv, norm_a = stats(normal_iters)
    c1_bw, c1_std, c1_cv, c1_a = stats(contention1_iters)
    c2_bw, c2_std, c2_cv, c2_a = stats(contention2_iters)
    all_c_bw, all_c_std, all_c_cv, all_c_a = stats(all_contention)
    tot_bw, tot_std, tot_cv, tot_a = stats(list(records.keys()))

    print(f"=== {name} Performance Summary ===")
    print(f"  Normal (50 iters):     {norm_bw:6.2f} ± {norm_std:5.2f} GB/s  (CV: {norm_cv:4.1f}%, mean alpha: {norm_a:.3f})")
    print(f"  Contention 1 [20,40):  {c1_bw:6.2f} ± {c1_std:5.2f} GB/s  (CV: {c1_cv:4.1f}%, mean alpha: {c1_a:.3f})")
    print(f"  Contention 2 [55,65):  {c2_bw:6.2f} ± {c2_std:5.2f} GB/s  (CV: {c2_cv:4.1f}%, mean alpha: {c2_a:.3f})")
    print(f"  All Contention (30 it):{all_c_bw:6.2f} ± {all_c_std:5.2f} GB/s  (CV: {all_c_cv:4.1f}%, mean alpha: {all_c_a:.3f})")
    print(f"  Overall (80 iters):    {tot_bw:6.2f} ± {tot_std:5.2f} GB/s  (CV: {tot_cv:4.1f}%, mean alpha: {tot_a:.3f})")
    print("")

if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        analyze_log(sys.argv[1], "split_ratio")
    if len(sys.argv) > 2 and Path(sys.argv[2]).is_file():
        analyze_log(sys.argv[2], "tcp")
EOF
"${SPLIT_LOG}" "${TCP_LOG}"

echo "=========================================================="
echo "Experiment completed successfully!"
echo "Logs saved in: ${LOG_ROOT}"
echo "Images saved in: ${IMAGE_DIR}"
echo "Prefix: ${IMAGE_PREFIX}"
echo "=========================================================="
