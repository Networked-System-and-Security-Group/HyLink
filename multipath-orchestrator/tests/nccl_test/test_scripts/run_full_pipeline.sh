#!/bin/bash
# Full pipeline: parse logs → compute standard bandwidth → plot charts
#
# Usage:
#   bash run_full_pipeline.sh [--trim-n 8] [--baseline-only] [--skip-plots]
#
# Prerequisites:
#   1. Run compare_multi_case.sh first to generate logs in logs/compare/
#   2. Python 3 with matplotlib available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_N="${TRIM_N:-8}"
RUN_PLOTS="${RUN_PLOTS:-1}"
MPI_MODE=0

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

Options:
  --trim-n N        Number of highest/lowest samples to trim (default: 8)
  --skip-plots      Skip the plotting step
  --bw-column COL   Bandwidth column to plot: busbw_gbps (default) or algbw_gbps
  --mpi             Process the multi-process (MPI) experiment instead:
                    logs from ./logs/mpi_compare, CSV to ./csv_mpi
  --help            Show this help

Environment:
  CSV_DIR           Output directory for CSV files (default: ./csv, or ./csv_mpi with --mpi)
  LOGS_ROOT         Logs directory (default: ./logs/compare, or ./logs/mpi_compare with --mpi)

Pipeline:
  1. process_compare_logs.py  → csv/raw_bandwidth.csv, csv/summary_bandwidth.csv
  2. compute_standard_bw.py   → csv/standard_bw.csv
  3. plot_bw_comparison.py    → csv/plots/{allgather,allreduce,alltoall,reducescatter}_bw.png
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trim-n)
            TRIM_N="$2"
            shift 2
            ;;
        --skip-plots)
            RUN_PLOTS=0
            shift
            ;;
        --bw-column)
            BW_COLUMN="$2"
            shift 2
            ;;
        --mpi)
            MPI_MODE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${MPI_MODE}" == "1" ]]; then
    CSV_DIR="${CSV_DIR:-${SCRIPT_DIR}/csv_mpi}"
    LOGS_ROOT="${LOGS_ROOT:-${SCRIPT_DIR}/logs/mpi_compare}"
else
    CSV_DIR="${CSV_DIR:-${SCRIPT_DIR}/csv}"
    LOGS_ROOT="${LOGS_ROOT:-${SCRIPT_DIR}/logs/compare}"
fi

echo "============================================"
echo "Bandwidth Processing Pipeline"
echo "============================================"
echo "Script dir:  ${SCRIPT_DIR}"
echo "Logs root:   ${LOGS_ROOT}"
echo "CSV dir:     ${CSV_DIR}"
echo "Trim N:      ${TRIM_N}"
echo "Run plots:   ${RUN_PLOTS}"
echo "============================================"
echo ""

# Step 1: Parse logs → raw & summary CSV
echo "=== Step 1: process_compare_logs.py ==="
python3 "${SCRIPT_DIR}/process_compare_logs.py" \
    --logs-root "${LOGS_ROOT}" \
    --csv-dir "${CSV_DIR}"
echo ""

# Step 2: Compute standard bandwidth table
echo "=== Step 2: compute_standard_bw.py ==="
python3 "${SCRIPT_DIR}/compute_standard_bw.py" \
    --csv-dir "${CSV_DIR}" \
    --trim-n "${TRIM_N}"
echo ""

# Step 3: Plot charts
if [[ "${RUN_PLOTS}" == "1" ]]; then
    echo "=== Step 3: plot_bw_comparison.py ==="
    PLOT_ARGS=(
        --csv-dir "${CSV_DIR}"
    )
    if [[ -n "${BW_COLUMN:-}" ]]; then
        PLOT_ARGS+=(--bw-column "${BW_COLUMN}")
    fi
    python3 "${SCRIPT_DIR}/plot_bw_comparison.py" "${PLOT_ARGS[@]}"
    echo ""
fi

echo "============================================"
echo "Pipeline complete!"
echo "  Raw CSV:         ${CSV_DIR}/raw_bandwidth.csv"
echo "  Summary CSV:     ${CSV_DIR}/summary_bandwidth.csv"
echo "  Standard BW CSV: ${CSV_DIR}/standard_bw.csv"
if [[ "${RUN_PLOTS}" == "1" ]]; then
    echo "  Plots:           ${CSV_DIR}/plots/"
fi
echo "============================================"
