#!/bin/bash
# PCIe interference experiment: run allgather_with_pcie, 4 cards, adjustable interference period and copy size
# Logs to logs/pcie_interference/, filename includes card count and timestamp (YYYY-MM-DD_HHMM)

ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_ROOT="${SCRIPT_DIR}/logs"
LOG_DIR="${LOGS_ROOT}/pcie_interference"
TS="$(date +%Y-%m-%d_%H%M)"

allgather_dir="${ADAPTIVE_CCL_ROOT}/tests/hccl_test/allgather"
run_sh="${allgather_dir}/run.sh"
exe="${allgather_dir}/hccl_allgather_with_pcie"

devices=4
send_count=$(( 64 * 1024 * 1024 ))   # 64M floats
iters=20
period_ms=50
interference_size=$(( 256 * 1024 * 1024 ))   # 256MB D2H per wave

mkdir -p "${LOG_DIR}"

if [ ! -f "${run_sh}" ]; then
  echo "Error: ${run_sh} not found."
  exit 1
fi
if [ ! -x "${exe}" ]; then
  echo "Error: ${exe} not found or not executable. Build allgather_with_pcie first."
  exit 1
fi

LOG_FILE="${LOG_DIR}/pcie_interference_${devices}gpu_${TS}.log"
echo "========== PCIe interference (devices=${devices}) => ${LOG_FILE} =========="
{
  echo "========== devices=${devices} period_ms=${period_ms} interference_size=${interference_size} =========="
  "${run_sh}" "${exe}" --devices="${devices}" --send-count="${send_count}" --iters="${iters}" \
    --interference-period-ms="${period_ms}" --interference-size="${interference_size}"
} | tee "${LOG_FILE}"

echo ""
echo "PCIe interference run done. Log: ${LOG_FILE}"
