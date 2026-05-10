#!/bin/bash
# AllReduce multi-case: 1024 bytes to 1G, 2/4/8 cards, iters=20
# send_count = size / data_byte; each step size *= size_step
# Logs to logs/allreduce/, filename includes card count and timestamp (YYYY-MM-DD_HHMM)

ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_ROOT="${SCRIPT_DIR}/logs"
LOG_DIR="${LOGS_ROOT}/allreduce"
TS="$(date +%Y-%m-%d_%H%M)"

allreduce_dir="${ADAPTIVE_CCL_ROOT}/tests/hccl_test/allreduce"
run_sh="${allreduce_dir}/run.sh"
exe="${allreduce_dir}/hccl_allreduce_param"

begin_size=1024
end_size=$((1024*1024*1024))   # 1G
size_step=4
data_byte=4
iters=20

mkdir -p "${LOG_DIR}"

if [ ! -f "${run_sh}" ]; then
  echo "Error: ${run_sh} not found."
  exit 1
fi
if [ ! -x "${exe}" ]; then
  echo "Error: ${exe} not found or not executable. Build allreduce_param first."
  exit 1
fi

echo "AllReduce multi-case: begin_size=${begin_size} end_size=${end_size} size_step=${size_step} data_byte=${data_byte} iters=${iters}"
echo "Log dir: ${LOG_DIR}"
echo ""

# ---------- 2-card experiment (comment out entire block if not needed) ----------
LOG_FILE="${LOG_DIR}/allreduce_2gpu_${TS}.log"
echo "========== devices=2 => ${LOG_FILE} =========="
{
  echo "========== devices=2 =========="
  devices=2
  size=$begin_size
  while [ $size -le $end_size ]; do
    send_count=$(( size / data_byte ))
    echo "--- size=${size} (send_count=${send_count}) ---"
    "${run_sh}" "${exe}" "$devices" "$send_count" "$iters"
    size=$(( size * size_step ))
  done
} | tee "${LOG_FILE}"
echo ""

# ---------- 4-card experiment (comment out entire block if not needed) ----------
LOG_FILE="${LOG_DIR}/allreduce_4gpu_${TS}.log"
echo "========== devices=4 => ${LOG_FILE} =========="
{
  echo "========== devices=4 =========="
  devices=4
  size=$begin_size
  while [ $size -le $end_size ]; do
    send_count=$(( size / data_byte ))
    echo "--- size=${size} (send_count=${send_count}) ---"
    "${run_sh}" "${exe}" "$devices" "$send_count" "$iters"
    size=$(( size * size_step ))
  done
} | tee "${LOG_FILE}"
echo ""

# ---------- 8-card experiment (comment out entire block if not needed) ----------
LOG_FILE="${LOG_DIR}/allreduce_8gpu_${TS}.log"
echo "========== devices=8 => ${LOG_FILE} =========="
{
  echo "========== devices=8 =========="
  devices=8
  size=$begin_size
  while [ $size -le $end_size ]; do
    send_count=$(( size / data_byte ))
    echo "--- size=${size} (send_count=${send_count}) ---"
    "${run_sh}" "${exe}" "$devices" "$send_count" "$iters"
    size=$(( size * size_step ))
  done
} | tee "${LOG_FILE}"
echo ""

echo "AllReduce multi-case done. Logs: ${LOG_DIR}/*_${TS}.log"
