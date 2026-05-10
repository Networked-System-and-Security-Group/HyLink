#!/usr/bin/env bash
# 2x2 AllGather ablation: PCIeCCL route on/off x continuous bidirectional load on/off.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
root="${LOG_ROOT:-/logs/pccl_background_factorial_$(date -u +%Y%m%d_%H%M%S)}"
blocks="${BLOCKS:-3}"
np=4
input_bytes=$((512 * 1024 * 1024))
bg_bytes=$((64 * 1024 * 1024))
iters=1001
warmup=10
ranges="100-200,300-600,700-900"
static_alpha=0.1945
bin="${here}/bin/hccl_mpi_primitives"

test -x "${bin}" || { echo "Missing benchmark binary: ${bin}" >&2; exit 1; }
test -f "${repo}/build/libampccl_hccl.so" || { echo "Missing Adaptive-CCL library" >&2; exit 1; }
test -f "${repo}/../pcieccl/build/libpccl.so" || { echo "Missing PCIeCCL library" >&2; exit 1; }
mkdir -p "${root}/blocks"

while IFS='=' read -r name _; do
  case "${name}" in AMPCCL_*|PCCL_*|CCL_*) unset "${name}" ;; esac
done < <(env)
unset LD_PRELOAD BENCHMARK_DEBUG_TIMING PCCL_FORCE_DMA_LANES

export NP="${np}" OP=ag SIZE=$((input_bytes / 4)) ITERS="${iters}" WARMUP="${warmup}"
export ACL_VISIBLE_DEVICES=0,1,2,3 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export AMPCCL_ENABLE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_ALGO=static AMPCCL_STATIC_ALPHA="${static_alpha}"
export AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export AMPCCL_TCP_INITIAL_ALPHA=0.075
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir"
export AMP_CCL_PATH="${repo}/build/libampccl_hccl.so" BIN="${bin}"
unset CCL_MSPROF_MARKERS PCCL_MSPROF_HOST_TIMING PCCL_MSPROF_CPU_SAMPLE_PERIOD PCCL_DEBUG_TIMING
unset AMPCCL_FORCE_DISABLE AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK AMPCCL_SYNC_TIMER_EVENTS_IN_CALLBACK

{
  echo "design=2x2 PCIeCCL_route x bidirectional_background"
  echo "np=${np} input_bytes=${input_bytes} total_bytes=$((np * input_bytes)) iters=${iters} warmup=${warmup}"
  echo "background_bytes_per_direction=${bg_bytes} ranges=${ranges} static_alpha=${static_alpha} blocks=${blocks}"
  echo "pcie_on=AMPCCL_ENABLE_PCIE=1; pcie_off=AMPCCL_ENABLE_PCIE=0 (AMPCCL fast-only path)"
  git -C "${repo}" log -1 --oneline
  git -C "${repo}/../pcieccl" log -1 --oneline
  sha256sum "${bin}" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so"
} >"${root}/metadata.txt"

run_case() {
  local block="$1" condition="$2" pcie_on="$3" bg_on="$4"
  local out="${root}/blocks/${block}/${condition}"
  local bytes=0 args=()
  mkdir -p "${out}"
  export AMPCCL_ENABLE_PCIE="${pcie_on}"
  if [[ "${bg_on}" == 1 ]]; then
    bytes="${bg_bytes}"
    args+=(--pcie-size="${bg_bytes}" --pcie-ranges="${ranges}" --pcie-direction=both)
  fi
  {
    echo "algorithm=static condition=${condition} np=${np} input_bytes=${input_bytes} total_bytes=$((np * input_bytes)) interference_bytes=${bytes} background_enabled=${bg_on} pcie_route_enabled=${pcie_on} iters=${iters} warmup=${warmup} static_alpha=${static_alpha} pcie_direction=both ranges=${ranges} input_mode=reuse_once validation=final_only"
    timeout --signal=TERM --kill-after=10s 3600s \
      bash "${here}/../mpi_primitives/run.sh" --reuse-input "${args[@]}"
  } >"${out}/static.log" 2>&1
  echo "completed=${block}/${condition} at $(date -u +%FT%TZ)" | tee -a "${root}/progress.txt"
}

# Rotate and reverse the four-cell order across blocks to reduce time/order bias.
orders=(
  "pcie_on_bg_off pcie_on_bg_on pcie_off_bg_on pcie_off_bg_off"
  "pcie_off_bg_off pcie_off_bg_on pcie_on_bg_on pcie_on_bg_off"
  "pcie_on_bg_on pcie_off_bg_off pcie_on_bg_off pcie_off_bg_on"
)
for ((block=1; block<=blocks; ++block)); do
  read -r -a order <<<"${orders[$(((block - 1) % ${#orders[@]}))]}"
  for condition in "${order[@]}"; do
    case "${condition}" in
      pcie_on_bg_off) run_case "block_${block}" "${condition}" 1 0 ;;
      pcie_on_bg_on)  run_case "block_${block}" "${condition}" 1 1 ;;
      pcie_off_bg_off) run_case "block_${block}" "${condition}" 0 0 ;;
      pcie_off_bg_on)  run_case "block_${block}" "${condition}" 0 1 ;;
    esac
  done
done

python3 "${here}/summarize_pccl_background_factorial.py" "${root}"
echo "Done: ${root}"
