#!/usr/bin/env bash
# Repeat the 4-rank paired TCP/static candidate (64MiB each PCIe direction).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
root="${LOG_ROOT:-/logs/ag_tcp_static_paired_20261006_041100/repeats_4card_both64_$(date -u +%H%M%S)}"
repeats="${REPEATS:-3}"
input_bytes=536870912
interference_bytes=$((64 * 1024 * 1024))
ranges="100-200,300-600,700-900"
static_alpha=0.1945
tcp_initial=0.075
iters=1001
warmup=10
bin="${here}/bin/hccl_mpi_primitives"

test -x "${bin}" || { echo "Missing benchmark binary: ${bin}" >&2; exit 1; }
mkdir -p "${root}/pairs"
while IFS='=' read -r name _; do
  case "${name}" in AMPCCL_*|PCCL_*|CCL_*) unset "${name}" ;; esac
done < <(env)
unset LD_PRELOAD BENCHMARK_DEBUG_TIMING PCCL_FORCE_DMA_LANES

export NP=4 OP=ag SIZE=$((input_bytes / 4)) ITERS="${iters}" WARMUP="${warmup}"
export ACL_VISIBLE_DEVICES=0,1,2,3 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_TCP_INITIAL_ALPHA="${tcp_initial}" AMPCCL_STATIC_ALPHA="${static_alpha}"
export AMPCCL_TCP_HOLD_RATE_LOW="${TCP_HOLD_RATE_LOW:-0.95}"
export AMPCCL_TCP_HOLD_RATE_HIGH="${TCP_HOLD_RATE_HIGH:-1.10}"
export AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir"
export AMP_CCL_PATH="${repo}/build/libampccl_hccl.so" BIN="${bin}"
unset CCL_MSPROF_MARKERS PCCL_MSPROF_HOST_TIMING PCCL_MSPROF_CPU_SAMPLE_PERIOD PCCL_DEBUG_TIMING
unset AMPCCL_FORCE_DISABLE AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK AMPCCL_SYNC_TIMER_EVENTS_IN_CALLBACK

{
  echo "paired_repeats=${repeats} np=4 input_bytes=${input_bytes} total_bytes=$((4 * input_bytes))"
  echo "direction=both interference_bytes_per_direction=${interference_bytes} ranges=${ranges}"
  echo "static_alpha=${static_alpha} tcp_initial_alpha=${tcp_initial} iters=${iters} warmup=${warmup}"
  git -C "${repo}" log -1 --oneline
  git -C "${repo}/../pcieccl" log -1 --oneline
  sha256sum "${bin}" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so"
  env | rg '^(AMPCCL_|PCCL_|ASCEND_|ACL_|NP=|OP=|SIZE=|ITERS=|WARMUP=|BIN=)' | sort
} >"${root}/metadata.txt"

rep=1
while (( rep <= repeats )); do
  pair="${root}/pairs/repeat_${rep}"
  mkdir -p "${pair}"
  echo "repeat=${rep} order=$([[ $((rep % 2)) -eq 1 ]] && echo static,tcp || echo tcp,static)" >"${pair}/pair_metadata.txt"
  if (( rep % 2 )); then algorithms=(static tcp); else algorithms=(tcp static); fi
  for algorithm in "${algorithms[@]}"; do
    export AMPCCL_ALGO="${algorithm}"
    {
      echo "algorithm=${algorithm} np=4 input_bytes=${input_bytes} total_bytes=$((4 * input_bytes)) interference_bytes=${interference_bytes} iters=${iters} warmup=${warmup} static_alpha=${static_alpha} tcp_initial_alpha=${tcp_initial} pcie_direction=both ranges=${ranges} input_mode=reuse_once validation=final_only"
      timeout --signal=TERM --kill-after=10s 3600s \
        bash "${here}/../mpi_primitives/run.sh" --reuse-input \
          --pcie-size="${interference_bytes}" --pcie-ranges="${ranges}" --pcie-direction=both
    } >"${pair}/${algorithm}.log" 2>&1
  done
  python3 "${here}/process_results.py" "${pair}" --adaptive tcp >"${pair}/summary_stdout.json"
  python3 "${here}/plot_bandwidth.py" "${pair}" --adaptive tcp
  python3 "${here}/plot_variance_envelope.py" "${pair}" --adaptive tcp
  echo "completed=repeat_${rep} at $(date -u +%FT%TZ)" | tee -a "${root}/progress.txt"
  rep=$((rep + 1))
done
python3 "${here}/summarize_paired_matrix.py" "${root}"
echo "Done: ${root}"
