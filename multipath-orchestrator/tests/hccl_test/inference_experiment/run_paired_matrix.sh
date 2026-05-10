#!/usr/bin/env bash
# Paired TCP/static AllGather sweep. All generated data is kept under /logs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
log_root="${LOG_ROOT:-/logs/ag_tcp_static_paired_$(date -u +%Y%m%d_%H%M%S)}"
input_bytes="${INPUT_BYTES:-536870912}"
iters="${ITERS:-1001}"
warmup="${WARMUP:-10}"
tcp_low="${TCP_HOLD_RATE_LOW:-0.95}"
tcp_high="${TCP_HOLD_RATE_HIGH:-1.10}"
tcp_initial="${TCP_INITIAL_ALPHA:-0.075}"
ranges="100-200,300-600,700-900"
bin="${here}/bin/hccl_mpi_primitives"
start_np="${START_NP:-2}"
skip_calibration="${SKIP_CALIBRATION:-0}"

mkdir -p "${log_root}/calibration" "${log_root}/pairs"
test -x "${bin}" || { echo "Missing benchmark binary: ${bin}" >&2; exit 1; }
test -f "${repo}/build/libampccl_hccl.so" || { echo "Missing Adaptive-CCL library" >&2; exit 1; }
test -f "${repo}/../pcieccl/build/libpccl.so" || { echo "Missing PCIeCCL library" >&2; exit 1; }

while IFS='=' read -r name _; do
  case "${name}" in AMPCCL_*|PCCL_*|CCL_*) unset "${name}" ;; esac
done < <(env)
unset LD_PRELOAD BENCHMARK_DEBUG_TIMING PCCL_FORCE_DMA_LANES

export OP=ag ITERS="${iters}" WARMUP="${warmup}"
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_TCP_INITIAL_ALPHA="${tcp_initial}"
export AMPCCL_TCP_HOLD_RATE_LOW="${tcp_low}" AMPCCL_TCP_HOLD_RATE_HIGH="${tcp_high}"
export AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir"
export AMP_CCL_PATH="${repo}/build/libampccl_hccl.so"
export BIN="${bin}"
unset CCL_MSPROF_MARKERS PCCL_MSPROF_HOST_TIMING PCCL_MSPROF_CPU_SAMPLE_PERIOD PCCL_DEBUG_TIMING
unset AMPCCL_FORCE_DISABLE AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK AMPCCL_SYNC_TIMER_EVENTS_IN_CALLBACK

run_one() {
  local np="$1" algorithm="$2" alpha="$3" direction="$4" bytes="$5" out="$6"
  local args=(--reuse-input)
  export NP="${np}" SIZE=$((input_bytes / 4))
  export ACL_VISIBLE_DEVICES="$(seq -s, 0 $((np - 1)))"
  export ASCEND_RT_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES}"
  export AMPCCL_ALGO="${algorithm}" AMPCCL_STATIC_ALPHA="${alpha}"
  mkdir -p "${out}"
  if (( bytes > 0 )); then
    args+=(--pcie-size="${bytes}" --pcie-ranges="${ranges}" --pcie-direction="${direction}")
  fi
  {
    echo "algorithm=${algorithm} np=${np} input_bytes=${input_bytes} total_bytes=$((input_bytes * np)) interference_bytes=${bytes} iters=${iters} warmup=${warmup} static_alpha=${alpha} tcp_initial_alpha=${tcp_initial} tcp_hold_rate_low=${tcp_low} tcp_hold_rate_high=${tcp_high} pcie_direction=${direction} ranges=${ranges} input_mode=reuse_once validation=final_only"
    timeout --signal=TERM --kill-after=10s 3600s \
      bash "${here}/../mpi_primitives/run.sh" "${args[@]}"
  } >"${out}/${algorithm}.log" 2>&1
}

echo "log_root=${log_root}" | tee "${log_root}/RUNNING"
{
  date -u +%FT%TZ
  echo "np=2,4,8 input_bytes=${input_bytes} iters=${iters} warmup=${warmup} ranges=${ranges}"
  echo "directions=d2h,h2d,both sizes_per_direction_bytes=67108864,134217728,268435456"
  echo "tcp_initial_alpha=${tcp_initial} tcp_hold_rate_low=${tcp_low} tcp_hold_rate_high=${tcp_high}"
  git -C "${repo}" log -1 --oneline
  git -C "${repo}/../pcieccl" log -1 --oneline
  sha256sum "${bin}" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so"
  env | rg '^(AMPCCL_|PCCL_|ASCEND_|ACL_|OP=|ITERS=|WARMUP=|BIN=)' | sort
} >"${log_root}/metadata.txt"

# Calibrate one static fraction per rank count from the dominant split ratio in
# an interference-free TCP run; this is a frequency/mode rule, not a max-BW pick.
if [[ "${skip_calibration}" != 1 ]]; then
  for np in 2 4 8; do
    out="${log_root}/calibration/np${np}"
    run_one "${np}" tcp "${tcp_initial}" d2h 0 "${out}"
    python3 "${here}/select_static_alpha_mode.py" "${out}/tcp.log" "${out}/alpha_selection.json" \
      >"${out}/static_alpha.txt"
    alpha="$(cat "${out}/static_alpha.txt")"
    echo "np=${np} static_alpha=${alpha}" | tee -a "${log_root}/static_alpha_by_np.txt"
  done
else
  test -s "${log_root}/static_alpha_by_np.txt" || { echo "Missing calibration file under ${log_root}" >&2; exit 1; }
fi

# Broad screen: one complete adjacent TCP/static pair per condition. Device
# occupancy is exclusive; pair order alternates to reduce order/thermal bias.
pair_index=0
for np in 2 4 8; do
  (( np < start_np )) && continue
  alpha="$(awk -v n="${np}" '$1=="np="n {print $2}' "${log_root}/static_alpha_by_np.txt" | sed 's/static_alpha=//')"
  for direction in d2h h2d both; do
    for mib in 64 128 256; do
      pair_index=$((pair_index + 1))
      bytes=$((mib * 1024 * 1024))
      key="np${np}_${direction}_${mib}MiB"
      out="${log_root}/pairs/${key}"
      mkdir -p "${out}"
      echo "pair=${key} np=${np} direction=${direction} interference_bytes_per_direction=${bytes} static_alpha=${alpha} order=$((pair_index % 2))" >"${out}/pair_metadata.txt"
      if (( pair_index % 2 )); then
        run_one "${np}" tcp "${alpha}" "${direction}" "${bytes}" "${out}"
        run_one "${np}" static "${alpha}" "${direction}" "${bytes}" "${out}"
      else
        run_one "${np}" static "${alpha}" "${direction}" "${bytes}" "${out}"
        run_one "${np}" tcp "${alpha}" "${direction}" "${bytes}" "${out}"
      fi
      python3 "${here}/process_results.py" "${out}" --adaptive tcp >"${out}/summary_stdout.json"
      python3 "${here}/plot_bandwidth.py" "${out}" --adaptive tcp
      python3 "${here}/plot_variance_envelope.py" "${out}" --adaptive tcp
      echo "finished=${key} at $(date -u +%FT%TZ)" | tee -a "${log_root}/progress.txt"
    done
  done
done

python3 "${here}/summarize_paired_matrix.py" "${log_root}"
rm -f "${log_root}/RUNNING"
echo "Done: ${log_root}"
