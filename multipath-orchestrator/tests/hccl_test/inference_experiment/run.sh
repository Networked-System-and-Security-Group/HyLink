#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${experiment_dir}/../../.." && pwd)"
NP=2
INPUT_BYTES=$((512 * 1024 * 1024))
TOTAL_BYTES=$((INPUT_BYTES * NP))
INTERFERENCE_BYTES=$((256 * 1024 * 1024))
ITERS=1001
WARMUP=10
RANGES="100-200,300-600,700-900"
STATIC_ALPHA=0.39
TCP_INITIAL_ALPHA=0.075
TCP_HOLD_RATE_LOW="${TCP_HOLD_RATE_LOW:-0.95}"
TCP_HOLD_RATE_HIGH="${TCP_HOLD_RATE_HIGH:-1.10}"
NP="${EXPERIMENT_NP:-${NP}}"
INPUT_BYTES="${EXPERIMENT_INPUT_BYTES:-${INPUT_BYTES}}"
TOTAL_BYTES=$((INPUT_BYTES * NP))
PCIE_DIRECTION="${PCIE_DIRECTION:-d2h}"
case "${PCIE_DIRECTION}" in d2h|h2d|both) ;; *) echo "Invalid PCIE_DIRECTION" >&2; exit 2 ;; esac
config="np=${NP} input_bytes=${INPUT_BYTES} total_bytes=${TOTAL_BYTES} interference_bytes=${INTERFERENCE_BYTES} iters=${ITERS} warmup=${WARMUP} ranges=${RANGES} static_alpha=${STATIC_ALPHA} tcp_initial_alpha=${TCP_INITIAL_ALPHA} hold_rate_low=${TCP_HOLD_RATE_LOW} hold_rate_high=${TCP_HOLD_RATE_HIGH} input_mode=reuse_once validation=final_only"
if [[ "${1:-}" == --dry-run ]]; then
    echo "${config}"
    exit 0
fi
adaptive="${ADAPTIVE_ALGORITHM:-tcp}"
case "${adaptive}" in tcp|split_ratio) ;; *) echo "Invalid ADAPTIVE_ALGORITHM" >&2; exit 2 ;; esac
algorithms=("${adaptive}" static)
tcp_source=""
if [[ $# -eq 2 && "$1" == --static-only ]]; then
    tcp_source="$(cd "$2" && pwd)"
    [[ -f "${tcp_source}/tcp.log" ]] || { echo "Missing TCP log" >&2; exit 1; }
    algorithms=(static)
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash run.sh [--dry-run | --static-only existing_results_dir]" >&2
    exit 2
fi
run_dir="${experiment_dir}/results/$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}" "${experiment_dir}/bin"
if [[ -n "${tcp_source}" ]]; then
    cp "${tcp_source}/tcp.log" "${run_dir}/tcp.log"
    cp "${tcp_source}/metadata.txt" "${run_dir}/tcp_source_metadata.txt"
fi
while IFS='=' read -r variable ignored; do
    case "${variable}" in AMPCCL_*|PCCL_*|CCL_*) unset "${variable}" ;; esac
done < <(env)
export NP OP=ag SIZE=$((INPUT_BYTES / 4)) ITERS WARMUP
devices="$(seq -s, 0 $((NP - 1)))"
export ACL_VISIBLE_DEVICES="${devices}" ASCEND_RT_VISIBLE_DEVICES="${devices}"
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_STATIC_ALPHA="${STATIC_ALPHA}" AMPCCL_TCP_INITIAL_ALPHA="${TCP_INITIAL_ALPHA}"
export AMPCCL_TCP_HOLD_RATE_LOW="${TCP_HOLD_RATE_LOW}" AMPCCL_TCP_HOLD_RATE_HIGH="${TCP_HOLD_RATE_HIGH}"
export AMPCCL_SPLIT_RATIO_INITIAL_ALPHA="${TCP_INITIAL_ALPHA}"
export AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir"
export AMP_CCL_PATH="${repo}/build/libampccl_hccl.so"
export BIN="${experiment_dir}/bin/hccl_mpi_primitives"
unset LD_PRELOAD BENCHMARK_DEBUG_TIMING PCCL_FORCE_DMA_LANES
unset CCL_MSPROF_MARKERS PCCL_MSPROF_HOST_TIMING PCCL_MSPROF_CPU_SAMPLE_PERIOD PCCL_DEBUG_TIMING
unset AMPCCL_FORCE_DISABLE AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK AMPCCL_SYNC_TIMER_EVENTS_IN_CALLBACK
bash "${experiment_dir}/../mpi_primitives/build.sh" \
    "${experiment_dir}/../mpi_primitives/hccl_mpi_primitives.cc" "${BIN}" >"${run_dir}/build.log" 2>&1
{
    echo "${config}"
    echo "reused_tcp_source=${tcp_source}"
    echo "range_semantics=left_closed_right_open interference_mode=continuous_sync pcie_direction=${PCIE_DIRECTION}"
    echo "bandwidth=aggregate_output_${TOTAL_BYTES}_bytes/event_seconds"
    git -C "${repo}" log -1 --oneline
    git -C "${repo}/../pcieccl" log -1 --oneline
    sha256sum "${BIN}" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so"
    env | rg '^(AMPCCL_|PCCL_|ASCEND_|ACL_|NP=|OP=|SIZE=|ITERS=|WARMUP=|BIN=|LD_LIBRARY_PATH=)' | sort
} >"${run_dir}/metadata.txt"
echo "Results: ${run_dir}"
for algorithm in "${algorithms[@]}"; do
    if [[ "${algorithm}" == static && "${STATIC_FROM_TCP:-0}" == 1 ]]; then
        STATIC_ALPHA="$(python3 "${experiment_dir}/select_static_alpha.py" "${run_dir}/tcp.log" "${run_dir}/alpha_selection.json")"
        export AMPCCL_STATIC_ALPHA="${STATIC_ALPHA}"
        config="${config/static_alpha=0.39/static_alpha=${STATIC_ALPHA}}"
        echo "Selected static_alpha=${STATIC_ALPHA} from this run's interference-free TCP samples" | tee -a "${run_dir}/metadata.txt"
    fi
    export AMPCCL_ALGO="${algorithm}"
    echo "Starting ${algorithm} ($(date -u +%FT%TZ))"
    {
        echo "algorithm=${algorithm} ${config} pcie_direction=${PCIE_DIRECTION}"
        timeout --signal=TERM --kill-after=10s 3600s \
            bash "${experiment_dir}/../mpi_primitives/run.sh" \
                --pcie-size="${INTERFERENCE_BYTES}" --pcie-ranges="${RANGES}" --reuse-input --pcie-direction="${PCIE_DIRECTION}"
    } >"${run_dir}/${algorithm}.log" 2>&1
    echo "Finished ${algorithm} ($(date -u +%FT%TZ))"
done
python3 "${experiment_dir}/process_results.py" "${run_dir}" --adaptive "${adaptive}"
python3 "${experiment_dir}/plot_bandwidth.py" "${run_dir}" --adaptive "${adaptive}"
python3 "${experiment_dir}/plot_variance_envelope.py" "${run_dir}" --adaptive "${adaptive}"
echo "Done: ${run_dir}"
