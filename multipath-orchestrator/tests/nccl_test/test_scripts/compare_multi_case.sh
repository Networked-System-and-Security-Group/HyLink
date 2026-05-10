#!/bin/bash
# NCCL baseline vs Adaptive-CCL full microbenchmark sweep.
#
# Matrix:
#   collectives: allgather, allreduce, alltoall, reducescatter
#   devices:     2, 4, 8 by default
#   sizes:       1KB -> 1GB, x4
#   modes:       baseline, adaptive
#
# Notes:
#   - All tests use float, so the CLI element count is size_bytes / 4.
#   - For alltoall, the benchmark interprets the count as per-peer elements.
#     The logged size is therefore per-peer payload size.
#   - This script calls benchmark binaries directly. It intentionally does not
#     use per-collective run.sh because some run.sh files enable profiling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
LOGS_ROOT="${LOGS_ROOT:-${SCRIPT_DIR}/logs/compare}"
TS="$(date +%Y-%m-%d_%H%M)"

COLLECTIVES=(${COLLECTIVES:-allgather allreduce alltoall reducescatter})
CARD_COUNTS=(${CARD_COUNTS:-2 4 8})
SIZES_BYTES=(${SIZES_BYTES:-1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456 1073741824})
ITERS="${ITERS:-30}"
SLEEP_BETWEEN_CASES="${SLEEP_BETWEEN_CASES:-3}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_ADAPTIVE="${RUN_ADAPTIVE:-1}"

RUN_FAILURES=0
FAILED_CASES=()

export NCCL_BUFFSIZE=50331648 # 48MB优化
export AMPCCL_MAX_PCIE_RECVBUF_BYTES=$((4 * 1024 * 1024 * 1024))


usage() {
    cat <<USAGE
Usage: $0 [--quick] [--baseline-only] [--adaptive-only]

Environment overrides:
  ADAPTIVE_CCL_ROOT       repo root, default /workspace/Adaptive-CCL
  LOGS_ROOT               output root, default tests/nccl_test/test_scripts/logs/compare
  COLLECTIVES             space-separated list, default "allgather allreduce alltoall reducescatter"
  CARD_COUNTS             space-separated list, default "2 4 8"
  SIZES_BYTES             space-separated byte sizes, default 1KB..1GB x4
  ITERS                   measured iterations per case, default 20
  RUN_BASELINE            1/0, default 1
  RUN_ADAPTIVE            1/0, default 1
  CUDA_VISIBLE_DEVICES    optional GPU visibility mask

Examples:
  bash compare_multi_case.sh --quick
  CARD_COUNTS="8" COLLECTIVES="allreduce" bash compare_multi_case.sh
  RUN_BASELINE=0 ITERS=30 bash compare_multi_case.sh
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)
            CARD_COUNTS=(${CARD_COUNTS_QUICK:-2})
            SIZES_BYTES=(${SIZES_BYTES_QUICK:-1024 65536 1048576 16777216})
            ITERS="${ITERS_QUICK:-5}"
            ;;
        --baseline-only)
            RUN_BASELINE=1
            RUN_ADAPTIVE=0
            ;;
        --adaptive-only)
            RUN_BASELINE=0
            RUN_ADAPTIVE=1
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
    shift
done

die() {
    echo "ERROR: $*" >&2
    exit 1
}

elem_count() {
    echo $(( $1 / 4 ))
}

bin_name() {
    case "$1" in
        allgather)     echo "nccl_allgather_param" ;;
        allreduce)     echo "nccl_allreduce_param" ;;
        alltoall)      echo "nccl_alltoall_param" ;;
        reducescatter) echo "nccl_reducescatter_param" ;;
        *) die "unknown collective: $1" ;;
    esac
}

run_line_buffered() {
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "$@"
    else
        "$@"
    fi
}

record_case_failure() {
    local mode=$1 coll=$2 cards=$3 size_bytes=$4 rc=$5
    RUN_FAILURES=$((RUN_FAILURES + 1))
    FAILED_CASES+=("${mode} ${coll} ${cards}gpu size=${size_bytes} rc=${rc}")
}

setup_adaptive_env() {
    export AMPCCL_ENABLE="${AMPCCL_ENABLE:-1}"
    export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-1}"
    export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-off}"
    export AMPCCL_ALGO="${AMPCCL_ALGO:-split_ratio}"
    export AMPCCL_MIN_CHUNK_SIZE="${AMPCCL_MIN_CHUNK_SIZE:-1024}"
    export AMPCCL_MIN_MSG_SIZE="${AMPCCL_MIN_MSG_SIZE:-$((1024*1024))}" #以1MB作为最小的fallback
    export AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID="${AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID:-1}" # 开1表示单进程模式，0表示多进程模式需要mpi
    export AMPCCL_AVG_BW_LOG="${AMPCCL_AVG_BW_LOG:-0}" # 这里暂且不输出带宽了，需要等调整好双路带宽协同问题
    export AMPCCL_AVG_BW_WARMUP="${AMPCCL_AVG_BW_WARMUP:-2}"
    export PCCL_FORCE_NVLINK="${PCCL_FORCE_NVLINK:-1}" # 这个必开，确保PCIe数据不走NVLink
    export PCCL_WARMUP_KERNEL="${PCCL_WARMUP_KERNEL:-0}" # 当前 V100/NCCL smoke 下 PCCL warmup kernel 会失败，默认关闭

    local pcieccl_root
    pcieccl_root="${PCIECCL_ROOT:-$(cd "${ADAPTIVE_CCL_ROOT}/.." 2>/dev/null && pwd)/pcieccl}"
    if [ -z "${PCCL_IR_PATH:-}" ] && [ -d "${pcieccl_root}/ir" ]; then
        export PCCL_IR_PATH="${pcieccl_root}/ir"
    fi
}

AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_nccl.so"
if [ "$RUN_ADAPTIVE" = "1" ]; then
    [ -f "${AMP_CCL_PATH}" ] || die "${AMP_CCL_PATH} not found. Build Adaptive-CCL NCCL hook first."
    setup_adaptive_env
fi

run_one() {
    local mode=$1 coll=$2 cards=$3 size_bytes=$4
    local elems bin bin_path
    elems="$(elem_count "$size_bytes")"
    bin="$(bin_name "$coll")"
    bin_path="${ADAPTIVE_CCL_ROOT}/tests/nccl_test/${coll}/${bin}"

    [ -x "${bin_path}" ] || die "${bin_path} not found. Build ${coll} first."

    echo "--- mode=${mode} collective=${coll} cards=${cards} size_bytes=${size_bytes} elems_arg=${elems} iters=${ITERS} ---"

    if [ "$mode" = "baseline" ]; then
        run_line_buffered env -u LD_PRELOAD AMPCCL_FORCE_DISABLE=1 "${bin_path}" "$cards" "$elems" "$ITERS"
    else
        LD_PRELOAD="${AMP_CCL_PATH}" run_line_buffered "${bin_path}" "$cards" "$elems" "$ITERS"
    fi
}

run_case_continue_on_failure() {
    local mode=$1 coll=$2 cards=$3 size_bytes=$4
    local rc
    if run_one "$mode" "$coll" "$cards" "$size_bytes"; then
        echo ">>> CASE PASS: mode=${mode} collective=${coll} cards=${cards} size=${size_bytes}"
    else
        rc=$?
        echo ">>> CASE FAILED: mode=${mode} collective=${coll} cards=${cards} size=${size_bytes} rc=${rc}"
        record_case_failure "$mode" "$coll" "$cards" "$size_bytes" "$rc"
    fi
    sleep "${SLEEP_BETWEEN_CASES}"
}

run_mode_block() {
    local mode=$1 coll=$2 cards=$3 log_file=$4
    echo ">>> [${mode}] ${coll} ${cards}gpu -> ${log_file}"
    {
        echo "=== mode=${mode} backend=NCCL collective=${coll} cards=${cards} timestamp=${TS} ==="
        echo "=== sizes_bytes=${SIZES_BYTES[*]} iters=${ITERS} ==="
        echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
        for sz in "${SIZES_BYTES[@]}"; do
            run_case_continue_on_failure "$mode" "$coll" "$cards" "$sz"
        done
    } > >(tee "${log_file}") 2>&1
}

mkdir -p "${LOGS_ROOT}"

echo "============================================"
echo "NCCL baseline vs Adaptive-CCL comparison"
echo "Start: $(date)"
echo "Repo root: ${ADAPTIVE_CCL_ROOT}"
echo "Collectives: ${COLLECTIVES[*]}"
echo "Card counts: ${CARD_COUNTS[*]}"
echo "Sizes bytes: ${SIZES_BYTES[*]}"
echo "Iters per case: ${ITERS}"
echo "Run baseline: ${RUN_BASELINE}"
echo "Run adaptive: ${RUN_ADAPTIVE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Log root: ${LOGS_ROOT}"
echo "============================================"
echo ""

for coll in "${COLLECTIVES[@]}"; do
    LOG_DIR="${LOGS_ROOT}/${coll}"
    mkdir -p "${LOG_DIR}"

    for cards in "${CARD_COUNTS[@]}"; do
        if [ "$RUN_BASELINE" = "1" ]; then
            run_mode_block "baseline" "$coll" "$cards" "${LOG_DIR}/${cards}gpu_baseline_${TS}.log"
            echo ""
        fi

        if [ "$RUN_ADAPTIVE" = "1" ]; then
            run_mode_block "adaptive" "$coll" "$cards" "${LOG_DIR}/${cards}gpu_adaptive_${TS}.log"
            echo ""
        fi
    done
done

echo "============================================"
echo "Experiment done: $(date)"
echo "Logs: ${LOGS_ROOT}"
echo "Failed cases: ${RUN_FAILURES}"
if [ "${RUN_FAILURES}" -gt 0 ]; then
    printf '  %s\n' "${FAILED_CASES[@]}"
fi
echo "============================================"

if [ "${RUN_FAILURES}" -gt 0 ]; then
    exit 1
fi


# cd /home/songxb26/HyLink/Adaptive-CCL
#   rm -rf tests/nccl_test/test_scripts/logs/compare tests/nccl_test/test_scripts/csv
#   CUDA_VISIBLE_DEVICES=0,1,2,3 CARD_COUNTS="2 4" bash tests/nccl_test/test_scripts/compare_multi_case.sh