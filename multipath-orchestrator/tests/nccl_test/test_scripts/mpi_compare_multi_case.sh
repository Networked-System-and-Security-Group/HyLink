#!/bin/bash
# NCCL baseline vs Adaptive-CCL full microbenchmark sweep — MULTI-PROCESS (MPI) mode.
#
# Counterpart of compare_multi_case.sh (single-process multi-thread). Each case is
# one mpirun job: NP ranks, one GPU per rank, using tests/nccl_test/mpi_primitives.
#
# Matrix:
#   collectives: allgather, allreduce, reducescatter by default
#   ranks (NP):  2, 4, 8 by default
#   sizes:       1KB -> 1GB, x4
#   modes:       baseline, adaptive
#
# Notes:
#   - All tests use float, so the CLI element count is size_bytes / 4.
#   - Log markers (=== header / --- case ---) are identical to compare_multi_case.sh
#     so process_compare_logs.py parses both log trees with the same code.
#   - mpirun here is MPICH/Hydra: env is propagated with -genvall (no OpenMPI -x).
#   - Multi-process mode REQUIRES AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID to be unset/0,
#     and relies on AMPCCL_SHM_PARAMS=1 for cross-rank parameter aggregation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
LOGS_ROOT="${LOGS_ROOT:-${SCRIPT_DIR}/logs/mpi_compare}"
TS="$(date +%Y-%m-%d_%H%M)"

BIN="${BIN:-${ADAPTIVE_CCL_ROOT}/tests/nccl_test/mpi_primitives/nccl_mpi_primitives}"

COLLECTIVES=(${COLLECTIVES:-allgather allreduce reducescatter})
CARD_COUNTS=(${CARD_COUNTS:-2 4 8})
SIZES_BYTES=(${SIZES_BYTES:-1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456 1073741824})
ITERS="${ITERS:-30}"
WARMUP="${WARMUP:-3}"
SLEEP_BETWEEN_CASES="${SLEEP_BETWEEN_CASES:-3}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_ADAPTIVE="${RUN_ADAPTIVE:-1}"

RUN_FAILURES=0
FAILED_CASES=()

export NCCL_BUFFSIZE=50331648 # 48MB, same as single-process sweep
export AMPCCL_MAX_PCIE_RECVBUF_BYTES=$((4 * 1024 * 1024 * 1024))

usage() {
    cat <<USAGE
Usage: $0 [--quick] [--baseline-only] [--adaptive-only]

Environment overrides:
  ADAPTIVE_CCL_ROOT       repo root, default autodetected from script location
  NCCL_HOME               NCCL build dir, default /root/sxb/nccl/build
  BIN                     benchmark binary, default tests/nccl_test/mpi_primitives/nccl_mpi_primitives
  LOGS_ROOT               output root, default tests/nccl_test/test_scripts/logs/mpi_compare
  COLLECTIVES             space-separated list, default "allgather allreduce reducescatter"
  CARD_COUNTS             space-separated list (= MPI ranks), default "2 4 8"
  SIZES_BYTES             space-separated byte sizes, default 1KB..1GB x4
  ITERS                   measured iterations per case, default 30
  WARMUP                  warmup iterations per case, default 3
  RUN_BASELINE            1/0, default 1
  RUN_ADAPTIVE            1/0, default 1
  CUDA_VISIBLE_DEVICES    optional GPU visibility mask

Examples:
  bash mpi_compare_multi_case.sh --quick
  CARD_COUNTS="8" COLLECTIVES="allreduce" bash mpi_compare_multi_case.sh
  RUN_BASELINE=0 ITERS=30 bash mpi_compare_multi_case.sh
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
    export AMPCCL_MIN_MSG_SIZE="${AMPCCL_MIN_MSG_SIZE:-$((1024*1024))}" # 1MB fallback,与单进程 sweep 一致
    export AMPCCL_SHM_PARAMS="${AMPCCL_SHM_PARAMS:-1}"                  # 多进程依赖 shm 聚合
    export AMPCCL_INTRA_NODE_ONLY="${AMPCCL_INTRA_NODE_ONLY:-1}"
    export AMPCCL_NODE_SCOPE_WAIT_MS="${AMPCCL_NODE_SCOPE_WAIT_MS:-1000}"
    export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK="${AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK:-1}"
    export AMPCCL_AVG_BW_LOG="${AMPCCL_AVG_BW_LOG:-0}"
    export AMPCCL_AVG_BW_WARMUP="${AMPCCL_AVG_BW_WARMUP:-2}"
    export PCCL_FORCE_NVLINK="${PCCL_FORCE_NVLINK:-1}"   # 必开,确保 PCIe 数据不走 NVLink
    export PCCL_WARMUP_KERNEL="${PCCL_WARMUP_KERNEL:-0}"
    # 多进程模式必须用真实的 PCCL unique id 交换,严禁置 1
    unset AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID

    local pcieccl_root
    pcieccl_root="${PCIECCL_ROOT:-$(cd "${ADAPTIVE_CCL_ROOT}/.." 2>/dev/null && pwd)/pcieccl}"
    if [ -z "${PCCL_IR_PATH:-}" ] && [ -d "${pcieccl_root}/ir" ]; then
        export PCCL_IR_PATH="${pcieccl_root}/ir"
    fi
}

AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_nccl.so"
[ -x "${BIN}" ] || die "${BIN} not found. Build first: cd tests/nccl_test/mpi_primitives && ./build.sh"
if [ "$RUN_ADAPTIVE" = "1" ]; then
    [ -f "${AMP_CCL_PATH}" ] || die "${AMP_CCL_PATH} not found. Build Adaptive-CCL NCCL hook first."
    setup_adaptive_env
fi

run_one() {
    local mode=$1 coll=$2 cards=$3 size_bytes=$4
    local elems
    elems="$(elem_count "$size_bytes")"

    echo "--- mode=${mode} collective=${coll} cards=${cards} size_bytes=${size_bytes} elems_arg=${elems} iters=${ITERS} ---"

    if [ "$mode" = "baseline" ]; then
        run_line_buffered env -u LD_PRELOAD AMPCCL_FORCE_DISABLE=1 \
            mpirun -np "$cards" -genvall \
            "${BIN}" --op="${coll}" --count="${elems}" --iters="${ITERS}" --warmup="${WARMUP}"
    else
        LD_PRELOAD="${AMP_CCL_PATH}" run_line_buffered \
            mpirun -np "$cards" -genvall \
            "${BIN}" --op="${coll}" --count="${elems}" --iters="${ITERS}" --warmup="${WARMUP}"
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
    echo ">>> [${mode}] ${coll} ${cards}gpu (mpi) -> ${log_file}"
    {
        echo "=== mode=${mode} backend=NCCL collective=${coll} cards=${cards} timestamp=${TS} ==="
        echo "=== sizes_bytes=${SIZES_BYTES[*]} iters=${ITERS} warmup=${WARMUP} launch=mpi ==="
        echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
        for sz in "${SIZES_BYTES[@]}"; do
            run_case_continue_on_failure "$mode" "$coll" "$cards" "$sz"
        done
    } > >(tee "${log_file}") 2>&1
}

mkdir -p "${LOGS_ROOT}"

echo "============================================"
echo "NCCL baseline vs Adaptive-CCL comparison (MULTI-PROCESS / MPI)"
echo "Start: $(date)"
echo "Repo root: ${ADAPTIVE_CCL_ROOT}"
echo "Binary: ${BIN}"
echo "Collectives: ${COLLECTIVES[*]}"
echo "Card counts (NP): ${CARD_COUNTS[*]}"
echo "Sizes bytes: ${SIZES_BYTES[*]}"
echo "Iters per case: ${ITERS} (warmup ${WARMUP})"
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
