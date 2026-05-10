#!/usr/bin/env bash
set -euo pipefail

ampccl_root="${ADAPTIVE_CCL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ampccl_lib="${AMPCCL_LIB:-${ampccl_root}/build/libampccl_hccl.so}"
log_file="${LOG_FILE:-${ampccl_root}/tests/megatron_adapt/concurrent_subgroups_large.log}"
mode="${MODE:-adaptive}"

export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_ALGO="${AMPCCL_ALGO:-static}"
export AMPCCL_STATIC_ALPHA="${AMPCCL_STATIC_ALPHA:-0.1}"
export AMPCCL_FEEDBACK=0
export AMPCCL_MIN_MSG_SIZE="${AMPCCL_MIN_MSG_SIZE:-268435456}"
export AMPCCL_SIZE_BUCKET_BASE="${AMPCCL_SIZE_BUCKET_BASE:-4194304}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-info}"
export AMPCCL_LOG_COLOR=0
export PCCL_HOST_BUFFER_SIZE="${PCCL_HOST_BUFFER_SIZE:-8G}"

if [[ "${mode}" == "baseline" ]]; then
    export AMPCCL_ENABLE=0
    export AMPCCL_ENABLE_PCIE=0
    unset LD_PRELOAD
elif [[ "${mode}" == "adaptive" ]]; then
    if [[ ! -r "${ampccl_lib}" ]]; then
        echo "missing AMP-CCL library: ${ampccl_lib}" >&2
        exit 1
    fi
    export LD_PRELOAD="${ampccl_lib}"
else
    echo "MODE must be adaptive or baseline, got: ${mode}" >&2
    exit 1
fi

mkdir -p "$(dirname "${log_file}")"
torchrun \
    --nproc_per_node=8 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-16666}" \
    "${ampccl_root}/tests/megatron_adapt/test_concurrent_subgroups_large.py" \
    2>&1 | tee "${log_file}"

test_op="${TEST_OP:-both}"
if [[ "${test_op}" == "both" || "${test_op}" == "reducescatter" ]]; then
    grep -q 'ReduceScatter.*PASS' "${log_file}"
fi
if [[ "${test_op}" == "both" || "${test_op}" == "allgather" ]]; then
    grep -q 'AllGather.*PASS' "${log_file}"
fi
echo "concurrent large subgroup test: PASS (${mode})"
