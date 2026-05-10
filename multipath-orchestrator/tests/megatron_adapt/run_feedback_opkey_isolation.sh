#!/usr/bin/env bash
set -euo pipefail

ampccl_root="${ADAPTIVE_CCL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ampccl_lib="${AMPCCL_LIB:-${ampccl_root}/build/libampccl_hccl.so}"
log_file="${LOG_FILE:-/tmp/ampccl_feedback_opkey_isolation.log}"

if [[ ! -r "${ampccl_lib}" ]]; then
    echo "missing AMP-CCL library: ${ampccl_lib}" >&2
    exit 1
fi

export AMPCCL_ENABLE=1
export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-1}"
export AMPCCL_FEEDBACK=1
export AMPCCL_SHM_PARAMS=1
export AMPCCL_ALGO="${AMPCCL_ALGO:-split_ratio}"
export AMPCCL_MIN_MSG_SIZE="${AMPCCL_MIN_MSG_SIZE:-1}"
export AMPCCL_SIZE_BUCKET_BASE="${AMPCCL_SIZE_BUCKET_BASE:-1}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-info}"
export AMPCCL_LOG_COLOR=0
export AMPCCL_NODE_SCOPE_WAIT_MS="${AMPCCL_NODE_SCOPE_WAIT_MS:-5000}"
export PCCL_HOST_BUFFER_SIZE="${PCCL_HOST_BUFFER_SIZE:-8G}"
export LD_PRELOAD="${ampccl_lib}"

mkdir -p "$(dirname "${log_file}")"
torchrun \
    --nproc_per_node="${NPUS_PER_NODE:-8}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-16666}" \
    "${ampccl_root}/tests/megatron_adapt/test_feedback_opkey_isolation.py" \
    2>&1 | tee "${log_file}"

if grep -Eq 'native collective call failed|fast path failed|Unexpected result nan|Traceback' "${log_file}"; then
    echo "feedback OpKey isolation runtime test reported a failure" >&2
    exit 1
fi

python3 "${ampccl_root}/tests/megatron_adapt/check_feedback_plans.py" \
    "${log_file}" --require-independent-initial-alpha

grep -q 'ReduceScatter PASS' "${log_file}"
grep -q 'AllGather PASS' "${log_file}"
echo "feedback OpKey isolation runtime test: PASS"
