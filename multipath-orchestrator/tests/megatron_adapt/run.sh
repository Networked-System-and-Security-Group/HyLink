#!/usr/bin/env bash
set -euo pipefail

AMPCCL_ROOT="${ADAPTIVE_CCL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AMPCCL_LIB="${AMPCCL_LIB:-${AMPCCL_ROOT}/build/libampccl_hccl.so}"
NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
LOG_FILE="${LOG_FILE:-${AMPCCL_ROOT}/tests/megatron_adapt/torch_npu_hccl.log}"

"$(dirname "${BASH_SOURCE[0]}")/test_hccl_symbols.sh"
"$(dirname "${BASH_SOURCE[0]}")/test_typed_dtype.sh"

if [[ "${RUN_TORCH_NPU_TEST:-0}" != "1" ]]; then
    echo "torch_npu runtime test not requested; set RUN_TORCH_NPU_TEST=1 to run it"
    exit 0
fi

export AMPCCL_ENABLE="${AMPCCL_ENABLE:-1}"
export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-0}"
export AMPCCL_ALGO="${AMPCCL_ALGO:-static}"
export AMPCCL_STATIC_ALPHA="${AMPCCL_STATIC_ALPHA:-0.5}"
export AMPCCL_INTRA_NODE_ONLY="${AMPCCL_INTRA_NODE_ONLY:-1}"
export AMPCCL_NODE_SCOPE_WAIT_MS="${AMPCCL_NODE_SCOPE_WAIT_MS:-5000}"
export AMPCCL_MIN_MSG_SIZE="${AMPCCL_MIN_MSG_SIZE:-1}"
export AMPCCL_SIZE_BUCKET_BASE="${AMPCCL_SIZE_BUCKET_BASE:-1}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-info}"
export AMPCCL_LOG_COLOR=0
export LD_PRELOAD="${AMPCCL_LIB}"

mkdir -p "$(dirname "${LOG_FILE}")"
torchrun \
    --nproc_per_node="${NPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29591}" \
    "$(dirname "${BASH_SOURCE[0]}")/test_hccl_domain_registration.py" \
    2>&1 | tee "${LOG_FILE}"

for marker in \
    "Comm domain created" \
    "AMP-CCL node scope" \
    "Before AllGather" \
    "Before ReduceScatter" \
    "Before AllReduce"; do
    if ! grep -q "${marker}" "${LOG_FILE}"; then
        echo "missing runtime marker: ${marker}" >&2
        exit 1
    fi
done

if ! grep -Eq 'HCCL subcomm registered|HCCL domain registered: api=HcclCommInitRootInfoConfig' "${LOG_FILE}"; then
    echo "missing child/group HCCL domain registration marker" >&2
    exit 1
fi

if [[ "${AMPCCL_ENABLE_PCIE}" == "1" ]] && grep -q 'public PCCL API supports only 4-byte elements' "${LOG_FILE}"; then
    echo "stale untyped PCCL dtype guard was hit" >&2
    exit 1
fi

echo "torch_npu HCCL domain registration: PASS"
