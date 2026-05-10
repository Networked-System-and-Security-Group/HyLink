#!/usr/bin/env bash
set -euo pipefail

AMPCCL_ROOT="${ADAPTIVE_CCL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PCCL_ROOT="${PCIECCL_ROOT:-/workspace/pcieccl}"
PCCL_LIB="${PCCL_LIB:-${PCCL_ROOT}/build/libpccl.so}"
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "${TEST_DIR}"' EXIT

if [[ ! -r "${PCCL_LIB}" ]]; then
    echo "missing PCCL library: ${PCCL_LIB}" >&2
    exit 2
fi

required_symbols=(
    pcclAllGatherTyped
    pcclAllGatherStridedTyped
    pcclReduceScatterTyped
    pcclReduceScatterStridedTyped
    pcclAllReduceTyped
    pcclAllToAllTyped
)
symbols="$(nm -D --defined-only "${PCCL_LIB}")"
for symbol in "${required_symbols[@]}"; do
    if ! grep -Eq "[[:space:]]${symbol}$" <<<"${symbols}"; then
        echo "missing required PCCL typed export: ${symbol}" >&2
        exit 1
    fi
done

g++ -std=c++17 -Wall -Wextra \
    -DAMPCCL_BUILD_HCCL=1 \
    -I"${ASCEND_HOME}/include" \
    -I"${AMPCCL_ROOT}/libampccl" \
    -I"${PCCL_ROOT}/include" \
    "${AMPCCL_ROOT}/tests/megatron_adapt/test_hccl_pccl_dtype_mapping.cc" \
    -o "${TEST_DIR}/test_hccl_pccl_dtype_mapping"
"${TEST_DIR}/test_hccl_pccl_dtype_mapping"

echo "AMP-CCL HCCL/PCCL typed dtype mapping: PASS"
