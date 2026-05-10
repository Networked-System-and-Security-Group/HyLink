#!/usr/bin/env bash
set -euo pipefail

AMPCCL_ROOT="${AMPCCL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TEST_BIN="/tmp/ampccl_test_per_op_controller"

g++ -std=c++17 -O0 -g \
    -DAMPCCL_USE_ACL_TIMER \
    -I"${AMPCCL_ROOT}/libampccl" \
    -I"${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/include" \
    "${AMPCCL_ROOT}/tests/unit/test_per_op_controller.cc" \
    "${AMPCCL_ROOT}/libampccl/core/shm_store.cc" \
    -pthread \
    -o "${TEST_BIN}"

AMPCCL_ALGO=split_ratio AMPCCL_ENABLE_PCIE=1 "${TEST_BIN}"
