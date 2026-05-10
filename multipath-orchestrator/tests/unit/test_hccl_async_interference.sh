#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ascend_root="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
test_dir="$(mktemp -d /tmp/ampccl-async-interference.XXXXXX)"
mpicxx -std=c++11 -O2 -I"${ascend_root}/include" \
    "${repo}/tests/unit/test_hccl_async_interference.cc" \
    -L"${ascend_root}/lib64" -Wl,-rpath,"${ascend_root}/lib64" \
    -lhccl -lascendcl -pthread \
    -Wl,--wrap=aclrtSetDevice,--wrap=aclrtMemcpy,--wrap=aclrtMemcpyAsync,--wrap=aclrtSynchronizeStream \
    -o "${test_dir}/test_async_interference"
timeout --signal=TERM --kill-after=2s 10s "${test_dir}/test_async_interference"
