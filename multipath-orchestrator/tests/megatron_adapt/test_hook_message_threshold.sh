#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
compiler="${CXX:-g++}"
binary="${repo_dir}/tests/megatron_adapt/.test_hook_message_threshold"

cleanup() {
    rm -f "${binary}"
}
trap cleanup EXIT

"${compiler}" -std=c++17 -Wall -Wextra -Werror \
    -I"${repo_dir}/libampccl" \
    "${repo_dir}/tests/megatron_adapt/test_hook_message_threshold.cc" \
    -o "${binary}"
"${binary}"

if [[ "$(rg -c 'ShouldInterceptCollective' "${repo_dir}/libampccl/hook/hccl_hook.cc")" -ne 4 ]]; then
    echo "HCCL hook policy coverage is incomplete" >&2
    exit 1
fi
if [[ "$(rg -c 'ShouldInterceptCollective' "${repo_dir}/libampccl/hook/nccl_hook.cc")" -ne 4 ]]; then
    echo "NCCL hook policy coverage is incomplete" >&2
    exit 1
fi

echo "hook message threshold policy: PASS"
