#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
hook_file="${repo_dir}/libampccl/hook/hccl_hook.cc"
init_file="${repo_dir}/libampccl/core/comm_init.cc"

rg -q 'comm == nullptr \|\| nranks < 2 \|\| rank < 0' "${hook_file}"
rg -q 'rankNum < 2' "${hook_file}"
rg -q '!domain \|\| nranks < 2 \|\| rank < 0' "${init_file}"
rg -q 'ndev < 2' "${hook_file}"

echo "singleton AMP-CCL domain skip checks passed"
