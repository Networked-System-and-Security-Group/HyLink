#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_dir="$(mktemp -d /tmp/ampccl-shm-startup.XXXXXX)"
g++ -std=c++17 -O0 -g -fno-builtin-memset \
    -I"${repo}/libampccl" \
    "${repo}/tests/unit/test_shm_startup.cc" \
    "${repo}/libampccl/core/shm_store.cc" \
    -pthread -lrt -Wl,--wrap=memset -o "${test_dir}/test_shm_startup"
# Bound the entire process group too, including the initial scheduling gate.
timeout --signal=TERM --kill-after=2s 30s "${test_dir}/test_shm_startup"
