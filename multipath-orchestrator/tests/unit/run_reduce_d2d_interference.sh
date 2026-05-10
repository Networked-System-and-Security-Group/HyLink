#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset LD_PRELOAD
exec "${script_dir}/test_reduce_d2d_interference" "$@"
