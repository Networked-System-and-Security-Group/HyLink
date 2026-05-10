#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset LD_PRELOAD
exec "${script_dir}/test_value_wait_ring" "$@"
