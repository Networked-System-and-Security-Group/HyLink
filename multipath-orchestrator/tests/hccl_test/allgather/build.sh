#!/usr/bin/env bash
set -e
if [[ "${1##*/}" == "allgather_with_pcie.cc" ]]; then
    export CXX="${CXX:-mpicxx}"
fi
exec "$(cd "$(dirname "$0")/.." && pwd)/build.sh" "$@"
