#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "usage: $0 /path/to/libpccl_aicore_reduce.so" >&2
    exit 2
fi

symbols=$(nm -D --defined-only "$1")

if ! grep -Eq '[[:space:]]pcclAicoreReduceAsync$' <<<"$symbols"; then
    echo "pcclAicoreReduceAsync is not exported by $1" >&2
    exit 1
fi

if grep -Eq '[[:space:]]aclrtReduceAsync$' <<<"$symbols"; then
    echo "aclrtReduceAsync must not be exported by $1" >&2
    exit 1
fi
