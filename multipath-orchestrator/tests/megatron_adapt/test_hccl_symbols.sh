#!/usr/bin/env bash
set -euo pipefail

AMPCCL_ROOT="${ADAPTIVE_CCL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AMPCCL_LIB="${AMPCCL_LIB:-${AMPCCL_ROOT}/build/libampccl_hccl.so}"

if [[ ! -r "${AMPCCL_LIB}" ]]; then
    echo "missing AMP-CCL HCCL library: ${AMPCCL_LIB}" >&2
    exit 2
fi

required_symbols=(
    HcclCreateSubCommConfig
    HcclCommInitRootInfoConfig
    HcclCommInitClusterInfo
    HcclCommInitClusterInfoConfig
    HcclAllGather
    HcclReduceScatter
    HcclAllReduce
)

symbols="$(nm -D --defined-only "${AMPCCL_LIB}")"
for symbol in "${required_symbols[@]}"; do
    if ! grep -Eq "[[:space:]]${symbol}$" <<<"${symbols}"; then
        echo "missing required AMP-CCL export: ${symbol}" >&2
        exit 1
    fi
done

if [[ "${RUN_DLSYM_TEST:-1}" == "1" ]]; then
    AMPCCL_LIB="${AMPCCL_LIB}" AMPCCL_ENABLE=1 AMPCCL_INTERPOSE_DLSYM=1 \
    LD_PRELOAD="${AMPCCL_LIB}" python3 - <<'PY'
import ctypes
import os

path = os.environ["AMPCCL_LIB"]
amp = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
hccl = ctypes.CDLL("libhccl.so")
for name in ("HcclCommInitRootInfo", "HcclCreateSubCommConfig", "HcclAllGather",
             "HcclReduceScatter", "HcclAllReduce"):
    resolved = ctypes.cast(getattr(hccl, name), ctypes.c_void_p).value
    wrapper = ctypes.cast(getattr(amp, name), ctypes.c_void_p).value
    if resolved != wrapper:
        raise SystemExit(f"dlsym bridge mismatch for {name}: {resolved:#x} != {wrapper:#x}")
print("AMP-CCL HCCL dlsym bridge: PASS")
PY
fi

echo "AMP-CCL HCCL symbol exports: PASS"
