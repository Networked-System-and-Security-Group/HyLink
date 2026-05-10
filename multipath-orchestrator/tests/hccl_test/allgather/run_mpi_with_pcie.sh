#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${ADAPTIVE_CCL_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
app="${1:-${script_dir}/hccl_allgather_with_pcie}"
if [[ $# -gt 0 ]]; then shift; fi
export ACL_VISIBLE_DEVICES="${ACL_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export PCCL_IR_PATH="${PCCL_IR_PATH:-${repo}/../pcieccl/ir}"
export AMPCCL_ENABLE="${AMPCCL_ENABLE:-1}"
export AMPCCL_ENABLE_PCIE="${AMPCCL_ENABLE_PCIE:-1}"
export AMPCCL_LOG_LEVEL="${AMPCCL_LOG_LEVEL:-off}"
export AMPCCL_ALGO="${AMPCCL_ALGO:-split_ratio}"
export AMPCCL_STATIC_ALPHA="${AMPCCL_STATIC_ALPHA:-0.16}"
export AMPCCL_SHM_PARAMS="${AMPCCL_SHM_PARAMS:-1}"
# A nonzero HCCL root identity is required for real multiprocess PCIeCCL.
export AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK="${AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK:-0}"
if [[ "${BACKEND:-ampccl}" == native ]]; then
    unset LD_PRELOAD
elif [[ "${BACKEND:-ampccl}" == ampccl ]]; then
    library="${AMP_CCL_PATH:-${repo}/build/libampccl_hccl.so}"
    [[ -f "${library}" ]] || { echo "Library not found: ${library}" >&2; exit 1; }
    export LD_PRELOAD="${library}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
    echo "BACKEND must be native or ampccl" >&2
    exit 2
fi
[[ -x "${app}" ]] || { echo "Executable not found: ${app}" >&2; exit 1; }
exec timeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS:-600}s" \
    mpirun -np "${NP:-4}" -genvall "${app}" "$@"
