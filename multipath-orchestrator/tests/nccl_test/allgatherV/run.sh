#!/usr/bin/env bash
set -e
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_nccl.so"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [ ! -f "${AMP_CCL_PATH}" ]; then
  echo "Error: ${AMP_CCL_PATH} not found. Build NCCL target first." >&2
  exit 1
fi
TEST_APP="${1:?Usage: $0 <executable> [args...]}"
shift
export LD_PRELOAD="${AMP_CCL_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"
exec "${TEST_APP}" "$@"
