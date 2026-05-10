#!/bin/bash

ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_nccl.so"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export AMPCCL_ENABLE=1
export AMPCCL_LOG_LEVEL=off
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_MIN_CHUNK_SIZE=20
export AMPCCL_MIN_MSG_SIZE=20

if [ ! -f "${AMP_CCL_PATH}" ]; then
  echo "Error: ${AMP_CCL_PATH} not found."
  echo "Build it first: cd ${ADAPTIVE_CCL_ROOT} && mkdir -p build && cd build && cmake .. -DNCCL_ONLY=ON && make"
  exit 1
fi

TEST_APP="${1:?Usage: $0 <executable> [args...]}"
shift
export LD_PRELOAD="${AMP_CCL_PATH}"
exec stdbuf -oL "${TEST_APP}" "$@"
