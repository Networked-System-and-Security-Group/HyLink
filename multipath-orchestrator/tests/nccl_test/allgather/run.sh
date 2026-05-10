#!/bin/bash

# NCCL allgather test run script
# Optional: override Adaptive-CCL root via env var
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
ENV_PATH="${ADAPTIVE_CCL_ROOT}"
AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_nccl.so"

# Optional: load AMPCCL_* env vars from env.sh
# if [ -f "${ENV_PATH}/env.sh" ]; then
#   set -a
#   # shellcheck source=../../env.sh
#   source "${ENV_PATH}/env.sh"
#   set +a
# fi
# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3



export AMPCCL_ENABLE=1
# export AMPCCL_DEBUG_MERGE=1
# export AMPCCL_ALGO=tcp

# Log level
export AMPCCL_LOG_LEVEL=off

# Enable PCIe communication (set to 0 for native NCCL only)
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_MIN_CHUNK_SIZE=20
export AMPCCL_MIN_MSG_SIZE=20

# Check if .so exists
if [ ! -f "${AMP_CCL_PATH}" ]; then
  echo "Error: ${AMP_CCL_PATH} not found."
  echo "Build it first: cd ${ADAPTIVE_CCL_ROOT} && mkdir -p build && cd build && cmake .. -DNCCL_ONLY=ON && make"
  exit 1
fi

# Usage example:
#   ./run.sh ./nccl_allgather_param
#   ./run.sh ./nccl_allgather_param --devices=2 --send-count=1024 --iters=5
TEST_APP="${1:?Usage: $0 <executable> [args...]}"
shift
export LD_PRELOAD="${AMP_CCL_PATH}"
exec stdbuf -oL "${TEST_APP}" "$@"

# APP_CMD="${TEST_APP} $*"
# exec msprof --output=./prof --application="${APP_CMD}"
# nsys profile --trace=cuda,nvtx --output=./prof "${TEST_APP}" "$@"
