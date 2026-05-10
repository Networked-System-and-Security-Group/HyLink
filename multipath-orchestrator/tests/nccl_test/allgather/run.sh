#!/bin/bash

# NCCL allgather test run script
# Optional: override Adaptive-CCL root via env var
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
ENV_PATH="${ADAPTIVE_CCL_ROOT}"
AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_nccl.so"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Optional: load AMPCCL_* env vars from env.sh
# if [ -f "${ENV_PATH}/env.sh" ]; then
#   set -a
#   # shellcheck source=../../env.sh
#   source "${ENV_PATH}/env.sh"
#   set +a
# fi
# export NCCL_DEBUG=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_BUFFSIZE=50331648

export PCCL_FORCE_NVLINK=1
export PCCL_LOG_LEVEL=3
export PCCL_WARMUP_KERNEL=0

export AMPCCL_ENABLE=1
export AMPCCL_STATIC_ALPHA=1
# export AMPCCL_DEBUG_MERGE=1
# export AMPCCL_ALGO=DEBUG

# Log level
export AMPCCL_LOG_LEVEL=off
export AMPCCL_ALGO=split_ratio
# export AMPCCL_ALGO=tcp
# export AMPCCL_ALGO=static

export AMPCCL_MAX_PCIE_RECVBUF_BYTES=$((4 * 1024 * 1024 * 1024))
# export AMPCCL_ENABLE_NCCL_V=1

# Enable PCIe communication (set to 0 for native NCCL only)
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_MIN_CHUNK_SIZE=20
# export AMPCCL_MIN_MSG_SIZE=20

export AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1
export PCCL_LOG_LEVEL=1
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK=0
# export PCCL_DISABLE_STREAM_MEM_OPS=1
# export CUDA_DEVICE_MAX_CONNECTIONS=32

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
# exec "${TEST_APP}" "$@"
# exec stdbuf -oL "${TEST_APP}" "$@"

# APP_CMD="${TEST_APP} $*"
# exec msprof --output=./prof --application="${APP_CMD}"
nsys profile  --trace=cuda,nvtx,osrt --output=./prof "${TEST_APP}" "$@"
