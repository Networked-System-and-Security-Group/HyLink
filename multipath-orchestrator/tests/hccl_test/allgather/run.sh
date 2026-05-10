#!/bin/bash

# Optional: override path via env (local dev may not use /workspace/Adaptive-CCL)
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
ENV_PATH="${ADAPTIVE_CCL_ROOT}"
AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_hccl.so"

# Load AMPCCL_* env vars from env.sh
# if [ -f "${ENV_PATH}/env.sh" ]; then
#   set -a
#   # shellcheck source=../../env.sh
#   source "${ENV_PATH}/env.sh"
#   set +a
# fi

export AMPCCL_ENABLE=1
# export AMPCCL_DEBUG_MERGE=1

# Set log level
export AMPCCL_LOG_LEVEL=off

export AMPCCL_ALGO=tcp

# Relay buffer size in bytes (default 512MB)
# 4GB=4294967296, 2GB=2147483648, 1GB=1073741824 , 512MB=536870912, 256MB=268435456, 128MB=134217728, 
# 64M=67108864, 32MB=33554432, 16M=16777216, 4MB=4194304
export AMPCCL_MAX_PCIE_RECVBUF_BYTES=33554432

# Set CCL algorithm
# AMPCCL_ALGO
# Set minimum message size
# AMPCCL_MIN_MSG_SIZE

# Enable PCIe communication
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_MIN_CHUNK_SIZE=20
export AMPCCL_MIN_MSG_SIZE=20

# Check if libampccl_hccl.so exists (will error if not built)
if [ ! -f "${AMP_CCL_PATH}" ]; then
  echo "Error: ${AMP_CCL_PATH} not found."
  echo "Build it first: cd ${ADAPTIVE_CCL_ROOT} && mkdir -p build && cd build && cmake .. -DHCCL_ONLY=ON && make"
  exit 1
fi

# Run test: LD_PRELOAD injects libampccl_hccl.so, followed by the actual command
# Usage: ./run.sh ./hccl_allgather_2  or  ./run.sh ./hccl_allgather 0 0
TEST_APP="${1:?Usage: $0 <executable> [args...]}"
shift
export LD_PRELOAD="${AMP_CCL_PATH}"
exec "${TEST_APP}" "$@"
# APP_CMD="${TEST_APP} $@"

# exec msprof --output=./prof --application="${APP_CMD}"
