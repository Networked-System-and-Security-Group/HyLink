#!/bin/bash

# Optional: override path via env (local dev may not use /workspace/Adaptive-CCL)
ADAPTIVE_CCL_ROOT="${ADAPTIVE_CCL_ROOT:-/workspace/Adaptive-CCL}"
ENV_PATH="${ADAPTIVE_CCL_ROOT}"
AMP_CCL_PATH="${ADAPTIVE_CCL_ROOT}/build/libampccl_hccl.so"

# export HCCL_ENTRY_LOG_ENABLE=1
# pccl (pcieccl) loads IR XML from PCCL_IR_PATH or ./ir (see pcieccl src/ir/ir_loader.cpp).
# Running from this directory makes ./ir wrong; point at the pcieccl source tree.
PCIECCL_ROOT="${PCIECCL_ROOT:-$(cd "${ADAPTIVE_CCL_ROOT}/.." 2>/dev/null && pwd)/pcieccl}"
if [ -z "${PCCL_IR_PATH:-}" ] && [ -d "${PCIECCL_ROOT}/ir" ]; then
  export PCCL_IR_PATH="${PCIECCL_ROOT}/ir"
fi
# export ACL_VISIBLE_DEVICES=0,1,2,3
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ACL_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HCCL_OP_EXPANSION_MODE="AIV"
# Load AMPCCL_* env vars from env.sh
# if [ -f "${ENV_PATH}/env.sh" ]; then
#   set -a
#   # shellcheck source=../../env.sh
#   source "${ENV_PATH}/env.sh"
#   set +a
# fi
# Optional NDJSON agent log (Adaptive-CCL debug): export AMPCCL_AGENT_LOG_PATH=/tmp/amp-debug.log
export AMPCCL_ENABLE=1
# export AMPCCL_DEBUG_MERGE=1

# Set log level
export AMPCCL_LOG_LEVEL=off

export AMPCCL_ALGO=split_ratio
export AMPCCL_STATIC_ALPHA=0.125

# Relay buffer size in bytes (default 512MB)
# 4GB=4294967296, 2GB=2147483648, 1GB=1073741824 , 512MB=536870912, 256MB=268435456, 128MB=134217728,
# 64M=67108864, 32MB=33554432, 16M=16777216, 4MB=4194304
# export AMPCCL_MAX_PCIE_RECVBUF_BYTES=33554432

# Set CCL algorithm
# AMPCCL_ALGO
# Set minimum message size
# AMPCCL_MIN_MSG_SIZE

# Enable PCIe communication
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_MIN_CHUNK_SIZE=20
# export AMPCCL_MIN_MSG_SIZE=20

# These HCCL tests use std::thread (one process, multiple device ranks). PCCL
# multi-process mode (non-zero UniqueId from HcclRootInfo hash) runs ProcessState
# init + SHM barrier and deadlocks: thread 0 waits at barrier for nranks
# arrivals while other threads block on the same init mutex. Force zero id =
# single-process PCCL (see pcieccl tests singleProcessId()). For real multi-
# process MPI, do not set this — broadcast a real pcclUniqueId instead.
export AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=1
export AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK=0

# Single-process multi-thread ranks do not match MPI-style SPMD rounds; disable
# POSIX shm param sync so all threads keep the same default split (see Config::IsShmParamSyncEnabled).
# export AMPCCL_SHM_PARAMS=0

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

# exec msprof --output=test_log --application="${APP_CMD}"
