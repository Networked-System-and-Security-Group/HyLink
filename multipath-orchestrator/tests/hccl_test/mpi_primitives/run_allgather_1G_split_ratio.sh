#!/bin/bash
# Test 8-card AllGather with 1GB total size (128MB/rank) using split_ratio algorithm
# 40 iterations to measure stability and bandwidth

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build if needed
if [ ! -f "./hccl_allgather_mpi" ]; then
    echo "Building hccl_allgather_mpi..."
    ./build.sh hccl_mpi_primitives.cc hccl_allgather_mpi
fi

# Test configuration
NRANKS=8
SIZE_PER_RANK=$((128 * 1024 * 1024))  # 128MiB per rank = 1GiB total
WARMUP=5
ITERS=40

# Adaptive-CCL configuration: split_ratio algorithm
export AMPCCL_ENABLE=1
export AMPCCL_ALGO=split_ratio
export AMPCCL_ENABLE_PCIE=1
export AMPCCL_SHM_PARAMS=1
export AMPCCL_LOG_LEVEL=0
export AMPCCL_MIN_MSG_SIZE=$((4 * 1024 * 1024))
export AMPCCL_SPLIT_RATIO_SMOOTH=0.7
export AMPCCL_SPLIT_RATIO_ETA=0.3
export AMPCCL_AVG_BW_LOG=on
export AMPCCL_AVG_BW_WARMUP=$WARMUP

# pcieccl configuration: optimized H2H threads
export PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_IR_PATH=/workspace/pcieccl/ir

# Library paths
export LD_LIBRARY_PATH=/workspace/Adaptive-CCL/build:${LD_LIBRARY_PATH}
export LD_PRELOAD=/workspace/Adaptive-CCL/build/libampccl_hccl.so

echo "=========================================="
echo "8-card AllGather Split Ratio Test"
echo "=========================================="
echo "Configuration:"
echo "  Ranks: $NRANKS"
echo "  Size per rank: $SIZE_PER_RANK bytes (128 MiB)"
echo "  Total size: $((SIZE_PER_RANK * NRANKS / 1024 / 1024)) MiB"
echo "  Warmup: $WARMUP"
echo "  Iterations: $ITERS"
echo "  Algorithm: split_ratio (smooth=0.7, eta=0.3)"
echo "  H2H workers per NUMA: 8"
echo "=========================================="
echo ""

mpirun -n $NRANKS ./hccl_allgather_mpi --op=ag --size=$SIZE_PER_RANK --warmup=$WARMUP --iters=$ITERS

echo ""
echo "=========================================="
echo "Test completed"
echo "=========================================="
