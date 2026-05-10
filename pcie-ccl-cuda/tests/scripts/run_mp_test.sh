#!/bin/bash
# Multi-process collective test launcher.
#
# Usage:
#   ./tests/scripts/run_mp_test.sh [nranks] [test_binary]
#
# Examples:
#   ./tests/scripts/run_mp_test.sh 2                    # AllReduce 2 GPUs
#   ./tests/scripts/run_mp_test.sh 8                    # AllReduce 8 GPUs
#   ./tests/scripts/run_mp_test.sh 4 test_allreduce_mp  # explicit binary
#
# Requires: MPI (mpirun), build/test_allreduce_mp

set -e

NRANKS=${1:-2}
TEST_BIN=${2:-test_allreduce_mp}
BUILD_DIR="$(cd "$(dirname "$0")/../../build" && pwd)"
BIN="${BUILD_DIR}/${TEST_BIN}"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build with MPI enabled first:"
    echo "  cd build && cmake .. -DPCCL_USE_CANN=ON && make -j"
    exit 1
fi

# Check MPI availability
if ! command -v mpirun &>/dev/null; then
    echo "Error: mpirun not found in PATH"
    exit 1
fi

echo "=== Multi-Process Test: ${TEST_BIN} with ${NRANKS} GPUs ==="
echo ""

# Launch: one rank per GPU, bind each rank to its GPU
# --allow-run-as-root: needed in container environments
# --bind-to none: let PCCL handle NUMA affinity
exec mpirun \
    --allow-run-as-root \
    --bind-to none \
    -np "${NRANKS}" \
    "$BIN"
