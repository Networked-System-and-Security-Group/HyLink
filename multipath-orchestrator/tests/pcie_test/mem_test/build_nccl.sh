#!/usr/bin/env bash

# PCIe memcpy test build script
# Depends: CUDA
#
# Usage:
#   cd Adaptive-CCL/tests/pcie_test/mem_test
#   chmod +x build_nccl.sh
#   ./build_nccl.sh

set -e

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"

INCLUDE_FLAGS="-I${CUDA_HOME}/include"
LIB_DIRS="-L${CUDA_HOME}/lib64"
LIBS="-lcudart -lpthread"

echo "Using CUDA_HOME=${CUDA_HOME}"
echo "Compiling mem_test_nccl ..."

"${CXX}" -std=c++11 -O2 d2h_test_nccl.cpp -o mem_test_nccl \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}

echo "Build done: $(pwd)/mem_test_nccl"
