#!/usr/bin/env bash

# Simple NCCL AllGather test build script
# Dependencies:
#   - CUDA_HOME: CUDA install dir (default /usr/local/cuda)
#   - NCCL/cuda installed with headers and libnccl.so/libcudart.so
#
# Usage:
#   cd Adaptive-CCL/tests/nccl_test/allgather
#   chmod +x build.sh
#   ./build.sh allgather_param.cc nccl_allgather_param
#
# Output executable: ./<target>

src=$1
target=$2
set -e

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"

INCLUDE_FLAGS="-I${CUDA_HOME}/include"
LIB_DIRS="-L${CUDA_HOME}/lib64"
LIBS="-lnccl -lcudart -lpthread -lrt"

echo "Using CUDA_HOME=${CUDA_HOME}"
echo "Compiling ${target} ..."

"${CXX}" -std=c++11 -O2 ${src} -o ${target} \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}

echo "Build done: $(pwd)/${target}"

