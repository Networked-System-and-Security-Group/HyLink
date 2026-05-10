#!/usr/bin/env bash

# Simple HCCL AllReduce test build script
# Dependencies:
#   - ASCEND_HOME: Ascend CANN install dir (default /usr/local/Ascend/ascend-toolkit/latest)
#   - HCCL/ACL installed with headers and libhccl.so/libascendcl.so
#
# Usage:
#   cd Adaptive-CCL/tests/hccl_test/allreduce
#   chmod +x build.sh
#   ./build.sh allreduce_param.cc hccl_allreduce_param
#
# Output executable: ./hccl_allreduce_param
src=$1
target=$2
set -e

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
CXX="${CXX:-g++}"

INCLUDE_FLAGS="-I${ASCEND_HOME}/include"
LIB_DIRS="-L${ASCEND_HOME}/lib64 -L${ASCEND_HOME}/runtime/lib64"
LIBS="-lhccl -lascendcl -lrt -lpthread"

echo "Using ASCEND_HOME=${ASCEND_HOME}"
echo "Compiling ${target} ..."


"${CXX}" -std=c++11 -O2 ${src} -o ${target} \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}

echo "Build done: $(pwd)/${target}"
