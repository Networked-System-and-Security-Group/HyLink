#!/usr/bin/env bash

# PCIe D2H interference - Ascend/ACL build script (HCCL env counterpart of d2h_test.cpp)
# Depends: ASCEND_HOME (CANN install dir), ACL only, no HCCL.
#
# Usage:
#   cd Adaptive-CCL/tests/pcie_test/mem_test
#   chmod +x build_hccl.sh
#   ./build_hccl.sh

set -e

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
CXX="${CXX:-g++}"

INCLUDE_FLAGS="-I${ASCEND_HOME}/include"
LIB_DIRS="-L${ASCEND_HOME}/lib64 -L${ASCEND_HOME}/runtime/lib64"
LIBS="-lascendcl -lpthread"

TARGET="d2h_test_hccl"

echo "Using ASCEND_HOME=${ASCEND_HOME}"
echo "Compiling d2h_test_hccl.cpp -> ${TARGET} ..."

"${CXX}" -std=c++11 -O2 d2h_test_hccl.cpp -o "${TARGET}" \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}

echo "Build done: $(pwd)/${TARGET}"
