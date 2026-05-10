#!/usr/bin/env bash
set -euo pipefail

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
CXX="${CXX:-mpicxx}"
SRC="${1:-hccl_mpi_primitives.cc}"
OUT="${2:-hccl_mpi_primitives}"

INCLUDE_FLAGS="-I${ASCEND_HOME}/include"
LIB_DIRS="-L${ASCEND_HOME}/lib64 -L${ASCEND_HOME}/runtime/lib64"
LIBS="-lhccl -lascendcl -lrt -lpthread"

echo "Using CXX=${CXX} ASCEND_HOME=${ASCEND_HOME}"
echo "Compiling ${SRC} -> ${OUT}"
"${CXX}" -std=c++11 -O2 "${SRC}" -o "${OUT}" \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}
echo "Build done: $(pwd)/${OUT}"
