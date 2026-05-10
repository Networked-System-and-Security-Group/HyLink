#!/usr/bin/env bash
# Unified HCCL test compile for all subdirs (allgather, allreduce, reducescatter, ...).
# Usage from a primitive directory:
#   ./build.sh my_source.cc my_binary
set -e
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
CXX="${CXX:-g++}"
src=$1
target=$2
if test -z "$src" || test -z "$target"; then
  echo "Usage: $0 <source.cc> <output_binary>" >&2
  exit 1
fi
INCLUDE_FLAGS="-I${ASCEND_HOME}/include"
LIB_DIRS="-L${ASCEND_HOME}/lib64 -L${ASCEND_HOME}/runtime/lib64"
LIBS="-lhccl -lascendcl -lrt -lpthread"
echo "Using ASCEND_HOME=${ASCEND_HOME}"
echo "Compiling ${src} -> ${target}"
"${CXX}" -std=c++11 -O2 "${src}" -o "${target}" \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}
echo "Build done: $(pwd)/${target}"
