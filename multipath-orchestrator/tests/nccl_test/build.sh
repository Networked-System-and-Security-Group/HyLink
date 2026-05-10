#!/usr/bin/env bash
# Unified NCCL test compile for all subdirs.
# Usage from a primitive directory:
#   ./build.sh my_source.cc my_binary
# Optional: NCCL_HOME for extra headers/libs.
set -e
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
if [ -d "${NCCL_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
CXX="${CXX:-g++}"
src=$1
target=$2
if test -z "$src" || test -z "$target"; then
  echo "Usage: $0 <source.cc> <output_binary>" >&2
  exit 1
fi
INCLUDE_FLAGS="-I${CUDA_HOME}/include"
LIB_DIRS="-L${CUDA_HOME}/lib64"
LIBS="-lnccl -lcudart -lpthread -lrt"
if test -n "$NCCL_HOME" && test -d "${NCCL_HOME}/include"; then
  INCLUDE_FLAGS="${INCLUDE_FLAGS} -I${NCCL_HOME}/include"
fi
if test -n "$NCCL_HOME" && test -d "${NCCL_HOME}/lib"; then
  LIB_DIRS="${LIB_DIRS} -L${NCCL_HOME}/lib"
fi
echo "Using CUDA_HOME=${CUDA_HOME}"
echo "Compiling ${src} -> ${target}"
"${CXX}" -std=c++11 -O2 "${src}" -o "${target}" \
  ${INCLUDE_FLAGS} ${LIB_DIRS} ${LIBS}
echo "Build done: $(pwd)/${target}"
