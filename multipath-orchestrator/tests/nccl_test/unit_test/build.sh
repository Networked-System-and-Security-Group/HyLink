#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCCL_HOME="${NCCL_HOME:-/root/sxb/nccl/build}"
CXX="${CXX:-mpicxx}"
SRC="${1:-${SCRIPT_DIR}/nccl_commsplit_allgather_mpi.cc}"
OUT="${2:-${SCRIPT_DIR}/nccl_commsplit_allgather_mpi}"

include_flags="-I${CUDA_HOME}/include"
lib_dirs="-L${CUDA_HOME}/lib64"
libs="-lnccl -lcudart"

if [[ -n "${NCCL_HOME}" && -d "${NCCL_HOME}/include" ]]; then
  include_flags="${include_flags} -I${NCCL_HOME}/include"
fi
if [[ -n "${NCCL_HOME}" && -d "${NCCL_HOME}/lib" ]]; then
  lib_dirs="${lib_dirs} -L${NCCL_HOME}/lib"
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"${CXX}" -std=c++11 -O2 "${SRC}" -o "${OUT}" ${include_flags} ${lib_dirs} ${libs}
echo "Build done: ${OUT}"
