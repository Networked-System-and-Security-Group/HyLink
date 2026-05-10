#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ascend_home="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
cxx="${CXX:-g++}"

if [[ ! -d "${ascend_home}/include" ]]; then
    echo "ASCEND_HOME does not contain include/: ${ascend_home}" >&2
    exit 2
fi

common_flags=(-std=c++17 -O2 -Wall -Wextra -I"${ascend_home}/include")
kernel_source="${PCCL_REDUCE_KERNEL_DIR:-/workspace/pcieccl/reduceKernel_aclrt}"
kernel_build="${script_dir}/.pccl_aicore_reduce_build"

if [[ ! -f "${kernel_source}/CMakeLists.txt" ]]; then
    echo "AICore reduce source not found: ${kernel_source}" >&2
    exit 2
fi

cmake -S "${kernel_source}" -B "${kernel_build}" \
    -DASCEND_PATH="${ascend_home}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF
cmake --build "${kernel_build}" --target pccl_aicore_reduce -j"${JOBS:-$(nproc)}"

common_flags+=("-I${kernel_source}/inc")
base_link_flags=(-L"${ascend_home}/lib64" -L"${ascend_home}/runtime/lib64"
    -Wl,-rpath,"${ascend_home}/lib64:${ascend_home}/runtime/lib64"
    -lascendcl -lrt)
reduce_link_flags=(-L"${kernel_build}" -Wl,-rpath,"${kernel_build}"
    -lpccl_aicore_reduce)

"${cxx}" "${common_flags[@]}" "${script_dir}/test_value_wait_event.cc" \
    -o "${script_dir}/test_value_wait_event" "${base_link_flags[@]}" -ldl
"${cxx}" "${common_flags[@]}" "${script_dir}/test_value_wait_ring.cc" \
    -o "${script_dir}/test_value_wait_ring" "${base_link_flags[@]}" -ldl
"${cxx}" "${common_flags[@]}" "${script_dir}/test_stream_wait_event.cc" \
    -o "${script_dir}/test_stream_wait_event" "${base_link_flags[@]}"
"${cxx}" "${common_flags[@]}" "${script_dir}/test_reduce_hccl_interference.cc" \
    -o "${script_dir}/test_reduce_hccl_interference" "${base_link_flags[@]}" \
    "${reduce_link_flags[@]}" -lhccl -lpthread
"${cxx}" "${common_flags[@]}" "${script_dir}/test_reduce_d2d_interference.cc" \
    -o "${script_dir}/test_reduce_d2d_interference" "${base_link_flags[@]}" \
    "${reduce_link_flags[@]}"
