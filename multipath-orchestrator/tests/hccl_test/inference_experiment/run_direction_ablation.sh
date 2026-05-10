#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
ascend="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
out="${here}/results/direction_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${out}" "${here}/bin"
while IFS='=' read -r variable ignored; do
    case "${variable}" in AMPCCL_*|PCCL_*|CCL_*) unset "${variable}" ;; esac
done < <(env)
unset LD_PRELOAD
export NP=2 OP=ag SIZE=134217728 ITERS=1001 WARMUP=10
export ACL_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_ALGO=static AMPCCL_STATIC_ALPHA=0.39 AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8
export PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir" AMP_CCL_PATH="${repo}/build/libampccl_hccl.so"
export BIN="${here}/bin/hccl_mpi_primitives"
bash "${here}/../mpi_primitives/build.sh" "${here}/../mpi_primitives/hccl_mpi_primitives.cc" "${BIN}" >"${out}/build.log" 2>&1
mpicxx -std=c++11 -O2 -I"${ascend}/include" "${here}/probe_background.cc" \
    -L"${ascend}/lib64" -Wl,-rpath,"${ascend}/lib64" -lhccl -lascendcl -pthread -o "${here}/bin/probe_background"
sha256sum "${BIN}" "${here}/bin/probe_background" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so" >"${out}/metadata.txt"
echo "Results: ${out}"
for direction in d2h h2d both; do
    mkdir -p "${out}/${direction}"
    echo "Starting ${direction}"
    timeout --signal=TERM --kill-after=10s 30s mpirun -np 2 -genvall \
        "${here}/bin/probe_background" --pcie-direction="${direction}" >"${out}/${direction}/background_only.log" 2>&1
    {
        echo "algorithm=static static_alpha=0.39 pcie_direction=${direction} np=2 input_bytes=536870912 interference_bytes_per_direction=268435456 ranges=100-200,300-600,700-900 iters=1001"
        timeout --signal=TERM --kill-after=10s 3600s bash "${here}/../mpi_primitives/run.sh" \
            --reuse-input --pcie-size=268435456 --pcie-ranges=100-200,300-600,700-900 --pcie-direction="${direction}"
    } >"${out}/${direction}/static.log" 2>&1
    python3 "${here}/analyze_interference.py" "${out}/${direction}" "${out}/${direction}/background_only.log" >"${out}/${direction}/audit_stdout.json"
    echo "Finished ${direction}"
done
python3 "${here}/plot_direction_ablation.py" "${out}"
echo "Done: ${out}"
