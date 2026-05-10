#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
source_dir="${here}/results/20261005_143653"
out="${here}/results/split_ratio_8card_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${out}"
cp "${source_dir}/static.log" "${out}/static.log"
cp "${source_dir}/metadata.txt" "${out}/static_source_metadata.txt"
while IFS='=' read -r variable ignored; do
    case "${variable}" in AMPCCL_*|PCCL_*|CCL_*) unset "${variable}" ;; esac
done < <(env)
unset LD_PRELOAD
export NP=8 OP=ag SIZE=33554432 ITERS=1001 WARMUP=10
export ACL_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_ALGO=split_ratio AMPCCL_SPLIT_RATIO_INITIAL_ALPHA=0.075 AMPCCL_STATIC_ALPHA=0.07
export AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8 PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir" AMP_CCL_PATH="${repo}/build/libampccl_hccl.so"
export BIN="${here}/bin/hccl_mpi_primitives"
{
    echo "static_source=${source_dir}/static.log static_alpha=0.07 adaptive=split_ratio initial_alpha=0.075"
    sha256sum "${BIN}" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so"
    env | rg '^(AMPCCL_|PCCL_|NP=|OP=|SIZE=|ITERS=|WARMUP=)' | sort
} >"${out}/metadata.txt"
echo "Results: ${out}"
{
    echo 'algorithm=split_ratio np=8 input_bytes=134217728 total_bytes=1073741824 interference_bytes=268435456 iters=1001 warmup=10 static_alpha=0.07 split_ratio_initial_alpha=0.075 pcie_direction=both ranges=100-200,300-600,700-900 input_mode=reuse_once validation=final_only'
    timeout --signal=TERM --kill-after=10s 3600s bash "${here}/../mpi_primitives/run.sh" \
        --reuse-input --pcie-size=268435456 --pcie-ranges=100-200,300-600,700-900 --pcie-direction=both
} >"${out}/split_ratio.log" 2>&1
python3 "${here}/process_results.py" "${out}" --adaptive split_ratio >"${out}/processing.log"
python3 "${here}/plot_bandwidth.py" "${out}" --adaptive split_ratio
python3 "${here}/plot_variance_envelope.py" "${out}" --adaptive split_ratio
echo "Done: ${out}"
