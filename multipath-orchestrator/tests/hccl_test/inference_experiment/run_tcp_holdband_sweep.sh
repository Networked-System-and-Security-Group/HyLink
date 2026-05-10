#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
out="/logs/tcp_holdband_20261006/results_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${out}"
while IFS='=' read -r variable ignored; do
    case "${variable}" in AMPCCL_*|PCCL_*|CCL_*) unset "${variable}" ;; esac
done < <(env)
unset LD_PRELOAD
export NP=2 OP=ag SIZE=134217728 ITERS=1001 WARMUP=10
export ACL_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1
export AMPCCL_ENABLE=1 AMPCCL_ENABLE_PCIE=1 AMPCCL_FEEDBACK=1 AMPCCL_SHM_PARAMS=1
export AMPCCL_ALGO=tcp AMPCCL_STATIC_ALPHA=0.39 AMPCCL_TCP_INITIAL_ALPHA=0.075
export AMPCCL_LOG_LEVEL=off AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID=0
export AMPCCL_TCP_HOLD_RATE_HIGH=1.10
export PCCL_H2H_WORKERS=8 PCCL_H2H_WORKERS_PER_NUMA=8 PCCL_COMPLETION_WAIT_MODE=value PCCL_DISABLE_STAGING_LEASE=0
export PCCL_IR_PATH="${repo}/../pcieccl/ir" AMP_CCL_PATH="${repo}/build/libampccl_hccl.so"
export BIN="${here}/bin/hccl_mpi_primitives"
{
    sha256sum "${BIN}" "${AMP_CCL_PATH}" "${repo}/../pcieccl/build/libpccl.so"
    env | rg '^(AMPCCL_|PCCL_|NP=|OP=|SIZE=|ITERS=|WARMUP=)' | sort
} >"${out}/metadata.txt"
sha256sum "${here}/results/final/tcp.log" "${here}/results/final/static.log" "${here}/results/final.zip" >"${out}/archive_preservation.sha256"
echo "Results: ${out}"
for spec in 0.95_1 0.90_1 0.85_1 0.80_1 0.90_2 0.85_2 0.80_2; do
    low="${spec%_*}"
    export AMPCCL_TCP_HOLD_RATE_LOW="${low}"
    mkdir -p "${out}/${spec}"
    echo "Starting ${spec}"
    {
        echo "algorithm=tcp np=2 input_bytes=536870912 total_bytes=1073741824 interference_bytes=134217728 iters=1001 warmup=10 static_alpha=0.39 tcp_initial_alpha=0.075 pcie_direction=both hold_rate_low=${low} hold_rate_high=1.10 ranges=100-200,300-600,700-900 input_mode=reuse_once validation=final_only"
        timeout --signal=TERM --kill-after=10s 3600s bash "${here}/../mpi_primitives/run.sh" \
            --reuse-input --pcie-size=134217728 --pcie-ranges=100-200,300-600,700-900 --pcie-direction=both
    } >"${out}/${spec}/tcp.log" 2>&1
    echo "Finished ${spec}"
done
sha256sum -c "${out}/archive_preservation.sha256"
echo "Done: ${out}"
