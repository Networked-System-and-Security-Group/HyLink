#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_run="${1:?Usage: bash check_interference.sh existing_results_dir}"
ascend="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
mkdir -p "${here}/bin" "${here}/diagnostics"
mpicxx -std=c++11 -O2 -I"${ascend}/include" "${here}/test_interference_worker.cc" \
    -L"${ascend}/lib64" -Wl,-rpath,"${ascend}/lib64" -lhccl -lascendcl -pthread \
    -Wl,--wrap=aclrtSetDevice,--wrap=aclrtMemcpy -o "${here}/bin/test_interference_worker"
"${here}/bin/test_interference_worker"
mpicxx -std=c++11 -O2 -I"${ascend}/include" "${here}/probe_background.cc" \
    -L"${ascend}/lib64" -Wl,-rpath,"${ascend}/lib64" -lhccl -lascendcl -pthread \
    -o "${here}/bin/probe_background"
unset LD_PRELOAD
export ASCEND_RT_VISIBLE_DEVICES=0,1 ACL_VISIBLE_DEVICES=0,1
timeout --signal=TERM --kill-after=10s 30s mpirun -np 2 -genvall \
    "${here}/bin/probe_background" | tee "${here}/diagnostics/background_only_$(date -u +%Y%m%d_%H%M%S).log"
bash "${here}/run.sh" --static-only "${source_run}"
