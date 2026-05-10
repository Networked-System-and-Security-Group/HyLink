#!/bin/bash
# Generate merged PCCL + msprof traces for SP and MP AllGather 8GPU.
#
# Output:
#   /tmp/traces/sp_merged.json   — single-process merged trace
#   /tmp/traces/mp_merged_rank*.json — per-rank multi-process merged traces

set -e

TRACE_DIR=/tmp/traces
BUILD_DIR="$(cd "$(dirname "$0")/../../build" && pwd)"
MERGE_SCRIPT="$(cd "$(dirname "$0")/../trace" && pwd)/merge_msprof.py"

rm -rf "${TRACE_DIR}/sp_msprof" "${TRACE_DIR}/mp_msprof_rank"*
rm -f "${TRACE_DIR}/sp_pccl.json" "${TRACE_DIR}/mp_pccl_rank"*.json
rm -f "${TRACE_DIR}/sp_merged.json" "${TRACE_DIR}/mp_merged_rank"*.json
mkdir -p "${TRACE_DIR}"

echo "================================================================"
echo "  Step 1: Single-Process AllGather 8GPU with msprof + PCCL trace"
echo "================================================================"

PCCL_TRACE_FILE="${TRACE_DIR}/sp_pccl.json" \
  msprof --output="${TRACE_DIR}/sp_msprof" \
         --task-time=on --ascendcl=on --runtime-api=on \
         "${BUILD_DIR}/test_allgather" 8 2>&1 | grep -E '(512 MB|PASS|FAIL)'

# Find the msprof JSON output
SP_MSPROF=$(find "${TRACE_DIR}/sp_msprof" -name 'trace_view.json' | head -1)
if [ -z "$SP_MSPROF" ]; then
  echo "Warning: msprof trace_view.json not found, trying msprof_*.json..."
  SP_MSPROF=$(find "${TRACE_DIR}/sp_msprof" -name 'msprof_*.json' | head -1)
fi

if [ -n "$SP_MSPROF" ]; then
  echo "  msprof output: $SP_MSPROF"
  python3 "$MERGE_SCRIPT" "${TRACE_DIR}/sp_pccl.json" "$SP_MSPROF" \
    -o "${TRACE_DIR}/sp_merged.json"
else
  echo "  WARNING: No msprof JSON found, copying PCCL trace as-is"
  cp "${TRACE_DIR}/sp_pccl.json" "${TRACE_DIR}/sp_merged.json"
fi

echo ""
echo "================================================================"
echo "  Step 2: Multi-Process AllGather 8GPU with msprof + PCCL trace"
echo "================================================================"

# For MP, we need msprof to wrap the entire mpirun command.
# Each rank gets its own PCCL trace file.
# msprof captures all 8 processes' NPU activity.

msprof --output="${TRACE_DIR}/mp_msprof" \
       --task-time=on --ascendcl=on --runtime-api=on \
       mpirun --allow-run-as-root --bind-to none -np 8 \
       bash -c "PCCL_TRACE_FILE=${TRACE_DIR}/mp_pccl_rank\${OMPI_COMM_WORLD_RANK}.json exec ${BUILD_DIR}/test_allgather_mp" \
  2>&1 | grep -E '(512 MB|PASS|FAIL)'

# Find MP msprof JSON
MP_MSPROF=$(find "${TRACE_DIR}/mp_msprof" -name 'trace_view.json' | head -1)
if [ -z "$MP_MSPROF" ]; then
  MP_MSPROF=$(find "${TRACE_DIR}/mp_msprof" -name 'msprof_*.json' | head -1)
fi

# Merge each rank's PCCL trace with the shared msprof trace
for RANK_TRACE in "${TRACE_DIR}"/mp_pccl_rank*.json; do
  RANK=$(echo "$RANK_TRACE" | grep -oP 'rank\K[0-9]+')
  if [ -n "$MP_MSPROF" ]; then
    python3 "$MERGE_SCRIPT" "$RANK_TRACE" "$MP_MSPROF" \
      -o "${TRACE_DIR}/mp_merged_rank${RANK}.json"
  else
    cp "$RANK_TRACE" "${TRACE_DIR}/mp_merged_rank${RANK}.json"
  fi
done

echo ""
echo "================================================================"
echo "  Done! Output files:"
echo "================================================================"
ls -lh "${TRACE_DIR}"/{sp_merged,mp_merged_rank*}.json 2>/dev/null

echo ""
echo "Copy to workspace:"
cp "${TRACE_DIR}/sp_merged.json" /workspace/pcieccl/
cp "${TRACE_DIR}"/mp_merged_rank*.json /workspace/pcieccl/ 2>/dev/null || true
ls -lh /workspace/pcieccl/{sp_merged,mp_merged_rank*}.json 2>/dev/null
