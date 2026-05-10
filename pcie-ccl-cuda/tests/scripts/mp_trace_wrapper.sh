#!/bin/bash
# Wrapper for MP test under msprof: sets PCCL_TRACE_FILE per rank
TRACE_PREFIX=${PCCL_TRACE_PREFIX:-/tmp/traces/mp_pccl_rank}
export PCCL_TRACE_FILE="${TRACE_PREFIX}${OMPI_COMM_WORLD_RANK}.json"
exec ./build/test_allgather_mp "$@"
