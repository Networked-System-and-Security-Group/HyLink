#!/usr/bin/env bash
printf 'argv=%s\n' "$*"
printf 'zero_id=%s shm=%s algo=%s\n' "${AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID:-unset}" "${AMPCCL_SHM_PARAMS:-unset}" "${AMPCCL_ALGO:-unset}"
