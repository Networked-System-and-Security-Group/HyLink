#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_APP="${SCRIPT_DIR}/d2d_test_nccl"

if [ ! -x "$TEST_APP" ]; then
  echo "Error: $TEST_APP not found or not executable."
  echo "Build first: bash ${SCRIPT_DIR}/build_nccl.sh"
  exit 1
fi

exec "$TEST_APP" "$@"
