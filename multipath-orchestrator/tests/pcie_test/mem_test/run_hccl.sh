#!/usr/bin/env bash

# PCIe D2H interference - Ascend/ACL run script
# run.sh runs CUDA mem_test; this runs ACL d2h_test_hccl.
#
# Usage: ./run_hccl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_APP="${SCRIPT_DIR}/d2h_test_hccl"

if [ ! -x "$TEST_APP" ]; then
  echo "Error: $TEST_APP not found or not executable."
  echo "Build first: ./build_hccl.sh"
  exit 1
fi

exec "$TEST_APP" "$@"
