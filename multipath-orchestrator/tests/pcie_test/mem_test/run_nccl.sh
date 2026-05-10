#!/usr/bin/env bash

# PCIe memcpy test run script - NCCL version
#
# Usage:
#   ./run_nccl.sh
#   ./run_nccl.sh --devices=4 --size=64M --iters=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_APP="${SCRIPT_DIR}/mem_test_nccl"

if [ ! -x "$TEST_APP" ]; then
  echo "Error: $TEST_APP not found or not executable."
  echo "Build first: ./build_nccl.sh"
  exit 1
fi

exec "$TEST_APP" "$@"
