#!/usr/bin/env bash
set -euo pipefail

runtime_source=${1:-src/runtime/ascend_runtime.cpp}

grep -q 'pcclAicoreReduceAsync' "$runtime_source"
grep -q 'ACL_ERROR_API_NOT_SUPPORT' "$runtime_source"
grep -q 'AICore reduce unsupported; falling back to aclrtReduceAsync' "$runtime_source"
