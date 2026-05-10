#!/usr/bin/env bash
set -euo pipefail
# This workflow uses msprof exclusively, including when the parent shell still
# carries settings from a historical custom-trace experiment.
unset CCL_TRACE_DIR PCCL_TRACE_FILE
exec "${CCL_PROFILE_APP:?CCL_PROFILE_APP must point to the benchmark}" "$@"
