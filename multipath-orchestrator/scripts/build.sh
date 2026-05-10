#!/usr/bin/env bash
# Build script for Adaptive-CCL. Produces shared library .so for LD_PRELOAD.
#
# Usage:
#   ./scripts/build.sh [options]
#
# Options (passed to cmake, one required):
#   -DNCCL_ONLY=ON   Build NCCL hook -> libampccl_nccl.so
#   -DHCCL_ONLY=ON   Build HCCL hook -> libampccl_hccl.so
#
# PCIe backend is required: set PCIECCL_ROOT (e.g. export PCIECCL_ROOT=/path/to/pcieccl).
#
# Output: build/libampccl_nccl.so or build/libampccl_hccl.so


set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configure and build in $BUILD_DIR"
cmake "$PROJECT_ROOT" "$@"
cmake --build . --parallel

if [ -f libampccl_nccl.so ]; then
    echo "Built: $BUILD_DIR/libampccl_nccl.so"
elif [ -f libampccl_hccl.so ]; then
    echo "Built: $BUILD_DIR/libampccl_hccl.so"
fi
