#!/usr/bin/env bash
# Build pcieccl (PCCL) then Adaptive-CCL with PCIe backend enabled.
#
# Usage:
#   ./scripts/build_with_pcie.sh [pcieccl_dir]
#
# If pcieccl_dir is omitted, uses ../pcieccl relative to Adaptive-CCL root.
# Set ASCEND_HOME for PCCL (e.g. /usr/local/Ascend/ascend-toolkit/latest).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PCIECCL_ROOT="${1:-$(cd "$PROJECT_ROOT/../pcieccl" 2>/dev/null && pwd)}"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"

if [ -z "$PCIECCL_ROOT" ] || [ ! -d "$PCIECCL_ROOT" ]; then
    echo "Usage: $0 [pcieccl_root]"
    echo "  pcieccl_root: path to pcieccl source (default: ../pcieccl)"
    echo "  Build pcieccl first: cd pcieccl && make all DEVICE=ascend ASCEND_HOME=\$ASCEND_HOME"
    exit 1
fi

echo "=== Building pcieccl at $PCIECCL_ROOT (lib + kernels) ==="
(cd "$PCIECCL_ROOT" && make all DEVICE=ascend ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}")
if [ ! -f "$PCIECCL_ROOT/build/lib/libpccl.so" ] || [ ! -f "$PCIECCL_ROOT/build/lib/libpcieccl_kernels.so" ]; then
    echo "pcieccl build failed or libpccl.so / libpcieccl_kernels.so not found"
    exit 1
fi

echo "=== Building Adaptive-CCL (PCIe required) ==="
export PCIECCL_ROOT
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$PROJECT_ROOT" -DCMAKE_BUILD_TYPE=Release -DHCCL_ONLY=ON
cmake --build . --parallel

echo "Done. .so in $BUILD_DIR"
ls -la "$BUILD_DIR"/*.so 2>/dev/null || true
