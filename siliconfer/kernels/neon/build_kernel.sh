#!/bin/bash
# Build the siliconfer NEON kernel (.so) without cmake.
# Run from the repo root:  bash siliconfer/kernels/neon/build_kernel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $REPO_ROOT/.venv — activate the venv first."
    exit 1
fi

PYBIND11_INC=$("$PYTHON" -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_EXT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT="$SCRIPT_DIR/siliconfer_neon${PYTHON_EXT}"

echo "Building $OUT ..."
clang++ \
    -std=c++17 -O3 -ffast-math \
    -arch arm64 \
    -fPIC -shared \
    -undefined dynamic_lookup \
    -I"$PYBIND11_INC" \
    -I"$PYTHON_INC" \
    "$SCRIPT_DIR/q4_gemv.cpp" \
    "$SCRIPT_DIR/q4_gemm.cpp" \
    "$SCRIPT_DIR/bindings.cpp" \
    -o "$OUT"

echo "Done: $OUT"
