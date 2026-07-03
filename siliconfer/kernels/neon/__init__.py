"""NEON q4 kernel Python interface.

On first import, tries to load the compiled siliconfer_neon extension.
If not built yet, all functions fall back to a numpy reference.

Build the extension:
    bash siliconfer/kernels/neon/build_kernel.sh
  or via cmake:
    cd siliconfer/kernels/neon && mkdir -p build && cd build
    cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
    make -j$(sysctl -n hw.logicalcpu)
    cp siliconfer_neon*.so ..
"""

from __future__ import annotations

import importlib
import sys
import pathlib
import numpy as np

# Try to import the compiled extension from this package directory
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
try:
    from . import siliconfer_neon as _lib   # compiled .so
    NEON_AVAILABLE = _lib.neon_available()
    _BACKEND = "neon" if NEON_AVAILABLE else "scalar-c"
except ImportError:
    _lib = None
    NEON_AVAILABLE = False
    _BACKEND = "numpy"
finally:
    if str(_HERE) in sys.path:
        sys.path.remove(str(_HERE))


# ---------------------------------------------------------------------------
# Weight packing (Python / numpy)
# ---------------------------------------------------------------------------

def pack_weights_sym(W: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Quantize and pack a float32 weight matrix for the NEON kernel.

    Args:
        W:          float32 array [out_features, in_features].
        group_size: int4 quantization group size (64 or 128).

    Returns:
        packed: uint8 array [out_features, in_features // 2]
                lo nibble = even channel, hi nibble = odd channel,
                nibble encoding: 0..15 where 0 maps to -8, 8 maps to 0, 15 to 7.
        scales: float32 array [out_features, n_groups].
    """
    from siliconfer.quant.primitives import quantize_sym
    W_q, scales = quantize_sym(W.astype(np.float32), group_size)
    # W_q: int8 [out, in], values in [-8, 7]
    # Pack: lo nibble = even columns, hi nibble = odd columns
    lo = W_q[:, 0::2].astype(np.uint8) & 0x0F
    hi = W_q[:, 1::2].astype(np.uint8) & 0x0F
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed, scales


def pack_weights_asym(
    W: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize (asymmetric) and pack for the NEON kernel.

    Returns:
        packed: uint8 [out, in//2]  — unsigned nibbles in [0,15]
        scales: float32 [out, n_groups]
        zeros:  float32 [out, n_groups]  — zero-point (subtract from nibble before scale)
    """
    from siliconfer.quant.primitives import quantize_asym
    W_q, scales, zeros = quantize_asym(W.astype(np.float32), group_size)
    lo = W_q[:, 0::2].astype(np.uint8) & 0x0F
    hi = W_q[:, 1::2].astype(np.uint8) & 0x0F
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed, scales, zeros


# ---------------------------------------------------------------------------
# Kernel dispatch: compiled NEON → compiled scalar-C → numpy fallback
# ---------------------------------------------------------------------------

def _numpy_gemv_sym(packed, scales, x, group_size):
    """Pure numpy reference GEMV (no C++ required)."""
    out_f, in_h = packed.shape
    in_f = in_h * 2
    n_groups = in_f // group_size
    y = np.zeros(out_f, dtype=np.float32)
    for g in range(n_groups):
        lo = packed[:, g * group_size // 2:(g + 1) * group_size // 2] & 0x0F
        hi = packed[:, g * group_size // 2:(g + 1) * group_size // 2] >> 4
        # Sign extend: nibbles [0..15] → signed [-8..7]
        lo_s = np.where(lo < 8, lo, lo.astype(np.int16) - 16).astype(np.float32)
        hi_s = np.where(hi < 8, hi, hi.astype(np.int16) - 16).astype(np.float32)
        # Reconstruct [out, group_size] weight matrix
        W_g = np.empty((out_f, group_size), dtype=np.float32)
        W_g[:, 0::2] = lo_s
        W_g[:, 1::2] = hi_s
        x_g = x[g * group_size:(g + 1) * group_size]
        y += (W_g @ x_g) * scales[:, g]
    return y


def gemv_sym(
    packed: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """NEON q4 symmetric GEMV: y = dequant(W_packed) @ x.

    Falls back to scalar-C or numpy if the extension is not built.
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    x      = np.ascontiguousarray(x,      dtype=np.float32)
    if _lib is not None:
        return _lib.q4_gemv_sym(packed, scales, x, group_size)
    return _numpy_gemv_sym(packed, scales, x, group_size)


def gemv_scalar(
    packed: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Scalar-C reference GEMV (same result as NEON, slower). For testing."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    x      = np.ascontiguousarray(x,      dtype=np.float32)
    if _lib is not None:
        return _lib.q4_gemv_scalar(packed, scales, x, group_size)
    return _numpy_gemv_sym(packed, scales, x, group_size)


def gemv_asym(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    x: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """NEON q4 asymmetric GEMV."""
    if _lib is None:
        raise RuntimeError("Kernel not built; asymmetric fallback not implemented.")
    return _lib.q4_gemv_asym(
        np.ascontiguousarray(packed, np.uint8),
        np.ascontiguousarray(scales, np.float32),
        np.ascontiguousarray(zeros,  np.float32),
        np.ascontiguousarray(x,      np.float32),
        group_size,
    )


def gemm_sym(
    packed: np.ndarray,
    scales: np.ndarray,
    X: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """NEON q4 symmetric GEMM: Y[T, out] = X[T, in] @ W_q4.T."""
    X      = np.ascontiguousarray(X,      dtype=np.float32)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    if _lib is not None:
        return _lib.q4_gemm_sym(packed, scales, X, group_size)
    # Numpy fallback: loop over tokens
    T = X.shape[0]
    out_f = packed.shape[0]
    Y = np.empty((T, out_f), dtype=np.float32)
    for t in range(T):
        Y[t] = _numpy_gemv_sym(packed, scales, X[t], group_size)
    return Y


__all__ = [
    "NEON_AVAILABLE",
    "pack_weights_sym",
    "pack_weights_asym",
    "gemv_sym",
    "gemv_scalar",
    "gemv_asym",
    "gemm_sym",
]
