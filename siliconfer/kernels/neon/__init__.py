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

import pathlib
import sys

import numpy as np

# Try to import the compiled extension from this package directory
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
try:
    from . import siliconfer_neon as _lib  # compiled .so
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
                nibble encoding: two's complement — 0..7 map to 0..7,
                8..15 map to -8..-1 (sign-extend the nibble, then * scale).
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


def pack_weights_on_grid(
    W_q: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray | None = None,
    group_size: int = 128,
) -> np.ndarray:
    """Pack an already-quantized weight using the grid it was quantized on.

    `pack_weights_sym/asym` derive the grid from the weight (group maximum, or
    min/max). That recovers the original grid only when every group's largest
    code is ±7 (or its codes span 0..15). GPTQ fixes the grid first and then
    moves the weights, so a group can use code −8 or never reach ±7. With the
    scales and zero-points that were actually used, the codes are recovered
    exactly.

    Args:
        W_q:    float [out, in] quantized weight, (code − zero)·scale.
        scales: float32 [out, n_groups].
        zeros:  float32 [out, n_groups], or None for a symmetric grid.

    Returns:
        packed uint8 [out, in // 2], same layout as pack_weights_sym/asym.
    """
    sc = np.repeat(np.where(scales == 0, 1.0, scales).astype(np.float64), group_size, axis=1)
    codes = np.round(W_q.astype(np.float64) / sc)
    if zeros is None:
        codes = codes.clip(-8, 7).astype(np.int8)
    else:
        codes = (codes + np.repeat(zeros.astype(np.float64), group_size, axis=1)).clip(0, 15)
        codes = codes.astype(np.uint8)
    lo = codes[:, 0::2].astype(np.uint8) & 0x0F
    hi = codes[:, 1::2].astype(np.uint8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


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


def _numpy_gemv_asym(packed, scales, zeros, x, group_size):
    """Pure numpy reference asymmetric GEMV (no C++ required)."""
    out_f, in_h = packed.shape
    in_f = in_h * 2
    n_groups = in_f // group_size
    y = np.zeros(out_f, dtype=np.float32)
    for g in range(n_groups):
        lo = (packed[:, g * group_size // 2:(g + 1) * group_size // 2] & 0x0F).astype(np.float32)
        hi = (packed[:, g * group_size // 2:(g + 1) * group_size // 2] >> 4).astype(np.float32)
        W_g = np.empty((out_f, group_size), dtype=np.float32)
        W_g[:, 0::2] = lo
        W_g[:, 1::2] = hi
        x_g = x[g * group_size:(g + 1) * group_size]
        # dequant = (nibble - zero) * scale
        y += (W_g @ x_g - zeros[:, g] * x_g.sum()) * scales[:, g]
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


def gemm_asym(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    X: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Asymmetric q4 GEMM: Y[T, out] = X[T, in] @ ((W_q4 − zero) · scale).T."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    zeros  = np.ascontiguousarray(zeros,  dtype=np.float32)
    X      = np.ascontiguousarray(X,      dtype=np.float32)
    if _lib is not None:
        return _lib.q4_gemm_asym(packed, scales, zeros, X, group_size)
    # Numpy fallback: loop over tokens
    T = X.shape[0]
    out_f = packed.shape[0]
    Y = np.empty((T, out_f), dtype=np.float32)
    for t in range(T):
        Y[t] = _numpy_gemv_asym(packed, scales, zeros, X[t], group_size)
    return Y


__all__ = [
    "NEON_AVAILABLE",
    "pack_weights_sym",
    "pack_weights_asym",
    "gemv_sym",
    "gemv_scalar",
    "gemv_asym",
    "gemm_sym",
    "gemm_asym",
]


def dequant(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray | None = None,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize packed int4 weights to float32 [out, in]. zeros=None: symmetric."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    if zeros is not None:
        zeros = np.ascontiguousarray(zeros, dtype=np.float32)
    if _lib is not None:
        return _lib.q4_dequant(packed, scales, zeros, group_size)
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    if zeros is None:
        lo, hi = np.where(lo >= 8, lo - 16, lo), np.where(hi >= 8, hi - 16, hi)
    W = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    W[:, 0::2], W[:, 1::2] = lo, hi
    if zeros is not None:
        W -= np.repeat(zeros, group_size, axis=1)
    return W * np.repeat(scales, group_size, axis=1)


def set_num_threads(n: int) -> None:
    """Threads used by the compiled GEMV / dequantization (no-op without it).

    Defaults to the number of performance cores. Products under 2**19 weights
    always run on the calling thread.
    """
    if _lib is not None:
        _lib.set_num_threads(int(n))


def get_num_threads() -> int:
    return _lib.get_num_threads() if _lib is not None else 1
