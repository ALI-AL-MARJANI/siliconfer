"""Group-wise int4 quantization primitives: symmetric + asymmetric, pack/unpack, fake-quant.

All functions operate on numpy arrays. The quantization axis is always the last axis
(in_features for [out, in] weight matrices), matching GPTQ/AWQ convention.
"""

from __future__ import annotations

import numpy as np

_INT4_SYM_MIN = -8
_INT4_SYM_MAX = 7
_INT4_UINT_MAX = 15


# ---------------------------------------------------------------------------
# Symmetric int4 (range [-8, 7], zero-point = 0)
# ---------------------------------------------------------------------------

def quantize_sym(
    w: np.ndarray,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-wise symmetric int4 quantization along the last axis.

    Args:
        w: float32 array of any shape (..., cols). cols must be divisible by group_size.
        group_size: number of elements per quantization group.

    Returns:
        q: int8 array, same shape as w, values in [-8, 7].
        scales: float32 array of shape (..., n_groups).
    """
    orig_shape = w.shape
    cols = orig_shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    n_groups = cols // group_size

    w_grouped = w.reshape(*orig_shape[:-1], n_groups, group_size)

    abs_max = np.abs(w_grouped).max(axis=-1, keepdims=True)      # (..., n_groups, 1)
    scales = (abs_max / _INT4_SYM_MAX).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.ones_like(scales), scales)

    q = np.round(w_grouped / scales_safe).clip(_INT4_SYM_MIN, _INT4_SYM_MAX).astype(np.int8)
    return q.reshape(orig_shape), scales.squeeze(-1)


def dequantize_sym(
    q: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize symmetric int4: w_approx = q * scale.

    Args:
        q: int8 array, shape (..., cols).
        scales: float32 array, shape (..., n_groups).
        group_size: must match the value used during quantize_sym.

    Returns:
        float32 array, same shape as q.
    """
    orig_shape = q.shape
    cols = orig_shape[-1]
    n_groups = cols // group_size

    q_grouped = q.reshape(*orig_shape[:-1], n_groups, group_size).astype(np.float32)
    w = (q_grouped * scales[..., np.newaxis]).reshape(orig_shape)
    return w


# ---------------------------------------------------------------------------
# Asymmetric int4 (range [0, 15], explicit zero-point)
# ---------------------------------------------------------------------------

def quantize_asym(
    w: np.ndarray,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-wise asymmetric int4 quantization along the last axis.

    Args:
        w: float32 array of any shape (..., cols). cols must be divisible by group_size.
        group_size: number of elements per quantization group.

    Returns:
        q: uint8 array, same shape as w, values in [0, 15].
        scales: float32 array of shape (..., n_groups).
        zeros: float32 array of shape (..., n_groups). Float zero-points for dequant.
    """
    orig_shape = w.shape
    cols = orig_shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    n_groups = cols // group_size

    w_grouped = w.reshape(*orig_shape[:-1], n_groups, group_size)

    w_min = w_grouped.min(axis=-1, keepdims=True)
    w_max = w_grouped.max(axis=-1, keepdims=True)

    scales = ((w_max - w_min) / _INT4_UINT_MAX).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.ones_like(scales), scales)

    zeros = np.round(-w_min / scales_safe).clip(0, _INT4_UINT_MAX).astype(np.float32)

    q = (np.round(w_grouped / scales_safe) + zeros).clip(0, _INT4_UINT_MAX).astype(np.uint8)
    return q.reshape(orig_shape), scales.squeeze(-1), zeros.squeeze(-1)


def dequantize_asym(
    q: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize asymmetric int4: w_approx = (q - zero) * scale.

    Args:
        q: uint8 array, shape (..., cols).
        scales: float32 array, shape (..., n_groups).
        zeros: float32 array, shape (..., n_groups).
        group_size: must match the value used during quantize_asym.

    Returns:
        float32 array, same shape as q.
    """
    orig_shape = q.shape
    cols = orig_shape[-1]
    n_groups = cols // group_size

    q_grouped = q.reshape(*orig_shape[:-1], n_groups, group_size).astype(np.float32)
    w = ((q_grouped - zeros[..., np.newaxis]) * scales[..., np.newaxis]).reshape(orig_shape)
    return w


# ---------------------------------------------------------------------------
# Pack / unpack  (2 int4 values per byte)
# ---------------------------------------------------------------------------

def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack int4 values into uint8, two per byte.

    Layout: low nibble = element at even index, high nibble = element at odd index.
    Accepts int8 (signed sym) or uint8 (unsigned asym) input; uses & 0xF to mask.

    Args:
        q: 1-D or N-D integer array with int4 values. Total elements must be even.

    Returns:
        uint8 array with shape (q.size // 2,).
    """
    flat = q.reshape(-1).astype(np.int32)
    if flat.size % 2 != 0:
        raise ValueError("Total elements must be even for int4 packing")
    lo = (flat[0::2] & 0xF).astype(np.uint8)
    hi = (flat[1::2] & 0xF).astype(np.uint8)
    return lo | (hi << 4)


def unpack_int4(packed: np.ndarray, n: int, signed: bool = True) -> np.ndarray:
    """Unpack uint8 bytes into int4 values.

    Args:
        packed: uint8 array of shape (n // 2,).
        n: total number of int4 values to unpack.
        signed: if True return int8 in [-8, 7]; if False return uint8 in [0, 15].

    Returns:
        int8 or uint8 array of shape (n,).
    """
    lo = (packed & 0x0F).astype(np.int32)
    hi = ((packed >> 4) & 0x0F).astype(np.int32)

    out = np.empty(packed.size * 2, dtype=np.int32)
    out[0::2] = lo
    out[1::2] = hi
    out = out[:n]

    if signed:
        # Two's-complement reinterpret: values > 7 are negative
        out = np.where(out > 7, out - 16, out).astype(np.int8)
    else:
        out = out.astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Fake-quant (quantize → dequantize, isolates accuracy from kernel speed)
# ---------------------------------------------------------------------------

def fake_quantize(
    w: np.ndarray,
    group_size: int = 128,
    sym: bool = True,
) -> np.ndarray:
    """Round-to-nearest int4 fake-quant: quantize then immediately dequantize.

    The returned array has the same shape and dtype as w but contains the
    quantization error that RTN would introduce. Used to measure accuracy
    before any actual packed-int4 kernel exists.

    Args:
        w: float32 array, shape (..., cols). cols must be divisible by group_size.
        group_size: 64 or 128 are standard choices.
        sym: if True use symmetric quant (zero-point = 0), else asymmetric.

    Returns:
        float32 array, same shape as w.
    """
    if sym:
        q, scales = quantize_sym(w, group_size)
        return dequantize_sym(q, scales, group_size)
    else:
        q, scales, zeros = quantize_asym(w, group_size)
        return dequantize_asym(q, scales, zeros, group_size)
