"""Group-wise int4 quantization primitives: symmetric + asymmetric, pack/unpack, fake-quant.

All functions operate on numpy arrays. The quantization axis is always the last axis
(in_features for [out, in] weight matrices), matching GPTQ/AWQ convention.
"""

from __future__ import annotations

import numpy as np

_INT4_SYM_MIN = -8
_INT4_SYM_MAX = 7
_INT4_UINT_MAX = 15

_INT2_SYM_MIN = -2
_INT2_SYM_MAX = 1
_INT2_UINT_MAX = 3


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
# Symmetric int2 (range [-2, 1], zero-point = 0)
#
# Only one positive code (1) is available in 2-bit two's complement, so the
# scale is set by the *positive* max even though the negative side reaches
# further (-2) — the same "clip the wider side to match the narrower one"
# tradeoff every symmetric scheme makes, just far more visible at 2 bits.
# This is intentionally lossier than int4; it exists to be assigned only to
# the layers a sensitivity estimate says can tolerate it (see mixed_precision.py).
# ---------------------------------------------------------------------------

def quantize_sym_int2(
    w: np.ndarray,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-wise symmetric int2 quantization along the last axis.

    Args:
        w: float32 array of any shape (..., cols). cols must be divisible by group_size.
        group_size: number of elements per quantization group.

    Returns:
        q: int8 array, same shape as w, values in [-2, 1].
        scales: float32 array of shape (..., n_groups).
    """
    orig_shape = w.shape
    cols = orig_shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    n_groups = cols // group_size

    w_grouped = w.reshape(*orig_shape[:-1], n_groups, group_size)

    abs_max = np.abs(w_grouped).max(axis=-1, keepdims=True)
    scales = (abs_max / _INT2_SYM_MAX).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.ones_like(scales), scales)

    q = np.round(w_grouped / scales_safe).clip(_INT2_SYM_MIN, _INT2_SYM_MAX).astype(np.int8)
    return q.reshape(orig_shape), scales.squeeze(-1)


def dequantize_sym_int2(
    q: np.ndarray,
    scales: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize symmetric int2: w_approx = q * scale."""
    orig_shape = q.shape
    cols = orig_shape[-1]
    n_groups = cols // group_size

    q_grouped = q.reshape(*orig_shape[:-1], n_groups, group_size).astype(np.float32)
    w = (q_grouped * scales[..., np.newaxis]).reshape(orig_shape)
    return w


# ---------------------------------------------------------------------------
# Asymmetric int2 (range [0, 3], explicit zero-point)
# ---------------------------------------------------------------------------

def quantize_asym_int2(
    w: np.ndarray,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-wise asymmetric int2 quantization along the last axis.

    Returns:
        q: uint8 array, same shape as w, values in [0, 3].
        scales: float32 array of shape (..., n_groups).
        zeros: float32 array of shape (..., n_groups).
    """
    orig_shape = w.shape
    cols = orig_shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    n_groups = cols // group_size

    w_grouped = w.reshape(*orig_shape[:-1], n_groups, group_size)

    w_min = w_grouped.min(axis=-1, keepdims=True)
    w_max = w_grouped.max(axis=-1, keepdims=True)

    scales = ((w_max - w_min) / _INT2_UINT_MAX).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.ones_like(scales), scales)

    zeros = np.round(-w_min / scales_safe).clip(0, _INT2_UINT_MAX).astype(np.float32)

    q = (np.round(w_grouped / scales_safe) + zeros).clip(0, _INT2_UINT_MAX).astype(np.uint8)
    return q.reshape(orig_shape), scales.squeeze(-1), zeros.squeeze(-1)


def dequantize_asym_int2(
    q: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize asymmetric int2: w_approx = (q - zero) * scale."""
    orig_shape = q.shape
    cols = orig_shape[-1]
    n_groups = cols // group_size

    q_grouped = q.reshape(*orig_shape[:-1], n_groups, group_size).astype(np.float32)
    w = ((q_grouped - zeros[..., np.newaxis]) * scales[..., np.newaxis]).reshape(orig_shape)
    return w


# ---------------------------------------------------------------------------
# Generic arbitrary bit-width (2-8) — generalizes the sym/int2 pairs above.
#
# quantize_sym/quantize_sym_int2/quantize_asym/quantize_asym_int2 above are
# kept as-is (existing tests reference them by name), but every quantize
# function only differs from another by its clip range (q_max/q_min derived
# from `bits`) — dequantize_sym/dequantize_asym are already bit-width-agnostic
# (just `q * scale` / `(q - zero) * scale`, no reference to a specific range),
# so they're reused unchanged. Added to let fake_quantize (and HQQ's
# small-matrix fallback) support bits like 3, 5, 6 without a new named
# function per width every time one is needed — see mixed_precision.py's
# 3-bit low-tier experiment (NOTES.md) for why this came up.
# ---------------------------------------------------------------------------

def quantize_sym_n(
    w: np.ndarray,
    group_size: int = 128,
    bits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-wise symmetric quantization at an arbitrary bit-width.

    Equivalent to quantize_sym when bits=4 and quantize_sym_int2 when bits=2
    (same formulas, parameterized instead of duplicated per width).
    """
    if not (2 <= bits <= 8):
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1

    orig_shape = w.shape
    cols = orig_shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    n_groups = cols // group_size

    w_grouped = w.reshape(*orig_shape[:-1], n_groups, group_size)

    abs_max = np.abs(w_grouped).max(axis=-1, keepdims=True)
    scales = (abs_max / q_max).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.ones_like(scales), scales)

    q = np.round(w_grouped / scales_safe).clip(q_min, q_max).astype(np.int8)
    return q.reshape(orig_shape), scales.squeeze(-1)


def quantize_asym_n(
    w: np.ndarray,
    group_size: int = 128,
    bits: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-wise asymmetric quantization at an arbitrary bit-width.

    Equivalent to quantize_asym when bits=4 and quantize_asym_int2 when bits=2.
    """
    if not (2 <= bits <= 8):
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    q_max = 2 ** bits - 1

    orig_shape = w.shape
    cols = orig_shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    n_groups = cols // group_size

    w_grouped = w.reshape(*orig_shape[:-1], n_groups, group_size)

    w_min = w_grouped.min(axis=-1, keepdims=True)
    w_max = w_grouped.max(axis=-1, keepdims=True)

    scales = ((w_max - w_min) / q_max).astype(np.float32)
    scales_safe = np.where(scales == 0.0, np.ones_like(scales), scales)

    zeros = np.round(-w_min / scales_safe).clip(0, q_max).astype(np.float32)

    q = (np.round(w_grouped / scales_safe) + zeros).clip(0, q_max).astype(np.uint8)
    return q.reshape(orig_shape), scales.squeeze(-1), zeros.squeeze(-1)


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
# Pack / unpack  (4 int2 values per byte)
# ---------------------------------------------------------------------------

def pack_int2(q: np.ndarray) -> np.ndarray:
    """Pack int2 values into uint8, four per byte.

    Layout: element i occupies bits [2i, 2i+2) of the byte, i.e. element 0 is
    the lowest 2 bits, element 3 the highest — the 2-bit analogue of
    pack_int4's low/high nibble split.

    Args:
        q: 1-D or N-D integer array with int2 values. Total elements must be
           divisible by 4.

    Returns:
        uint8 array with shape (q.size // 4,).
    """
    flat = q.reshape(-1).astype(np.int32)
    if flat.size % 4 != 0:
        raise ValueError("Total elements must be divisible by 4 for int2 packing")
    b0 = (flat[0::4] & 0x3).astype(np.uint8)
    b1 = (flat[1::4] & 0x3).astype(np.uint8)
    b2 = (flat[2::4] & 0x3).astype(np.uint8)
    b3 = (flat[3::4] & 0x3).astype(np.uint8)
    return b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)


def unpack_int2(packed: np.ndarray, n: int, signed: bool = True) -> np.ndarray:
    """Unpack uint8 bytes into int2 values.

    Args:
        packed: uint8 array of shape (n // 4,).
        n: total number of int2 values to unpack.
        signed: if True return int8 in [-2, 1]; if False return uint8 in [0, 3].

    Returns:
        int8 or uint8 array of shape (n,).
    """
    b0 = (packed & 0x3).astype(np.int32)
    b1 = ((packed >> 2) & 0x3).astype(np.int32)
    b2 = ((packed >> 4) & 0x3).astype(np.int32)
    b3 = ((packed >> 6) & 0x3).astype(np.int32)

    out = np.empty(packed.size * 4, dtype=np.int32)
    out[0::4] = b0
    out[1::4] = b1
    out[2::4] = b2
    out[3::4] = b3
    out = out[:n]

    if signed:
        # Two's-complement reinterpret: values > 1 are negative (2->-2, 3->-1)
        out = np.where(out > 1, out - 4, out).astype(np.int8)
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
    bits: int = 4,
) -> np.ndarray:
    """Round-to-nearest fake-quant: quantize then immediately dequantize.

    The returned array has the same shape and dtype as w but contains the
    quantization error that RTN would introduce. Used to measure accuracy
    before any actual packed kernel exists.

    Args:
        w: float32 array, shape (..., cols). cols must be divisible by group_size.
        group_size: 64 or 128 are standard choices.
        sym: if True use symmetric quant (zero-point = 0), else asymmetric.
        bits: any width in [2, 8] (uses quantize_sym_n/quantize_asym_n — see
            those for why arbitrary widths, not just 2/4, are supported).
            Anything below 4 is only intended for layers a sensitivity
            estimate (see mixed_precision.py) has marked as tolerant of the
            extra error.

    Returns:
        float32 array, same shape as w.
    """
    if sym:
        q, scales = quantize_sym_n(w, group_size, bits=bits)
        return dequantize_sym(q, scales, group_size)
    else:
        q, scales, zeros = quantize_asym_n(w, group_size, bits=bits)
        return dequantize_asym(q, scales, zeros, group_size)
