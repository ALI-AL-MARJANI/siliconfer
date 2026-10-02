"""HQQ-style quantization: calibration-free clip-range search under an L_p loss.

Named after Half-Quadratic Quantization (Badri & Shaji, Mobius Labs, 2023,
https://mobiusml.github.io/hqq_blog/). HQQ fits the quantization parameters
by a half-quadratic iteration on an L_p reconstruction loss with p < 1.
This module keeps that objective and replaces the solver with a candidate
search, so it is not the HQQ algorithm.

For each (row, group):

    m, d = median(w), 1.4826 · MAD(w)
    for k in k_grid:                      # None = plain min/max (RTN)
        clip w to [m − k·d, m + k·d]; quantize asymmetrically; dequantize
        loss_k = Σ |w − ŵ|^p
    keep the candidate with the smallest loss

The untrimmed candidate is always included, so the result is never worse
than asymmetric RTN under the L_p loss.

The default thresholds are deliberately large. A reconstruction loss on
weights alone cannot tell a harmless outlier from a structurally important
one ("super weights", Yu et al., arXiv:2411.07191), and clipping the latter
breaks the model. With the default grid the method behaves like asymmetric
RTN on almost every group; README results show the two are equivalent on
Qwen2.5-0.5B.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from siliconfer.model.llama import LlamaModel

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core HQQ algorithm (pure numpy, no model knowledge)
# ---------------------------------------------------------------------------

_DEFAULT_K_GRID = (150.0, 100.0, 80.0, None)   # None = untrimmed (RTN)


def hqq_quantize_weight(
    W: np.ndarray,
    group_size: int = 128,
    p: float = 0.7,
    k_grid: tuple[float | None, ...] = _DEFAULT_K_GRID,
    bits: int = 4,
) -> np.ndarray:
    """Quantize a weight matrix: asymmetric grid, clip range chosen by L_p loss.

    Args:
        W: float32 [out_features, in_features]; in_features divisible by
            group_size, or smaller than it.
        group_size: quantization group size (64 or 128).
        p: exponent of the reconstruction loss (0 < p < 2). 0.7 follows HQQ;
            2 is plain least squares.
        k_grid: candidate clip thresholds in robust z-score units (distance from
            the median in scaled-MAD units). None means no clipping. Smaller
            values clip more.
        bits: bit-width (4 by default; 2 and 3 are used by mixed_precision.py).

    Returns:
        W_q: float32, same shape as W, quantized then dequantized.
    """
    q_max = float(2 ** bits - 1)
    out_features, in_features = W.shape
    if in_features < group_size:
        # Too small to group — fall back to RTN (no outlier search possible)
        from siliconfer.quant.primitives import fake_quantize
        return fake_quantize(W, group_size=in_features, sym=False, bits=bits)

    n_groups = in_features // group_size
    W_g = W.reshape(out_features, n_groups, group_size).astype(np.float64)

    w_min_full = W_g.min(axis=-1, keepdims=True)
    w_max_full = W_g.max(axis=-1, keepdims=True)
    median = np.median(W_g, axis=-1, keepdims=True)
    mad = np.median(np.abs(W_g - median), axis=-1, keepdims=True)
    robust_std = np.where(mad == 0.0, _EPS, mad) * 1.4826   # consistent estimator under normality

    best_loss = None
    best_scale = None
    best_zero = None

    for k in k_grid:
        if k is None:
            lo, hi = w_min_full, w_max_full
        else:
            lo = np.maximum(w_min_full, median - k * robust_std)
            hi = np.minimum(w_max_full, median + k * robust_std)

        scale = (hi - lo) / q_max
        scale = np.where(scale <= 0, 1.0, scale)
        zero = np.clip(np.round(-lo / scale), 0, q_max)

        q = np.clip(np.round(W_g / scale + zero), 0, q_max)
        recon = scale * (q - zero)

        # L_p loss of the clipped and rounded reconstruction.
        loss = ((np.abs(W_g - recon) + _EPS) ** p).mean(axis=-1, keepdims=True)   # [out, n_groups, 1]

        if best_loss is None:
            best_loss, best_scale, best_zero = loss, scale, zero
        else:
            better = loss < best_loss
            best_loss = np.where(better, loss, best_loss)
            best_scale = np.where(better, scale, best_scale)
            best_zero = np.where(better, zero, best_zero)

    q_final = np.clip(np.round(W_g / best_scale + best_zero), 0, q_max)
    W_q = (best_scale * (q_final - best_zero)).reshape(out_features, in_features)
    return W_q.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: apply HQQ to every linear layer in a LlamaModel
# ---------------------------------------------------------------------------

def _hqq_weight(
    w: mx.array,
    group_size: int,
    p: float,
    k_grid: tuple[float | None, ...],
    bits: int = 4,
) -> mx.array:
    """Apply HQQ to a single MLX weight tensor, padding in_features if needed."""
    in_features = w.shape[-1]
    if in_features < group_size:
        from siliconfer.quant.primitives import fake_quantize
        w_np = np.array(w.astype(mx.float32))
        w_q = fake_quantize(w_np, group_size=in_features, sym=False, bits=bits)
        return mx.array(w_q).astype(w.dtype)

    orig_dtype = w.dtype
    w_np = np.array(w.astype(mx.float32))

    pad = (-in_features) % group_size
    if pad > 0:
        pad_shape = list(w_np.shape)
        pad_shape[-1] = pad
        w_np = np.concatenate([w_np, np.zeros(pad_shape, dtype=np.float32)], axis=-1)

    w_q = hqq_quantize_weight(w_np, group_size=group_size, p=p, k_grid=k_grid, bits=bits)

    if pad > 0:
        w_q = w_q[..., :in_features]

    return mx.array(w_q).astype(orig_dtype)


def apply_hqq(
    model: LlamaModel,
    group_size: int = 128,
    p: float = 0.7,
    k_grid: tuple[float | None, ...] = _DEFAULT_K_GRID,
    bits: int = 4,
    verbose: bool = True,
) -> LlamaModel:
    """Apply the HQQ-style quantizer to all attention and MLP projections.

    No calibration data is needed; each weight is quantized independently.

    Args:
        model: a loaded LlamaModel, modified in place.
        group_size: quantization group size (64 or 128).
        p: loss exponent (0 < p < 2, default 0.7).
        k_grid: candidate clip thresholds (see `hqq_quantize_weight`).
        bits: quantization bit-width.
        verbose: print per-layer progress.

    Returns:
        The same model with quantized weights.
    """
    n_layers = len(model.layers)
    for i, layer in enumerate(model.layers):
        if verbose:
            print(f"  HQQ layer {i+1}/{n_layers} ...", end=" ", flush=True)

        attn = layer.self_attn
        mlp = layer.mlp

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            proj.weight = _hqq_weight(proj.weight, group_size, p, k_grid, bits=bits)

        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            proj.weight = _hqq_weight(proj.weight, group_size, p, k_grid, bits=bits)

        mx.eval(layer.parameters())

        if verbose:
            print("done")

    return model
