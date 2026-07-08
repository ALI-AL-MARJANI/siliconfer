"""HQQ from scratch: Half-Quadratic Quantization (calibration-free).

Reference: Badri & Shaji, "Half-Quadratic Quantization of Large Machine
Learning Models" (Mobius Labs, 2023). https://mobiusml.github.io/hqq_blog/

Core idea: standard asymmetric RTN picks scale/zero from the group's exact
min/max, so a single extreme outlier stretches the whole grid and starves
resolution for every other weight in the group. HQQ instead treats the
reconstruction error as hyper-Laplacian (heavy-tailed, i.e. tolerant of a few
large errors) and fits scale/zero under an L_p loss with p<1 — which prefers
clipping a handful of outliers over stretching the grid to include them, if
that buys much finer resolution for the bulk of the group. No calibration
activations are used — this only touches the weight tensor, unlike GPTQ/AWQ.

Implementation note (deviation from the original paper): Mobius Labs' solver
treats the zero-point as a continuous half-quadratic optimization variable
with a closed-form shrinkage update. We instead solve the *same* variational
problem — "which (scale, zero) minimizes the group's L_p reconstruction
loss under real round+clip quantization" — via a direct, from-scratch
equivalent: a grid search over candidate clip ranges, evaluating the
*actual* quantized (round + clip + dequant) L_p loss for each candidate and
keeping the best.

A first version of this search picked candidates by trimming a fixed
*fraction* of elements from each tail (e.g. "drop the top/bottom 5%").
Empirically, on real Qwen2.5-0.5B weight matrices, this was too aggressive:
minimizing the raw mean L_p (p<1) loss unconstrained will happily clip a
chunk of the *ordinary* sample tail — not real outliers — because
`|error|^p` heavily discounts large errors, so shrinking the scale for the
bulk "pays for itself" under the L_p metric even when it clearly hurts MSE
and (empirically measured) downstream perplexity. Real weight tensors from a
trained model rarely have pathological per-element outliers the way the
motivating hyper-Laplacian story assumes; naive Lp-loss minimization treats
their ordinary tail as if it were outliers to discard.

The fix: gate candidate clip ranges on genuine statistical outlier-ness via
a robust z-score (distance from the group median in units of the
median-absolute-deviation, MAD, scaled by 1.4826 to be a consistent
estimator of the standard deviation under normality). Candidates only clip
elements that are >= k MAD from the median, for k in a fixed descending
grid; the loosest candidate (k=∞) is exactly the untrimmed min-max range,
i.e. plain RTN — guaranteeing HQQ's L_p loss is never worse than RTN's by
construction, while only engaging real clipping when a group actually
contains elements far enough from its own center to look like true outliers.

A second, more serious finding while validating against real Qwen2.5-0.5B
weights: trained models can contain lone "super weight" outliers (see Yu et
al., "The Super Weight in Large Language Models," 2024) — single elements
with an extreme robust z-score (58, empirically, in one down_proj group of
this model) that are nonetheless structurally critical: clipping that one
value corrupts a residual-stream channel enough to blow up the whole
forward pass (verified: WikiText-2 PPL went from ~26 to 400+ with a
moderately aggressive k grid). Magnitude statistics computed on weights
alone cannot tell a "safe to compress" heavy tail apart from a lone critical
outlier — that distinction requires sensitivity information (a Hessian, like
GPTQ, or activation magnitudes, like AWQ), which calibration-free HQQ does
not have by design. Since real weight matrices also showed *negligible*
benefit from clipping in the first place (0.99x-1.13x RTN's MSE, i.e. within
noise), the responsible default keeps `k_grid` conservative enough that no
realistic super-weight gets caught: the mechanism is still fully exercised
and verified correct by the unit tests (which use unambiguous synthetic
outliers, deliberately scaled far beyond any plausible real super-weight's
z-score), but in practice it now behaves as a safety net for truly
pathological data rather than a lever expected to move real model PPL.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel

_UINT4_MAX = 15
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
) -> np.ndarray:
    """Quantize a weight matrix with HQQ: asymmetric int4, robust L_p scale/zero fit.

    Args:
        W: float32 array, shape [out_features, in_features].
            in_features must be divisible by group_size (or < group_size, see below).
        group_size: int4 quantization group size (64 or 128).
        p: exponent of the robust reconstruction loss (0 < p < 2). p=0.7 matches
            the hyper-Laplacian prior used in the original HQQ paper; p=2 would
            recover a plain least-squares criterion (no robustness benefit).
        k_grid: candidate outlier thresholds, in units of the group's robust
            z-score (distance from the median in scaled-MAD units). `None`
            means "no clipping" (plain min-max, i.e. RTN) and is always
            included so HQQ can never be worse than RTN under its own L_p
            objective. Smaller k = more aggressive clipping considered.

    Returns:
        W_q: float32 array, same shape as W, fake-dequantized (quantize→dequantize).
    """
    out_features, in_features = W.shape
    if in_features < group_size:
        # Too small to group — fall back to RTN (no outlier search possible)
        from siliconfer.quant.primitives import fake_quantize
        return fake_quantize(W, group_size=in_features, sym=False)

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

        scale = (hi - lo) / _UINT4_MAX
        scale = np.where(scale <= 0, 1.0, scale)
        zero = np.clip(np.round(-lo / scale), 0, _UINT4_MAX)

        q = np.clip(np.round(W_g / scale + zero), 0, _UINT4_MAX)
        recon = scale * (q - zero)

        # The real, robust (L_p, p<1) reconstruction loss — computed on the
        # actual clipped+rounded reconstruction, not a linearized surrogate.
        loss = ((np.abs(W_g - recon) + _EPS) ** p).mean(axis=-1, keepdims=True)   # [out, n_groups, 1]

        if best_loss is None:
            best_loss, best_scale, best_zero = loss, scale, zero
        else:
            better = loss < best_loss
            best_loss = np.where(better, loss, best_loss)
            best_scale = np.where(better, scale, best_scale)
            best_zero = np.where(better, zero, best_zero)

    q_final = np.clip(np.round(W_g / best_scale + best_zero), 0, _UINT4_MAX)
    W_q = (best_scale * (q_final - best_zero)).reshape(out_features, in_features)
    return W_q.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: apply HQQ to every linear layer in a LlamaModel
# ---------------------------------------------------------------------------

def _hqq_weight(
    w: mx.array, group_size: int, p: float, k_grid: tuple[float | None, ...]
) -> mx.array:
    """Apply HQQ to a single MLX weight tensor, padding in_features if needed."""
    in_features = w.shape[-1]
    if in_features < group_size:
        from siliconfer.quant.primitives import fake_quantize
        w_np = np.array(w.astype(mx.float32))
        w_q = fake_quantize(w_np, group_size=in_features, sym=False)
        return mx.array(w_q).astype(w.dtype)

    orig_dtype = w.dtype
    w_np = np.array(w.astype(mx.float32))

    pad = (-in_features) % group_size
    if pad > 0:
        pad_shape = list(w_np.shape)
        pad_shape[-1] = pad
        w_np = np.concatenate([w_np, np.zeros(pad_shape, dtype=np.float32)], axis=-1)

    w_q = hqq_quantize_weight(w_np, group_size=group_size, p=p, k_grid=k_grid)

    if pad > 0:
        w_q = w_q[..., :in_features]

    return mx.array(w_q).astype(orig_dtype)


def apply_hqq(
    model: LlamaModel,
    group_size: int = 128,
    p: float = 0.7,
    k_grid: tuple[float | None, ...] = _DEFAULT_K_GRID,
    verbose: bool = True,
) -> LlamaModel:
    """Apply HQQ int4 quantization to all attention + MLP projections.

    Unlike GPTQ/AWQ, HQQ needs no calibration data or forward passes — each
    weight tensor is quantized independently and layers can be processed in
    any order (no cascading).

    Args:
        model: a loaded LlamaModel (modified in place, returned for convenience).
        group_size: int4 quantization group size (64 or 128).
        p: robust loss exponent (0 < p < 2, default 0.7).
        k_grid: candidate robust-z-score outlier thresholds (see hqq_quantize_weight).
        verbose: print per-layer progress.

    Returns:
        The same model with HQQ-quantized weights.
    """
    n_layers = len(model.layers)
    for i, layer in enumerate(model.layers):
        if verbose:
            print(f"  HQQ layer {i+1}/{n_layers} ...", end=" ", flush=True)

        attn = layer.self_attn
        mlp = layer.mlp

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            proj.weight = _hqq_weight(proj.weight, group_size, p, k_grid)

        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            proj.weight = _hqq_weight(proj.weight, group_size, p, k_grid)

        mx.eval(layer.parameters())

        if verbose:
            print("done")

    return model
