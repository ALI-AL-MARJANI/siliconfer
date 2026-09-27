"""SINQ-style quantization: calibration-free per-column rescaling.

SINQ (arXiv:2509.22944) fits a row scale and a column scale before
quantizing. Only the column scale is implemented here. The quantizer in
`quant/primitives.py` already has an independent scale per (row, group), so
multiplying a row by any t > 0 multiplies its scales by t and leaves every
code unchanged: round(t·w / (t·scale)) = round(w / scale). A row scale is
therefore a no-op.

    s = ones(in_features)
    repeat n_iters times:
        W_eff = Q(W · diag(s)) · diag(s)⁻¹
        rel_err[j] = RMS(W[:, j] − W_eff[:, j]) / RMS(W[:, j])
        s *= (rel_err / mean(rel_err)) ^ beta
        clip s to [s_min, s_max]

The update uses relative error. Within a group the rounding step is shared,
so absolute error is about the same for every column and carries no signal;
relative error singles out the small-magnitude columns that the shared step
has crushed (tests/test_sinq.py::test_sinq_relative_error_signal_matters).

Only the weight tensor is needed: no activations, no forward passes.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from siliconfer.model.llama import LlamaModel
from siliconfer.quant.primitives import fake_quantize

_S_MIN = 0.1
_S_MAX = 10.0
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core SINQ algorithm (pure numpy, no model knowledge, no calibration data)
# ---------------------------------------------------------------------------

def sinq_quantize_weight_components(
    W: np.ndarray,
    group_size: int = 128,
    sym: bool = True,
    n_iters: int = 10,
    beta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the column-rescale search; return the grid-aligned weight and the scale.

    Args:
        W: float32 [out_features, in_features], in_features divisible by
            group_size (`_sinq_weight` pads when it is not).
        group_size: quantization group size (64 or 128).
        sym: symmetric (True) or asymmetric (False) grid.
        n_iters: number of refinement rounds.
        beta: step size of the multiplicative update (0 < beta <= 1).

    Returns:
        W_grid: float32 [out, in], Q(W·diag(s)) dequantized. It lies on the
            group grid, so packing it is lossless.
        s: float64 [in], the column scale. The caller applies 1/s to the layer
            input (`Q4Linear.input_scale`). See docs/packing.md.
    """
    out_features, in_features = W.shape
    if in_features < group_size:
        return fake_quantize(W, group_size=in_features, sym=sym), np.ones(in_features, dtype=np.float64)

    W64 = W.astype(np.float64)
    s = np.ones(in_features, dtype=np.float64)
    col_mag = np.sqrt(np.mean(W64 ** 2, axis=0)) + _EPS  # [in_f], fixed reference magnitude

    for _ in range(n_iters):
        W_scaled = (W64 * s[None, :]).astype(np.float32)
        W_q = fake_quantize(W_scaled, group_size=group_size, sym=sym).astype(np.float64)
        W_eff = W_q / s[None, :]

        col_err = np.sqrt(np.mean((W64 - W_eff) ** 2, axis=0))  # [in_f]
        rel_err = col_err / col_mag                              # relative, not absolute
        mean_rel = max(float(rel_err.mean()), _EPS)
        rel_err_safe = np.where(rel_err == 0, _EPS, rel_err)

        s = s * (rel_err_safe / mean_rel) ** beta
        s = np.clip(s, _S_MIN, _S_MAX)

    W_scaled = (W64 * s[None, :]).astype(np.float32)
    W_grid = fake_quantize(W_scaled, group_size=group_size, sym=sym)
    return W_grid, s


def sinq_quantize_weight(
    W: np.ndarray,
    group_size: int = 128,
    sym: bool = True,
    n_iters: int = 10,
    beta: float = 0.5,
) -> np.ndarray:
    """Quantize a weight matrix with column rescaling.

    Returns W_eff = Q(W·diag(s))·diag(1/s), for fake-quant evaluation. W_eff is
    not on the group grid and must not be packed; the serving path uses
    `sinq_quantize_weight_components`.
    """
    W_grid, s = sinq_quantize_weight_components(W, group_size, sym, n_iters, beta)
    return (W_grid.astype(np.float64) / s[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: apply SINQ to every linear layer in a LlamaModel
# ---------------------------------------------------------------------------

def _sinq_weight(
    w: mx.array, group_size: int, sym: bool, n_iters: int, beta: float
) -> tuple[mx.array, np.ndarray, np.ndarray]:
    """Apply the search to one MLX weight tensor, padding in_features if needed.

    Returns (w_eff, W_grid, input_scale), truncated to the original in_features.
    """
    in_features = w.shape[-1]
    if in_features < group_size:
        w_np = np.array(w.astype(mx.float32))
        w_q = fake_quantize(w_np, group_size=in_features, sym=sym)
        input_scale = np.ones(in_features, dtype=np.float32)
        return mx.array(w_q).astype(w.dtype), w_q.astype(np.float32), input_scale

    orig_dtype = w.dtype
    w_np = np.array(w.astype(mx.float32))

    pad = (-in_features) % group_size
    if pad > 0:
        pad_shape = list(w_np.shape)
        pad_shape[-1] = pad
        w_np = np.concatenate([w_np, np.zeros(pad_shape, dtype=np.float32)], axis=-1)

    w_grid, s = sinq_quantize_weight_components(w_np, group_size=group_size, sym=sym, n_iters=n_iters, beta=beta)
    w_eff = (w_grid.astype(np.float64) / s[None, :]).astype(np.float32)
    input_scale = (1.0 / s).astype(np.float32)

    if pad > 0:
        w_grid = w_grid[..., :in_features]
        w_eff = w_eff[..., :in_features]
        input_scale = input_scale[:in_features]

    return mx.array(w_eff).astype(orig_dtype), w_grid, input_scale


def apply_sinq(
    model: LlamaModel,
    group_size: int = 128,
    sym: bool = True,
    n_iters: int = 10,
    beta: float = 0.5,
    verbose: bool = True,
) -> LlamaModel:
    """Apply SINQ-style int4 quantization to all attention and MLP projections.

    Args:
        model: a loaded LlamaModel, modified in place.
        group_size: quantization group size (64 or 128).
        sym: symmetric (True) or asymmetric (False) grid.
        n_iters: refinement rounds per weight.
        beta: column-scale update step size.
        verbose: print per-layer progress.

    Returns:
        The same model with quantized weights.
    """
    n_layers = len(model.layers)
    for i, layer in enumerate(model.layers):
        if verbose:
            print(f"  SINQ layer {i+1}/{n_layers} ...", end=" ", flush=True)

        attn = layer.self_attn
        mlp = layer.mlp

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                     mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            w_eff, w_grid, input_scale = _sinq_weight(proj.weight, group_size, sym, n_iters, beta)
            proj.weight = w_eff
            # Read by q4_loader._pack_and_replace_linears (docs/packing.md).
            proj._sinq_w_grid = w_grid
            proj._sinq_input_scale = input_scale

        mx.eval(layer.parameters())

        if verbose:
            print("done")

    return model
