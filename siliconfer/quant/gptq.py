"""GPTQ from scratch: Optimal Brain Quantization for LLMs.

Core algorithm:
  - Hessian H = 2 X Xᵀ  (collected by calibration.py)
  - Dampen: H += λ·mean(diag H)·I,  λ = 0.01
  - Compute H⁻¹ via Cholesky, then take its upper Cholesky factor U (H⁻¹ = UᵀU).
  - Quantize columns left→right in blocks, propagating error via U (not H⁻¹ directly).

WHY U not H⁻¹: U[q,q] is the conditional inverse-Hessian diagonal for column q given
that columns 0..q-1 have been quantized (Schur complement), not the unconditional
H⁻¹[q,q]. Using H⁻¹[q,q] directly gives wrong error-correction magnitudes.
This matches the original GPTQ implementation (Frantar et al., 2022).

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", ICLR 2023.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.quant.calibration import collect_layer_H

_DAMP = 0.01         # Hessian dampening λ
_INT4_SYM_MIN = -8
_INT4_SYM_MAX = 7
_INT4_UINT_MAX = 15


# ---------------------------------------------------------------------------
# Core GPTQ algorithm (pure numpy, no model knowledge)
# ---------------------------------------------------------------------------

def gptq_quantize_weight(
    W: np.ndarray,
    H: np.ndarray,
    group_size: int = 128,
    block_size: int = 128,
    sym: bool = True,
) -> np.ndarray:
    """Quantize a single weight matrix using GPTQ with error feedback.

    Args:
        W: float32 array, shape [out_features, in_features].
            in_features must be divisible by group_size.
        H: float64 array, shape [in_features, in_features].
            Hessian H = 2 X Xᵀ from calibration activations.
        group_size: int4 quantization group size (64 or 128).
        block_size: number of columns processed per GPTQ block (typically 128).
        sym: True for symmetric int4 (zero-point = 0), False for asymmetric.

    Returns:
        W_q: float32 array, same shape as W, fake-dequantized.
    """
    out_features, in_features = W.shape
    if in_features < group_size:
        # Too small to group — fall back to RTN
        from siliconfer.quant.primitives import fake_quantize
        return fake_quantize(W, group_size=in_features, sym=sym)

    # --- 1. Prepare H in float64 ---
    H = H.astype(np.float64)
    diag_mean = float(np.diag(H).mean())
    H += _DAMP * diag_mean * np.eye(in_features, dtype=np.float64)

    # --- 2. Compute U = upper Cholesky of H⁻¹ (H⁻¹ = UᵀU, U upper-triangular) ---
    # Step 1: Cholesky of H → H_inv
    try:
        L = np.linalg.cholesky(H)           # H = L Lᵀ, L lower-triangular
    except np.linalg.LinAlgError:
        H += 1e-3 * diag_mean * np.eye(in_features, dtype=np.float64)
        L = np.linalg.cholesky(H)
    L_inv = np.linalg.inv(L)
    H_inv = L_inv.T @ L_inv                # [in, in] float64, H⁻¹ = (L⁻¹)ᵀ L⁻¹

    # Step 2: upper Cholesky of H_inv → U
    # H_inv = L2 L2ᵀ  (numpy always returns lower L2)
    # U = L2ᵀ  (upper-triangular), so H_inv = Uᵀ U
    try:
        L2 = np.linalg.cholesky(H_inv)
    except np.linalg.LinAlgError:
        H_inv += 1e-8 * np.eye(in_features, dtype=np.float64)
        L2 = np.linalg.cholesky(H_inv)
    U = L2.T                               # [in, in] upper-triangular float64

    # --- 3. Pre-compute group scales from original W ---
    # Scales are fixed before quantization so all columns in a group share one scale.
    n_groups = (in_features + group_size - 1) // group_size
    scales = np.zeros((out_features, n_groups), dtype=np.float32)
    zeros  = np.zeros((out_features, n_groups), dtype=np.float32)

    for g in range(n_groups):
        gs = g * group_size
        ge = min(gs + group_size, in_features)
        W_g = W[:, gs:ge]
        if sym:
            abs_max = np.abs(W_g).max(axis=1)       # [out]
            scales[:, g] = abs_max / _INT4_SYM_MAX
        else:
            w_min = W_g.min(axis=1)
            w_max = W_g.max(axis=1)
            scales[:, g] = (w_max - w_min) / _INT4_UINT_MAX
            sc_safe = np.where(scales[:, g] == 0, 1.0, scales[:, g])
            zeros[:, g]  = np.round(-w_min / sc_safe).clip(0, _INT4_UINT_MAX)

    # --- 4. Block GPTQ (using U, the upper Cholesky of H⁻¹) ---
    # U[q, q] is the conditional inverse-Hessian diagonal for column q;
    # U[q, c] for c > q drives the error correction to subsequent columns.
    W_q = W.astype(np.float64)              # updated in-place

    for blk_start in range(0, in_features, block_size):
        blk_end = min(blk_start + block_size, in_features)
        B = blk_end - blk_start

        errs = np.zeros((out_features, B), dtype=np.float64)

        for qi in range(B):
            q = blk_start + qi
            g = q // group_size

            w_col = W_q[:, q]               # [out] float64
            sc = scales[:, g].astype(np.float64)
            sc_safe = np.where(sc == 0, 1.0, sc)

            if sym:
                w_q_col = np.round(w_col / sc_safe).clip(_INT4_SYM_MIN, _INT4_SYM_MAX) * sc_safe
            else:
                z = zeros[:, g].astype(np.float64)
                w_q_col = (np.round(w_col / sc_safe) + z).clip(0, _INT4_UINT_MAX)
                w_q_col = (w_q_col - z) * sc_safe

            u_qq = U[q, q]
            if u_qq < 1e-12:
                errs[:, qi] = 0.0
            else:
                err = (w_col - w_q_col) / u_qq     # [out]
                errs[:, qi] = err
                # Update within-block remaining columns via U row
                if qi + 1 < B:
                    W_q[:, blk_start + qi + 1 : blk_end] -= (
                        err[:, None] * U[q, blk_start + qi + 1 : blk_end]
                    )

            W_q[:, q] = w_q_col

        # Propagate accumulated block error to all subsequent columns
        if blk_end < in_features:
            W_q[:, blk_end:] -= errs @ U[blk_start:blk_end, blk_end:]

    return W_q.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: apply GPTQ to every linear layer in a LlamaModel
# ---------------------------------------------------------------------------

def _quantize_proj(proj, H_np: np.ndarray, group_size: int, block_size: int, sym: bool) -> None:
    """Quantize a single nn.Linear weight in-place."""
    W_np = np.array(proj.weight.astype(mx.float32))
    W_q  = gptq_quantize_weight(W_np, H_np, group_size, block_size, sym)
    proj.weight = mx.array(W_q).astype(proj.weight.dtype)


def apply_gptq(
    model: LlamaModel,
    calib_sequences: list[mx.array],
    group_size: int = 128,
    block_size: int = 128,
    sym: bool = True,
    verbose: bool = True,
) -> LlamaModel:
    """Apply GPTQ int4 quantization to all attention + MLP projections.

    Uses a cascaded strategy: each layer is quantized using calibration
    activations that flow through the already-quantized previous layers.

    Args:
        model: a loaded LlamaModel (modified in place, returned for convenience).
        calib_sequences: list of (1, seq_len) mx.arrays from load_calibration_sequences().
        group_size: int4 quantization group size (64 or 128).
        block_size: GPTQ block size (number of columns per error-propagation block).
        sym: symmetric (True) or asymmetric (False) int4.
        verbose: print per-layer progress.

    Returns:
        The same model with GPTQ-quantized weights.
    """
    # Embed calibration sequences → hidden states entering layer 0
    hidden_states: list[mx.array] = []
    for seq in calib_sequences:
        h = model.embed_tokens(seq)          # (1, T, hidden)
        mx.eval(h)
        hidden_states.append(h)

    n_layers = len(model.layers)

    for i, layer in enumerate(model.layers):
        if verbose:
            print(f"  GPTQ layer {i+1}/{n_layers} ...", end=" ", flush=True)

        # Collect H for all projections in this layer from current hidden_states
        H_dict = collect_layer_H(layer, hidden_states)

        # Quantize each projection
        attn = layer.self_attn
        mlp  = layer.mlp

        for name, proj in (
            ("q_proj",    attn.q_proj),
            ("k_proj",    attn.k_proj),
            ("v_proj",    attn.v_proj),
            ("o_proj",    attn.o_proj),
            ("gate_proj", mlp.gate_proj),
            ("up_proj",   mlp.up_proj),
            ("down_proj", mlp.down_proj),
        ):
            _quantize_proj(proj, H_dict[name], group_size, block_size, sym)

        mx.eval(layer.parameters())

        # Advance hidden states through this (now quantized) layer
        new_hs: list[mx.array] = []
        for h in hidden_states:
            h_out, _ = layer(h, cache=None)
            mx.eval(h_out)
            new_hs.append(h_out)
        hidden_states = new_hs

        if verbose:
            print("done")

    return model
