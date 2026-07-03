"""AWQ from scratch: Activation-aware Weight Quantization.

Core idea (Lin et al., 2023):
  Protect salient input channels by applying an equivalent per-channel scale:

    Y = W X  =  (W · diag(s)) · (diag(s)⁻¹ · X)

  where  s[j] = act_scale[j]^α,  act_scale[j] = mean(|X[:,j]|).

  Quantizing (W · diag(s)) keeps high-activation channels at higher effective
  resolution.  diag(s)⁻¹ is folded into the preceding RMSNorm (zero runtime cost).

  α ∈ [0,1] is found by a grid search that minimises per-layer output MSE.

No Hessian required — activation statistics replace Hessian-based error
feedback. AWQ is much faster to run than GPTQ but is slightly less accurate.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.llama import LlamaModel
from siliconfer.model.layers import apply_rope
from siliconfer.quant.primitives import fake_quantize

_MAX_SAMPLES = 512   # token samples kept for the α-search MSE evaluation


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def _np32(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.array(x.astype(mx.float32))


def collect_act_scales_and_samples(
    layer,
    hidden_states: list[mx.array],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect per-channel activation scales and a token subsample per projection.

    Returns:
        dict: projection_name → (act_scales[in_f], X_samples[n_tok, in_f])
        act_scales[j] = mean(|X[:, j]|)  over all calibration tokens.
        q_proj / k_proj / v_proj share one entry (same input).
        gate_proj / up_proj share one entry (same input).
    """
    attn = layer.self_attn
    mlp  = layer.mlp

    raw: dict[str, list[np.ndarray]] = {
        "qkv":     [],
        "o":       [],
        "gate_up": [],
        "down":    [],
    }

    for x in hidden_states:
        B, T, _ = x.shape

        x_norm = layer.input_layernorm(x)
        raw["qkv"].append(_np32(x_norm.reshape(-1, x_norm.shape[-1])))

        q = attn.q_proj(x_norm).reshape(B, T, attn.num_heads,    attn.head_dim).transpose(0, 2, 1, 3)
        k = attn.k_proj(x_norm).reshape(B, T, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = attn.v_proj(x_norm).reshape(B, T, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, 0, attn.rope_freqs)
        k = apply_rope(k, 0, attn.rope_freqs)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(q.dtype) if T > 1 else None
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        raw["o"].append(_np32(attn_out.reshape(-1, attn_out.shape[-1])))

        h = attn.o_proj(attn_out)
        x_post = x + h
        x_pn   = layer.post_attention_layernorm(x_post)
        raw["gate_up"].append(_np32(x_pn.reshape(-1, x_pn.shape[-1])))

        down_in = nn.silu(mlp.gate_proj(x_pn)) * mlp.up_proj(x_pn)
        raw["down"].append(_np32(down_in.reshape(-1, down_in.shape[-1])))

    def _stats(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        X = np.concatenate(chunks, axis=0)               # [total_tok, in_f]
        act_scales = np.abs(X).mean(axis=0)              # [in_f]
        # Keep a random subsample for grid-search MSE
        n = X.shape[0]
        if n > _MAX_SAMPLES:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, _MAX_SAMPLES, replace=False)
            X = X[idx]
        return act_scales.astype(np.float32), X.astype(np.float32)

    s_qkv,     X_qkv     = _stats(raw["qkv"])
    s_o,       X_o       = _stats(raw["o"])
    s_gate_up, X_gate_up = _stats(raw["gate_up"])
    s_down,    X_down    = _stats(raw["down"])

    return {
        "q_proj":    (s_qkv,     X_qkv),
        "k_proj":    (s_qkv,     X_qkv),
        "v_proj":    (s_qkv,     X_qkv),
        "o_proj":    (s_o,       X_o),
        "gate_proj": (s_gate_up, X_gate_up),
        "up_proj":   (s_gate_up, X_gate_up),
        "down_proj": (s_down,    X_down),
    }


# ---------------------------------------------------------------------------
# Core AWQ: α search + quantization
# ---------------------------------------------------------------------------

def awq_search_alpha(
    W: np.ndarray,
    act_scales: np.ndarray,
    X_samples: np.ndarray,
    group_size: int = 128,
    sym: bool = True,
    n_alpha: int = 20,
) -> float:
    """Grid-search α ∈ [0,1] minimising ||W X − Q(W·diag(s)) · diag(s⁻¹) · X||_F.

    Args:
        W:          [out, in_f] float32 weight.
        act_scales: [in_f] float32 per-channel activation magnitudes.
        X_samples:  [n_tok, in_f] float32 activation subsample (columns = tokens).
        n_alpha:    number of grid points (default 20 → step 0.05).

    Returns:
        best α (float in [0, 1]).
    """
    W64  = W.astype(np.float64)
    X64  = X_samples.T.astype(np.float64)   # [in_f, n_tok]
    ref  = W64 @ X64                          # [out, n_tok]  — target

    best_alpha = 0.0
    best_err   = np.linalg.norm(ref - fake_quantize(W, group_size, sym).astype(np.float64) @ X64, "fro")

    act_safe = np.where(act_scales == 0, 1.0, act_scales).astype(np.float64)

    for i in range(1, n_alpha + 1):
        alpha = i / n_alpha                  # [0.05 … 1.0]
        s     = act_safe ** alpha            # [in_f]
        s_inv = 1.0 / s

        W_scaled = (W.astype(np.float64) * s[None, :]).astype(np.float32)
        W_q      = fake_quantize(W_scaled, group_size, sym).astype(np.float64)
        W_eff    = W_q * s_inv[None, :]

        err = np.linalg.norm(ref - W_eff @ X64, "fro")
        if err < best_err:
            best_err   = err
            best_alpha = alpha

    return best_alpha


def awq_quantize_weight(
    W: np.ndarray,
    act_scales: np.ndarray,
    alpha: float,
    group_size: int = 128,
    sym: bool = True,
) -> np.ndarray:
    """Apply AWQ scale + RTN quantization. Returns fake-dequantized W_eff.

    W_eff = Q(W · diag(s)) · diag(s⁻¹),   s = act_scales^alpha.

    1/s is already absorbed into W_eff. Do NOT additionally fold 1/s into the
    preceding norm (that would double-apply the inverse scale and corrupt outputs).
    fold_scale_into_norm() is only correct when storing Q(W·s) without dividing by s.
    """
    act_safe = np.where(act_scales == 0, 1.0, act_scales.astype(np.float64))
    s     = act_safe ** alpha
    s_inv = 1.0 / s

    W_scaled = (W.astype(np.float64) * s[None, :]).astype(np.float32)
    W_q      = fake_quantize(W_scaled, group_size, sym).astype(np.float64)
    return (W_q * s_inv[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Scale folding into RMSNorm
# ---------------------------------------------------------------------------

def fold_scale_into_norm(norm_layer, s: np.ndarray) -> None:
    """Fold 1/s into an RMSNorm weight in-place.

    After folding, the norm effectively pre-scales each channel by 1/s so the
    downstream linear projection can use W·diag(s) without a separate multiply.

    norm.weight[j]  ←  norm.weight[j] / s[j]
    """
    w_np = np.array(norm_layer.weight.astype(mx.float32))
    s_safe = np.where(s == 0, 1.0, s.astype(np.float64))
    w_np = (w_np / s_safe).astype(np.float32)
    norm_layer.weight = mx.array(w_np).astype(norm_layer.weight.dtype)


# ---------------------------------------------------------------------------
# High-level: apply AWQ to a full LlamaModel
# ---------------------------------------------------------------------------

def apply_awq(
    model: LlamaModel,
    calib_sequences: list[mx.array],
    group_size: int = 128,
    sym: bool = True,
    n_alpha: int = 20,
    fold_scales: bool = False,
    verbose: bool = True,
) -> LlamaModel:
    """Apply AWQ int4 quantization to all attention + MLP projections.

    Strategy:
    - For each layer, collect per-channel activation stats and a token subsample.
    - Grid-search α per projection group {q/k/v}, {o}, {gate/up}, {down}.
    - Replace weight with W_eff = Q(W·diag(s)) · diag(s⁻¹).
    - Optionally fold 1/s_{qkv} into input_layernorm and 1/s_{gate_up} into
      post_attention_layernorm (zero-cost inference when using real kernels).
    - Cascaded: hidden states flow through already-quantized layers.

    Args:
        model:          LlamaModel, modified in place.
        calib_sequences: list of (1, T) mx.arrays.
        group_size:     int4 group size.
        sym:            symmetric (True) or asymmetric (False) int4.
        n_alpha:        grid resolution for α search.
        fold_scales:    if True, fold 1/s into the preceding RMSNorm weights for
                        qkv and gate/up groups. Only correct when storing Q(W·s)
                        without the diag(s⁻¹) factor. Since awq_quantize_weight
                        returns Q(W·s)·s⁻¹, leave this False (the default).
        verbose:        print per-layer progress.

    Returns:
        The same model (modified in place).
    """
    hidden_states: list[mx.array] = []
    for seq in calib_sequences:
        h = model.embed_tokens(seq)
        mx.eval(h)
        hidden_states.append(h)

    n_layers = len(model.layers)

    for i, layer in enumerate(model.layers):
        if verbose:
            print(f"  AWQ layer {i+1}/{n_layers} ...", end=" ", flush=True)

        stats = collect_act_scales_and_samples(layer, hidden_states)

        attn = layer.self_attn
        mlp  = layer.mlp

        # Shared α search for projection groups (they share the same input)
        # qkv group
        act_s_qkv, X_qkv = stats["q_proj"]
        W_q_np = np.array(attn.q_proj.weight.astype(mx.float32))
        alpha_qkv = awq_search_alpha(W_q_np, act_s_qkv, X_qkv, group_size, sym, n_alpha)
        s_qkv = np.where(act_s_qkv == 0, 1.0, act_s_qkv.astype(np.float64)) ** alpha_qkv

        # gate/up group
        act_s_gu, X_gu = stats["gate_proj"]
        W_g_np = np.array(mlp.gate_proj.weight.astype(mx.float32))
        alpha_gu = awq_search_alpha(W_g_np, act_s_gu, X_gu, group_size, sym, n_alpha)
        s_gu = np.where(act_s_gu == 0, 1.0, act_s_gu.astype(np.float64)) ** alpha_gu

        # Quantize each projection
        def _quant(proj, act_s, alpha):
            W_np = np.array(proj.weight.astype(mx.float32))
            W_eff = awq_quantize_weight(W_np, act_s, alpha, group_size, sym)
            proj.weight = mx.array(W_eff).astype(proj.weight.dtype)

        _quant(attn.q_proj, act_s_qkv, alpha_qkv)
        _quant(attn.k_proj, act_s_qkv, alpha_qkv)
        _quant(attn.v_proj, act_s_qkv, alpha_qkv)

        # o_proj and down_proj: individual search (different input distributions)
        act_s_o, X_o = stats["o_proj"]
        W_o_np = np.array(attn.o_proj.weight.astype(mx.float32))
        alpha_o = awq_search_alpha(W_o_np, act_s_o, X_o, group_size, sym, n_alpha)
        _quant(attn.o_proj, act_s_o, alpha_o)

        _quant(mlp.gate_proj, act_s_gu, alpha_gu)
        _quant(mlp.up_proj,   act_s_gu, alpha_gu)

        act_s_d, X_d = stats["down_proj"]
        W_d_np = np.array(mlp.down_proj.weight.astype(mx.float32))
        alpha_d = awq_search_alpha(W_d_np, act_s_d, X_d, group_size, sym, n_alpha)
        _quant(mlp.down_proj, act_s_d, alpha_d)

        # Fold 1/s into the preceding RMSNorm weights
        if fold_scales:
            fold_scale_into_norm(layer.input_layernorm,           s_qkv)
            fold_scale_into_norm(layer.post_attention_layernorm,  s_gu)

        mx.eval(layer.parameters())

        # Advance hidden states through the now-quantized layer
        new_hs = []
        for h in hidden_states:
            h_out, _ = layer(h, cache=None)
            mx.eval(h_out)
            new_hs.append(h_out)
        hidden_states = new_hs

        if verbose:
            print(f"α_qkv={alpha_qkv:.2f} α_o={alpha_o:.2f} α_gu={alpha_gu:.2f} α_down={alpha_d:.2f}")

    return model
