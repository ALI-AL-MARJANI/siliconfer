"""AWQ: Activation-aware Weight Quantization (Lin et al., arXiv:2306.00978).

Protect salient input channels with an equivalent per-channel scale:

    Y = W X = (W · diag(s)) · (diag(s)⁻¹ · X),     s[j] = mean(|X[:, j]|)^α

Quantizing W·diag(s) gives high-activation channels finer effective
resolution. α ∈ [0, 1] is grid-searched to minimise output MSE. The serving
path stores Q(W·diag(s)) and applies 1/s to the layer input.

Two options follow the official implementation (mit-han-lab/llm-awq); both
default to off:

  block_loss  score each α on the output of the enclosing block (attention
              for q/k/v, MLP for gate/up) with the whole group quantized,
              instead of on one projection's own output.
  auto_clip   after scaling, shrink each (row, group)'s clipping range by up
              to 50% when that lowers the group's output error (not applied
              to q/k).

Differences from llm-awq: docs/awq-vs-reference.md.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from siliconfer.model.layers import apply_rope
from siliconfer.model.llama import LlamaModel
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


def awq_search_alpha_by_loss(
    Ws: list[np.ndarray],
    act_scales: np.ndarray,
    loss_fn,
    group_size: int = 128,
    sym: bool = True,
    n_alpha: int = 20,
) -> float:
    """Grid-search one shared α for a group of projections, scored by `loss_fn`.

    `loss_fn(W_effs)` receives one fake-dequantized weight per entry of `Ws`
    and returns a scalar error — e.g. the MSE of the enclosing block's output.
    α = 0 (plain RTN) is the starting point, so the result is never worse than
    RTN on the calibration data.
    """
    best_alpha = 0.0
    best_err = loss_fn([fake_quantize(W, group_size, sym) for W in Ws])
    for i in range(1, n_alpha + 1):
        alpha = i / n_alpha
        err = loss_fn([awq_quantize_weight(W, act_scales, alpha, group_size, sym) for W in Ws])
        if err < best_err:
            best_err, best_alpha = err, alpha
    return best_alpha


def awq_clip_weight(
    W_scaled: np.ndarray,
    X_scaled: np.ndarray,
    group_size: int = 128,
    sym: bool = True,
    n_grid: int = 20,
    max_shrink: float = 0.5,
) -> np.ndarray:
    """Search a clipping range per (row, group) that minimises the group's output error.

    For each candidate `max = org_max · (1 − i/n_grid)`, `i < max_shrink·n_grid`,
    the group is clamped to ±max and quantized; the candidate with the lowest
    mean-squared error of the group's partial output `w_g · x_g` over the
    samples is kept. `i = 0` is "no clipping", so the result is never worse
    than unclipped quantization on these samples.

    Args:
        W_scaled: [out, in] weight, already multiplied by the AWQ column scale.
        X_scaled: [n_tok, in] input samples as this weight sees them (divided by s).

    Returns:
        float32 [out, in]: W_scaled clamped to the chosen ranges (not yet quantized).
    """
    out_f, in_f = W_scaled.shape
    n_groups = in_f // group_size
    W64 = W_scaled.astype(np.float64)

    # Group output error ||d·x||² averaged over samples = d · C_g · dᵀ, C_g = XᵀX / n.
    Xg = X_scaled.astype(np.float64).reshape(-1, n_groups, group_size).transpose(1, 0, 2)
    C = np.matmul(Xg.transpose(0, 2, 1), Xg) / Xg.shape[1]          # [n_groups, G, G]

    org_max = np.abs(W64).reshape(out_f, n_groups, group_size).max(axis=-1)
    best_max = org_max.copy()
    best_err = np.full((out_f, n_groups), np.inf)

    for i in range(int(max_shrink * n_grid)):
        cur_max = org_max * (1 - i / n_grid)
        bound = np.repeat(cur_max, group_size, axis=1)
        W_q = fake_quantize(np.clip(W64, -bound, bound).astype(np.float32), group_size, sym)
        D = (W_q.astype(np.float64) - W64).reshape(out_f, n_groups, group_size).transpose(1, 0, 2)
        err = (np.matmul(D, C) * D).sum(axis=-1).T                   # [out, n_groups]
        better = err < best_err
        best_err[better] = err[better]
        best_max[better] = cur_max[better]

    bound = np.repeat(best_max, group_size, axis=1)
    return np.clip(W64, -bound, bound).astype(np.float32)


def awq_quantize_weight_components(
    W: np.ndarray,
    act_scales: np.ndarray,
    alpha: float,
    group_size: int = 128,
    sym: bool = True,
    clip_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return AWQ's grid-aligned quantized weight and its column scale.

    Args:
        clip_samples: optional [n_tok, in] unscaled input samples. When given,
            the scaled weight is clipped with `awq_clip_weight` before quantizing.

    Returns:
        W_grid: float32 [out, in], Q(W·diag(s)) dequantized. It lies on the
            group grid, so packing it is lossless.
        s: float64 [in], the column scale. The caller applies 1/s to the layer
            input (`Q4Linear.input_scale`). See docs/packing.md.
    """
    act_safe = np.where(act_scales == 0, 1.0, act_scales.astype(np.float64))
    s = act_safe ** alpha

    W_scaled = (W.astype(np.float64) * s[None, :]).astype(np.float32)
    if clip_samples is not None:
        X_scaled = clip_samples.astype(np.float64) / s[None, :]
        W_scaled = awq_clip_weight(W_scaled, X_scaled, group_size, sym)
    W_grid   = fake_quantize(W_scaled, group_size, sym)
    return W_grid, s


def awq_quantize_weight(
    W: np.ndarray,
    act_scales: np.ndarray,
    alpha: float,
    group_size: int = 128,
    sym: bool = True,
) -> np.ndarray:
    """Return W_eff = Q(W · diag(s)) · diag(s)⁻¹ with s = act_scales^alpha.

    For fake-quant evaluation. W_eff is not on the group grid and must not be
    packed; the serving path uses `awq_quantize_weight_components`.
    """
    W_grid, s = awq_quantize_weight_components(W, act_scales, alpha, group_size, sym)
    return (W_grid.astype(np.float64) * (1.0 / s)[None, :]).astype(np.float32)


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

def _block_loss_fn(projs, forward, ref: mx.array):
    """Build loss_fn(W_effs) = MSE(forward() with `projs` set to W_effs, ref).

    The projections' original weights are restored after every evaluation.
    """
    originals = [p.weight for p in projs]
    ref32 = ref.astype(mx.float32)

    def loss_fn(W_effs: list[np.ndarray]) -> float:
        for p, W, w0 in zip(projs, W_effs, originals):
            p.weight = mx.array(W).astype(w0.dtype)
        err = mx.mean(mx.square(forward().astype(mx.float32) - ref32)).item()
        for p, w0 in zip(projs, originals):
            p.weight = w0
        return err

    return loss_fn


def apply_awq(
    model: LlamaModel,
    calib_sequences: list[mx.array],
    group_size: int = 128,
    sym: bool = True,
    n_alpha: int = 20,
    fold_scales: bool = False,
    block_loss: bool = False,
    auto_clip: bool = False,
    n_loss_seqs: int = 32,
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
        block_loss:     score α for {q/k/v} on the attention module's output and
                        for {gate/up} on the MLP's output, with the whole group
                        quantized (llm-awq's `module2inspect`). When False, α is
                        scored on q_proj's / gate_proj's own output only.
        auto_clip:      clip each (row, group) of v/o/gate/up/down after scaling
                        (llm-awq's `auto_clip`; q/k are skipped there too).
        n_loss_seqs:    calibration sequences used for the block loss.
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
        act_s_gu, X_gu = stats["gate_proj"]
        if block_loss:
            hs = mx.concatenate(hidden_states[:n_loss_seqs], axis=0)
            x_norm = layer.input_layernorm(hs)
            attn_ref, _ = attn(x_norm)
            x_pn = layer.post_attention_layernorm(hs + attn_ref)
            mlp_ref = mlp(x_pn)
            mx.eval(x_norm, attn_ref, x_pn, mlp_ref)

            alpha_qkv = awq_search_alpha_by_loss(
                [np.array(p.weight.astype(mx.float32)) for p in (attn.q_proj, attn.k_proj, attn.v_proj)],
                act_s_qkv,
                _block_loss_fn((attn.q_proj, attn.k_proj, attn.v_proj),
                               lambda: attn(x_norm)[0], attn_ref),
                group_size, sym, n_alpha,
            )
            alpha_gu = awq_search_alpha_by_loss(
                [np.array(p.weight.astype(mx.float32)) for p in (mlp.gate_proj, mlp.up_proj)],
                act_s_gu,
                _block_loss_fn((mlp.gate_proj, mlp.up_proj), lambda: mlp(x_pn), mlp_ref),
                group_size, sym, n_alpha,
            )
        else:
            W_q_np = np.array(attn.q_proj.weight.astype(mx.float32))
            alpha_qkv = awq_search_alpha(W_q_np, act_s_qkv, X_qkv, group_size, sym, n_alpha)
            W_g_np = np.array(mlp.gate_proj.weight.astype(mx.float32))
            alpha_gu = awq_search_alpha(W_g_np, act_s_gu, X_gu, group_size, sym, n_alpha)
        s_qkv = np.where(act_s_qkv == 0, 1.0, act_s_qkv.astype(np.float64)) ** alpha_qkv
        s_gu = np.where(act_s_gu == 0, 1.0, act_s_gu.astype(np.float64)) ** alpha_gu

        # Each projection keeps two representations:
        #   proj.weight                              W_eff, for fake-quant evaluation
        #   proj._awq_w_grid, proj._awq_input_scale  grid-aligned weight and 1/s,
        #                                            read by q4_loader when packing
        def _quant(proj, act_s, alpha, clip_X=None):
            W_np = np.array(proj.weight.astype(mx.float32))
            W_grid, s = awq_quantize_weight_components(
                W_np, act_s, alpha, group_size, sym, clip_samples=clip_X if auto_clip else None
            )
            W_eff = (W_grid.astype(np.float64) * (1.0 / s)[None, :]).astype(np.float32)
            proj.weight = mx.array(W_eff).astype(proj.weight.dtype)
            proj._awq_w_grid = W_grid.astype(np.float32)
            proj._awq_input_scale = (1.0 / s).astype(np.float32)

        # o_proj and down_proj: individual search (different input distributions).
        # Searched before any projection of this layer is replaced, so every α
        # is chosen against the layer's original weights.
        act_s_o, X_o = stats["o_proj"]
        W_o_np = np.array(attn.o_proj.weight.astype(mx.float32))
        alpha_o = awq_search_alpha(W_o_np, act_s_o, X_o, group_size, sym, n_alpha)

        act_s_d, X_d = stats["down_proj"]
        W_d_np = np.array(mlp.down_proj.weight.astype(mx.float32))
        alpha_d = awq_search_alpha(W_d_np, act_s_d, X_d, group_size, sym, n_alpha)

        # q/k are never clipped: attention scores are a product of the two, and
        # clipping either changes them disproportionately (same rule as llm-awq).
        _quant(attn.q_proj, act_s_qkv, alpha_qkv)
        _quant(attn.k_proj, act_s_qkv, alpha_qkv)
        _quant(attn.v_proj, act_s_qkv, alpha_qkv, clip_X=X_qkv)
        _quant(attn.o_proj, act_s_o, alpha_o, clip_X=X_o)
        _quant(mlp.gate_proj, act_s_gu, alpha_gu, clip_X=X_gu)
        _quant(mlp.up_proj,   act_s_gu, alpha_gu, clip_X=X_gu)
        _quant(mlp.down_proj, act_s_d, alpha_d, clip_X=X_d)

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
