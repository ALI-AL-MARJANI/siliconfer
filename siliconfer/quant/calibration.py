"""Calibration data loading and per-layer Hessian collection for GPTQ.

H = 2 X Xᵀ where X is [in_features, n_tokens] (columns = token activations).
We collect X by running calibration sequences through the model and intercepting
the input at each linear projection.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.layers import apply_rope


# ---------------------------------------------------------------------------
# Calibration sequence loading
# ---------------------------------------------------------------------------

def load_calibration_sequences(
    tokenizer_id: str,
    n_seqs: int = 128,
    seq_len: int = 512,
    seed: int = 42,
) -> list[mx.array]:
    """Load n_seqs random chunks of seq_len tokens from WikiText-2 train split.

    Returns a list of mx.array of shape (1, seq_len) with dtype int32.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    all_ids = tokenizer.encode(text)

    rng = np.random.default_rng(seed)
    max_start = len(all_ids) - seq_len - 1
    starts = rng.integers(0, max_start, size=n_seqs)

    sequences: list[mx.array] = []
    for s in starts:
        chunk = np.array(all_ids[s : s + seq_len], dtype=np.int32)
        sequences.append(mx.array(chunk[None, :]))   # (1, seq_len)

    return sequences


# ---------------------------------------------------------------------------
# H collection: manual forward interception (no monkey-patching)
# ---------------------------------------------------------------------------

def _np32(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.array(x.astype(mx.float32))


def _add_to(store: dict[str, list], key: str, x: mx.array) -> None:
    """Flatten [B, T, D] → [B*T, D] in float32 and append to store."""
    flat = _np32(x.reshape(-1, x.shape[-1]))
    store[key].append(flat)


def collect_layer_H(
    layer,                             # TransformerBlock
    hidden_states: list[mx.array],    # list of (1, T, hidden) tensors
) -> dict[str, np.ndarray]:
    """Compute H = 2 X^T X for all 7 linear projections in one transformer layer.

    We manually decompose the forward pass so we can capture each projection's
    input without modifying the model structure.

    Returns a dict: projection_name → H of shape [in_features, in_features] (float64).
    q_proj / k_proj / v_proj share the same H (identical input after input_layernorm).
    gate_proj / up_proj share the same H.
    """
    attn = layer.self_attn
    mlp  = layer.mlp

    # Collect raw input arrays per projection group
    raw: dict[str, list[np.ndarray]] = {
        "qkv":      [],   # input to q / k / v projections
        "o":        [],   # input to o_proj (= attention output, pre-projection)
        "gate_up":  [],   # input to gate / up projections
        "down":     [],   # input to down_proj
    }

    for x in hidden_states:
        B, T, _ = x.shape

        # --- Q / K / V inputs ---
        x_norm = layer.input_layernorm(x)          # (1, T, H)
        _add_to(raw, "qkv", x_norm)

        # --- attention forward to capture o_proj input ---
        q = attn.q_proj(x_norm).reshape(B, T, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = attn.k_proj(x_norm).reshape(B, T, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = attn.v_proj(x_norm).reshape(B, T, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

        q = apply_rope(q, 0, attn.rope_freqs)
        k = apply_rope(k, 0, attn.rope_freqs)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(q.dtype) if T > 1 else None
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)   # (1, T, num_heads * head_dim)
        _add_to(raw, "o", attn_out)

        # --- gate / up inputs ---
        h = attn.o_proj(attn_out)
        x_post_attn = x + h
        x_post_norm = layer.post_attention_layernorm(x_post_attn)
        _add_to(raw, "gate_up", x_post_norm)

        # --- down_proj input ---
        down_in = nn.silu(mlp.gate_proj(x_post_norm)) * mlp.up_proj(x_post_norm)
        _add_to(raw, "down", down_in)

    def _H(inp_list: list[np.ndarray]) -> np.ndarray:
        X = np.concatenate(inp_list, axis=0)   # [total_tokens, in_features]
        return (2.0 * X.T @ X).astype(np.float64)

    H_qkv     = _H(raw["qkv"])
    H_o       = _H(raw["o"])
    H_gate_up = _H(raw["gate_up"])
    H_down    = _H(raw["down"])

    return {
        "q_proj":    H_qkv,
        "k_proj":    H_qkv,       # same input as q_proj
        "v_proj":    H_qkv,       # same input as q_proj
        "o_proj":    H_o,
        "gate_proj": H_gate_up,
        "up_proj":   H_gate_up,   # same input as gate_proj
        "down_proj": H_down,
    }
