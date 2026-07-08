"""Core transformer layers: RMSNorm, RoPE, GQA Attention, SwiGLU MLP."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.config import ModelConfig
from siliconfer.model.kv_cache import QuantizedKVCache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _compute_rope_freqs(config: ModelConfig) -> mx.array:
    """Compute RoPE frequency tensor, handling Llama3-style scaling."""
    head_dim = config.head_dim
    theta = config.rope_theta
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))

    if config.rope_scaling and config.rope_scaling.type == "llama3":
        sc = config.rope_scaling
        factor = sc.factor
        low_freq_factor = sc.low_freq_factor
        high_freq_factor = sc.high_freq_factor
        old_context = sc.original_max_position_embeddings

        low_freq_wavelen = old_context / low_freq_factor
        high_freq_wavelen = old_context / high_freq_factor

        wavelens = 2.0 * math.pi / freqs
        new_freqs = []
        for i in range(freqs.size):
            wl = wavelens[i].item()
            f = freqs[i].item()
            if wl > low_freq_wavelen:
                new_freqs.append(f / factor)
            elif wl < high_freq_wavelen:
                new_freqs.append(f)
            else:
                smooth = (old_context / wl - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * f / factor + smooth * f)
        freqs = mx.array(new_freqs, dtype=mx.float32)

    return freqs


def apply_rope(x: mx.array, offset: int, freqs: mx.array) -> mx.array:
    T = x.shape[2]
    positions = mx.arange(offset, offset + T, dtype=mx.float32)
    angles = mx.outer(positions, freqs)
    cos_vals = mx.cos(angles)[None, None, :, :]
    sin_vals = mx.sin(angles)[None, None, :, :]

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * mx.concatenate([cos_vals, cos_vals], axis=-1) + rotated * mx.concatenate([sin_vals, sin_vals], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, rope_freqs: mx.array):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.rope_freqs = rope_freqs

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | QuantizedKVCache | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | QuantizedKVCache]:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        quantized_cache = isinstance(cache, QuantizedKVCache)

        if quantized_cache:
            offset = cache.length()
        elif cache is not None:
            prev_k, prev_v = cache
            offset = prev_k.shape[2]
        else:
            offset = 0

        q = apply_rope(q, offset, self.rope_freqs)
        k = apply_rope(k, offset, self.rope_freqs)

        if quantized_cache:
            k, v = cache.update(k, v)
            new_cache = cache
        elif cache is not None:
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)
            new_cache = (k, v)
        else:
            new_cache = (k, v)

        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(q.dtype)
            if offset > 0:
                prefix = mx.zeros((T, offset), dtype=q.dtype)
                mask = mx.concatenate([prefix, mask], axis=1)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out), new_cache


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, rope_freqs: mx.array):
        super().__init__()
        self.self_attn = Attention(config, rope_freqs)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | QuantizedKVCache | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | QuantizedKVCache]:
        h, new_cache = self.self_attn(self.input_layernorm(x), cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache
