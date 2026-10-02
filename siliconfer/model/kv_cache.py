"""int8-quantized KV cache.

Related work: KIVI (arXiv:2402.02750), KVQuant.

- int8 rather than int4. Each key/value vector (length head_dim) is one
  quantization group with one scale. There is no row of groups to pack
  nibbles along, and attention scores are more sensitive to error than
  weights are.
- MLX ops only. The cache is quantized on every decode step for every
  layer; a NumPy round trip would force a GPU↔CPU synchronisation per token.
- Dequantize on read. Each attention call dequantizes the accumulated cache
  and runs standard attention. That gives the memory saving; avoiding the
  repeated dequantization needs the fused kernel in kernels/metal.
"""

from __future__ import annotations

import mlx.core as mx

_INT8_MAX = 127
_INT8_MIN = -128


def quantize_kv(x: mx.array) -> tuple[mx.array, mx.array]:
    """Symmetric int8 quantization, one scale per (batch, head, token) vector.

    Args:
        x: float array, shape [B, n_kv_heads, T, head_dim].

    Returns:
        q: int8 array, same shape as x.
        scale: float32 array, shape [B, n_kv_heads, T, 1].
    """
    x32 = x.astype(mx.float32)
    abs_max = mx.max(mx.abs(x32), axis=-1, keepdims=True)
    scale = mx.where(abs_max == 0, mx.ones_like(abs_max), abs_max / _INT8_MAX)
    q = mx.clip(mx.round(x32 / scale), _INT8_MIN, _INT8_MAX).astype(mx.int8)
    return q, scale


def dequantize_kv(q: mx.array, scale: mx.array, dtype: mx.Dtype = mx.float16) -> mx.array:
    """Inverse of quantize_kv: w_approx = q * scale."""
    return (q.astype(mx.float32) * scale).astype(dtype)


class QuantizedKVCache:
    """Per-layer KV cache stored as int8 codes plus one scale per vector.

    Grows through `update()` like the plain (k, v) tuple cache; the stored
    representation between calls is the compressed one.
    """

    def __init__(self) -> None:
        self.q_k: mx.array | None = None
        self.s_k: mx.array | None = None
        self.q_v: mx.array | None = None
        self.s_v: mx.array | None = None

    def length(self) -> int:
        """Number of KV positions currently stored (0 if empty)."""
        return 0 if self.q_k is None else self.q_k.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize and append new (k, v), return the full dequantized cache."""
        qk, sk = quantize_kv(k)
        qv, sv = quantize_kv(v)

        if self.q_k is None:
            self.q_k, self.s_k = qk, sk
            self.q_v, self.s_v = qv, sv
        else:
            self.q_k = mx.concatenate([self.q_k, qk], axis=2)
            self.s_k = mx.concatenate([self.s_k, sk], axis=2)
            self.q_v = mx.concatenate([self.q_v, qv], axis=2)
            self.s_v = mx.concatenate([self.s_v, sv], axis=2)

        dtype = k.dtype
        return dequantize_kv(self.q_k, self.s_k, dtype), dequantize_kv(self.q_v, self.s_v, dtype)

    def trim(self, n: int) -> None:
        """Trim the cache to the first n positions in place."""
        self.q_k = self.q_k[:, :, :n, :]
        self.s_k = self.s_k[:, :, :n, :]
        self.q_v = self.q_v[:, :, :n, :]
        self.s_v = self.s_v[:, :, :n, :]

    def nbytes(self) -> int:
        """Packed footprint in bytes (codes + scales), for memory reporting."""
        if self.q_k is None:
            return 0
        return (
            self.q_k.nbytes + self.s_k.nbytes
            + self.q_v.nbytes + self.s_v.nbytes
        )


def make_quantized_cache(n_layers: int) -> list[QuantizedKVCache]:
    """One fresh QuantizedKVCache per transformer layer."""
    return [QuantizedKVCache() for _ in range(n_layers)]
