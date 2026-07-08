"""Group-wise int8 quantized KV cache (Phase 9b).

Reference lineage: KIVI (arXiv:2402.02750), KVQuant — group-wise quantization
of the KV cache to curb its otherwise-unbounded memory growth with context
length. This is the same "int4 keeps quality ≈ fp16, cuts memory" thesis
already proven for weights in Phases 2-4, applied to the *cache* instead.

Design choices (why int8, not int4, and why native MLX ops):

- **int8, not int4.** Each key/value vector (length `head_dim`, typically
  64/128) is quantized as a *single group* — there's no natural sub-grouping
  to pack 2 int4 values per byte the way weight quantization does (that
  packing exploits contiguous *rows* of many groups; a KV vector only has
  one). int4 here would mean 4-bit codes stored one per byte anyway (no
  packing win) while roughly doubling quantization error over int8 for a
  much smaller cache that's already far more error-sensitive than weights
  (it feeds directly into softmax). int8 gives a real ~2x memory reduction
  (1 byte/value + a small per-vector float32 scale vs 2 bytes/value fp16)
  with much safer accuracy — matching the conservative default chosen for
  HQQ after the Phase 9a "super weight" incident. Packed int4 KV cache is a
  possible future extension once a fused kernel (Phase 9c) can dequantize it
  inline instead of materializing a full-precision copy every attention call.
- **Native `mx.array` ops throughout, no numpy round-trip.** Weight
  quantization (Phase 2-4) uses numpy, which is fine — it runs once, offline,
  before serving starts. The KV cache is quantized *on every decode step, for
  every layer*, so a numpy round-trip here would reproduce Phase 6's
  documented CPU<->GPU sync bottleneck, except worse (per-token, not
  per-model-load). Everything below — abs/max reduction, round, clip, the
  cache's growing concatenation — stays as `mx.array` ops, so the cache never
  leaves the compute graph.
- **Dequantize-on-read, not a fused kernel.** `mx.fast.scaled_dot_product_attention`
  has no quantized-KV variant (confirmed: MLX has no first-class primitive for
  this as of this writing). So each attention call dequantizes the *whole*
  accumulated cache back to float before running standard SDPA. This wastes
  some redundant dequant work on already-seen tokens (real compute-speed
  parity needs a fused kernel — see kernels/metal, Phase 9c) but is correct,
  simple, and already delivers the memory-footprint win, which is what this
  phase targets.
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
    """Per-layer KV cache stored as packed int8 codes + per-vector scales.

    Grows via `update()` exactly like the plain (k, v) tuple cache used
    elsewhere in the engine, but the *stored* representation between calls is
    the compressed one — the memory savings persist across the whole
    generation loop, not just inside a single attention call.
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
