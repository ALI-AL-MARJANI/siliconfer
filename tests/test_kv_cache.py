"""Phase 9b unit tests: group-wise int8 quantized KV cache.

No model downloads for the core tests — quantize_kv/dequantize_kv and
QuantizedKVCache are tested directly with random mx.arrays, plus a tiny
synthetic LlamaModel for the end-to-end generation test.
"""

import numpy as np
import mlx.core as mx
import pytest

from siliconfer.model.kv_cache import (
    quantize_kv,
    dequantize_kv,
    QuantizedKVCache,
    make_quantized_cache,
)


# ---------------------------------------------------------------------------
# Core quantize/dequantize round trip
# ---------------------------------------------------------------------------

def test_quantize_kv_shapes():
    x = mx.random.normal((2, 4, 8, 64))
    q, scale = quantize_kv(x)
    assert q.shape == x.shape
    assert q.dtype == mx.int8
    assert scale.shape == (2, 4, 8, 1)
    assert scale.dtype == mx.float32


def test_quantize_kv_round_trip_low_error():
    """int8 group-wise quant should reconstruct values with small relative error."""
    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (1, 2, 16, 64)).astype(np.float32)
    x = mx.array(x_np)

    q, scale = quantize_kv(x)
    x_hat = dequantize_kv(q, scale, dtype=mx.float32)
    mx.eval(x_hat)

    err = np.abs(np.array(x_hat) - x_np)
    # int8 symmetric, 256 levels over the per-vector range — should be tight
    assert err.max() < 0.05 * np.abs(x_np).max()


def test_quantize_kv_zero_vector_safe():
    """An all-zero KV vector shouldn't divide by zero / produce NaN."""
    x = mx.zeros((1, 1, 1, 64))
    q, scale = quantize_kv(x)
    x_hat = dequantize_kv(q, scale)
    mx.eval(x_hat)
    assert np.isfinite(np.array(x_hat)).all()
    assert np.allclose(np.array(x_hat), 0.0)


# ---------------------------------------------------------------------------
# QuantizedKVCache: incremental update matches batch quantization
# ---------------------------------------------------------------------------

def test_cache_incremental_matches_batch_quantization():
    """Quantizing token-by-token (as decode does) must give bit-identical
    results to quantizing the whole sequence at once, since each token's
    vector is its own independent quantization group."""
    rng = np.random.default_rng(1)
    B, H, T, D = 1, 2, 5, 64
    k_np = rng.normal(0, 1, (B, H, T, D)).astype(np.float32)
    v_np = rng.normal(0, 1, (B, H, T, D)).astype(np.float32)

    # Batch: quantize the whole thing in one call
    q_batch, s_batch = quantize_kv(mx.array(k_np))

    # Incremental: one token at a time through the cache
    cache = QuantizedKVCache()
    for t in range(T):
        cache.update(mx.array(k_np[:, :, t:t+1, :]), mx.array(v_np[:, :, t:t+1, :]))

    mx.eval(cache.q_k, q_batch)
    np.testing.assert_array_equal(np.array(cache.q_k), np.array(q_batch))
    np.testing.assert_allclose(np.array(cache.s_k), np.array(s_batch), atol=1e-6)


def test_cache_length_and_trim():
    cache = QuantizedKVCache()
    assert cache.length() == 0

    k = mx.random.normal((1, 2, 3, 64))
    v = mx.random.normal((1, 2, 3, 64))
    cache.update(k, v)
    assert cache.length() == 3

    k2 = mx.random.normal((1, 2, 2, 64))
    v2 = mx.random.normal((1, 2, 2, 64))
    cache.update(k2, v2)
    assert cache.length() == 5

    cache.trim(4)
    assert cache.length() == 4


def test_make_quantized_cache():
    caches = make_quantized_cache(6)
    assert len(caches) == 6
    assert all(isinstance(c, QuantizedKVCache) for c in caches)
    assert all(c.length() == 0 for c in caches)


def test_cache_nbytes_roughly_half_of_fp16():
    """int8 codes + float32 scale should be roughly half the size of fp16
    storage for realistic head_dim (scale overhead is 4 bytes / head_dim
    values, i.e. small for head_dim >= 32)."""
    cache = QuantizedKVCache()
    B, H, T, D = 1, 2, 100, 64
    k = mx.random.normal((B, H, T, D))
    v = mx.random.normal((B, H, T, D))
    cache.update(k, v)

    fp16_equivalent_bytes = 2 * B * H * T * D * 2   # k+v, 2 bytes/value
    ratio = cache.nbytes() / fp16_equivalent_bytes
    print(f"\n  quantized/fp16 byte ratio: {ratio:.3f}")
    assert 0.4 < ratio < 0.7


# ---------------------------------------------------------------------------
# Attention forward with a quantized cache
# ---------------------------------------------------------------------------

def _make_tiny_config():
    from siliconfer.model.config import ModelConfig
    return ModelConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        tie_word_embeddings=True,
        hidden_act="silu",
    )


def test_attention_with_quantized_cache_runs_and_matches_fp16_closely():
    from siliconfer.model.llama import LlamaModel

    config = _make_tiny_config()
    model = LlamaModel(config)
    mx.eval(model.parameters())

    input_ids = mx.array([[1, 2, 3, 4, 5]])

    logits_fp, _ = model(input_ids, cache=None)
    mx.eval(logits_fp)

    quant_cache = make_quantized_cache(config.num_hidden_layers)
    logits_q, new_cache = model(input_ids, cache=quant_cache)
    mx.eval(logits_q)

    assert logits_q.shape == logits_fp.shape
    assert np.isfinite(np.array(logits_q)).all()
    assert all(isinstance(c, QuantizedKVCache) for c in new_cache)
    assert new_cache[0].length() == 5

    rel_err = np.abs(np.array(logits_q) - np.array(logits_fp)).max() / (
        np.abs(np.array(logits_fp)).max() + 1e-6
    )
    print(f"\n  quantized-cache vs fp16 max relative logit error: {rel_err:.4f}")
    assert rel_err < 0.2


def test_generate_with_quantized_kv_cache():
    from siliconfer.model.llama import LlamaModel
    from siliconfer.engine.generate import generate, SamplingParams

    config = _make_tiny_config()
    model = LlamaModel(config)
    mx.eval(model.parameters())

    params = SamplingParams(temperature=0.0, max_tokens=8)
    prompt_ids = mx.array([[1, 2, 3]])

    result = generate(model, prompt_ids, params=params, quantize_kv_cache=True)
    assert len(result.token_ids) == 8
    assert all(isinstance(t, int) for t in result.token_ids)


# ---------------------------------------------------------------------------
# Analytical memory footprint (Phase 9b)
# ---------------------------------------------------------------------------

def test_measure_kv_cache_memory():
    from siliconfer.eval.bench import measure_kv_cache_memory

    config = _make_tiny_config()   # head_dim = 128/4 = 32
    r = measure_kv_cache_memory(config, seq_len=1024)

    assert r["int8_mb"] < r["fp16_mb"]
    assert 1.0 < r["compression"] < 2.0   # int8 has scale overhead, never hits a full 2x

    # Compression should scale toward 2x as head_dim grows (scale overhead
    # amortizes over more values per vector)
    r_wide = measure_kv_cache_memory(config, seq_len=1024)
    config.head_dim = 256
    r_wider = measure_kv_cache_memory(config, seq_len=1024)
    assert r_wider["compression"] > r_wide["compression"]
