"""Phase 9c/9f unit tests: fused quantized-KV attention Metal kernel.

Verified against the reference "dequant int8 cache, then call
mx.fast.scaled_dot_product_attention" path (the same one Phase 9b's
QuantizedKVCache uses internally). As of the v3 tiled kernel (Phase 9f) this
*does* beat native SDPA at long context (measured 1.03-1.13x from T=4096 to
16384 — see CLAUDE.md/NOTES.md for the full v1/v2/v3 story and numbers), but
these tests check correctness only, not throughput — performance is measured
separately in the ad-hoc benchmark script referenced in the module docstring,
since micro-benchmark timing in a unit test is flaky by nature.
"""

import numpy as np
import mlx.core as mx
import pytest

from siliconfer.kernels.metal.q4_attention import fused_quantized_attention_decode
from siliconfer.model.kv_cache import quantize_kv, dequantize_kv


def _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups):
    """dequant-then-native-SDPA reference, matching QuantizedKVCache's own path."""
    B, n_heads, head_dim = q.shape
    k_deq = dequantize_kv(k_codes, k_scales, dtype=mx.float32)
    v_deq = dequantize_kv(v_codes, v_scales, dtype=mx.float32)
    k_rep = mx.repeat(k_deq, gqa_groups, axis=1)
    v_rep = mx.repeat(v_deq, gqa_groups, axis=1)
    q4 = q.reshape(B, n_heads, 1, head_dim)
    out = mx.fast.scaled_dot_product_attention(
        q4, k_rep, v_rep, scale=1.0 / np.sqrt(head_dim), mask=None
    )
    return out.reshape(B, n_heads, head_dim)


def _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T):
    q = mx.array(rng.normal(0, 1, (B, n_heads, head_dim)).astype(np.float32))
    k = mx.array(rng.normal(0, 1, (B, n_kv_heads, T, head_dim)).astype(np.float32))
    v = mx.array(rng.normal(0, 1, (B, n_kv_heads, T, head_dim)).astype(np.float32))
    k_codes, k_scales = quantize_kv(k)
    v_codes, v_scales = quantize_kv(v)
    mx.eval(k_codes, k_scales, v_codes, v_scales)
    return q, k_codes, k_scales, v_codes, v_scales


@pytest.mark.parametrize("T", [1, 3, 5, 128])
def test_matches_reference_mha(T):
    """No GQA (n_heads == n_kv_heads): kernel output should match reference
    to float32 precision."""
    rng = np.random.default_rng(0)
    B, n_heads, n_kv_heads, head_dim = 1, 4, 4, 8
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)

    out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales)
    ref = _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups=1)
    mx.eval(out, ref)

    diff = np.abs(np.array(out) - np.array(ref)).max()
    assert diff < 1e-4, f"T={T}: max diff {diff}"


@pytest.mark.parametrize("T", [1, 3, 128, 512])
def test_matches_reference_gqa(T):
    """GQA case (Qwen2.5-0.5B-style: 14 q heads, 2 kv heads)."""
    rng = np.random.default_rng(3)
    B, n_heads, n_kv_heads, head_dim = 1, 14, 2, 64
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)

    out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales)
    ref = _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups=n_heads // n_kv_heads)
    mx.eval(out, ref)

    diff = np.abs(np.array(out) - np.array(ref)).max()
    assert diff < 1e-4, f"T={T}: max diff {diff}"


def test_output_finite_and_shape():
    rng = np.random.default_rng(5)
    B, n_heads, n_kv_heads, head_dim, T = 2, 8, 4, 32, 64
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)

    out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales)
    mx.eval(out)
    assert out.shape == (B, n_heads, head_dim)
    assert np.isfinite(np.array(out)).all()


def test_no_race_condition_across_repeated_calls():
    """Threadgroup-parallel reduction with barriers — run many times to catch
    any nondeterministic race condition (none should be observed: every
    thread's memory access pattern is either fully partitioned by `d` or
    guarded by a barrier before the next read)."""
    rng = np.random.default_rng(7)
    B, n_heads, n_kv_heads, head_dim, T = 1, 14, 2, 64, 256
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)
    ref = _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups=n_heads // n_kv_heads)
    mx.eval(ref)
    ref_np = np.array(ref)

    for _ in range(10):
        out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales)
        mx.eval(out)
        diff = np.abs(np.array(out) - ref_np).max()
        assert diff < 1e-4, f"Nondeterministic result detected: max diff {diff}"


def test_batch_dimension():
    """B > 1 should route each batch element to its own threadgroups."""
    rng = np.random.default_rng(9)
    B, n_heads, n_kv_heads, head_dim, T = 3, 4, 2, 16, 32
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)

    out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales)
    ref = _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups=n_heads // n_kv_heads)
    mx.eval(out, ref)

    diff = np.abs(np.array(out) - np.array(ref)).max()
    assert diff < 1e-4


# ---------------------------------------------------------------------------
# Phase 9f: explicit n_tiles — the tiled-merge mechanism itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tiles", [1, 2, 3, 7, 16, 64, 200])
def test_explicit_n_tiles_matches_reference(n_tiles):
    """The flash-attention-style partial-result merge must be correct for
    any tile count, including values that don't divide T_cache evenly
    (forcing a smaller last tile) and tile counts larger than T_cache
    (forcing many single-timestep tiles)."""
    rng = np.random.default_rng(11)
    B, n_heads, n_kv_heads, head_dim, T = 1, 14, 2, 64, 100
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)

    out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales, n_tiles=n_tiles)
    ref = _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups=n_heads // n_kv_heads)
    mx.eval(out, ref)

    diff = np.abs(np.array(out) - np.array(ref)).max()
    assert diff < 1e-4, f"n_tiles={n_tiles}: max diff {diff}"


def test_different_n_tiles_agree_with_each_other():
    """Independent of the reference path: n_tiles=1 (old v2-equivalent
    behavior) and a heavily-tiled run on the *same* inputs must produce the
    same result, since they're computing the identical mathematical
    quantity — this isolates the merge math from any reference-path
    assumptions."""
    rng = np.random.default_rng(13)
    B, n_heads, n_kv_heads, head_dim, T = 1, 14, 2, 64, 300
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)

    out_1tile = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales, n_tiles=1)
    out_many = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales, n_tiles=37)
    mx.eval(out_1tile, out_many)

    diff = np.abs(np.array(out_1tile) - np.array(out_many)).max()
    assert diff < 1e-4


def test_default_heuristic_picks_more_tiles_for_longer_context():
    from siliconfer.kernels.metal.q4_attention import _choose_n_tiles
    assert _choose_n_tiles(8) == 1
    assert _choose_n_tiles(128) > _choose_n_tiles(32) > _choose_n_tiles(8)
    # capped, not unbounded
    assert _choose_n_tiles(1_000_000) <= 256


def test_no_race_condition_with_many_tiles():
    """Same race-condition guard as before, but now specifically stressing
    the higher threadgroup counts the tiled kernel launches (this is the
    axis that changed in Phase 9f — worth its own dedicated repeat-check)."""
    rng = np.random.default_rng(17)
    B, n_heads, n_kv_heads, head_dim, T = 1, 14, 2, 64, 2048
    q, k_codes, k_scales, v_codes, v_scales = _random_inputs(rng, B, n_heads, n_kv_heads, head_dim, T)
    ref = _reference_attention(q, k_codes, k_scales, v_codes, v_scales, gqa_groups=n_heads // n_kv_heads)
    mx.eval(ref)
    ref_np = np.array(ref)

    for _ in range(10):
        out = fused_quantized_attention_decode(q, k_codes, k_scales, v_codes, v_scales)
        mx.eval(out)
        diff = np.abs(np.array(out) - ref_np).max()
        assert diff < 1e-4, f"Nondeterministic result detected: max diff {diff}"
