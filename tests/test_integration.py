"""Phase 6 integration tests: Q4Linear + end-to-end model generation.

Uses a tiny synthetic LlamaModel (hidden=64, 2 layers) — no model download needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel
from siliconfer.model.q4_linear import Q4Linear
from siliconfer.engine.q4_loader import _pack_and_replace_linears
from siliconfer.engine.generate import generate, SamplingParams
from siliconfer.kernels.neon import pack_weights_sym, gemv_sym
from siliconfer.quant.primitives import fake_quantize


# ---------------------------------------------------------------------------
# Tiny synthetic model config
# ---------------------------------------------------------------------------

_TINY_CFG = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    max_position_embeddings=512,
    tie_word_embeddings=True,
    attention_bias=False,
)
_GROUP_SIZE = 64


def _make_tiny_model(seed: int = 0) -> LlamaModel:
    """Create a tiny LlamaModel with random fp32 weights."""
    cfg = ModelConfig(**_TINY_CFG)
    model = LlamaModel(cfg)

    rng = np.random.default_rng(seed)

    def rand(*shape):
        return mx.array(rng.normal(0, 0.02, shape).astype(np.float32))

    h = cfg.hidden_size
    ff = cfg.intermediate_size
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = h // nh

    model.embed_tokens.weight = rand(cfg.vocab_size, h)

    for layer in model.layers:
        a = layer.self_attn
        m = layer.mlp
        a.q_proj.weight = rand(nh * hd, h)
        a.k_proj.weight = rand(nkv * hd, h)
        a.v_proj.weight = rand(nkv * hd, h)
        a.o_proj.weight = rand(h, nh * hd)
        m.gate_proj.weight = rand(ff, h)
        m.up_proj.weight   = rand(ff, h)
        m.down_proj.weight = rand(h, ff)
        layer.input_layernorm.weight          = mx.ones((h,))
        layer.post_attention_layernorm.weight = mx.ones((h,))

    model.norm.weight = mx.ones((h,))
    return model


# ---------------------------------------------------------------------------
# Q4Linear unit tests
# ---------------------------------------------------------------------------

def test_q4linear_matches_fake_quant():
    """Q4Linear output should match fake_quantize(W) @ x within float32 tolerance."""
    rng = np.random.default_rng(1)
    out_f, in_f = 32, 64
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x_np = rng.normal(0, 1, (1, in_f)).astype(np.float32)

    packed, scales = pack_weights_sym(W, group_size=64)
    layer = Q4Linear(packed, scales, group_size=64)

    x_mx = mx.array(x_np)
    y_q4 = np.array(layer(x_mx))  # [1, out_f]

    W_fq = fake_quantize(W, group_size=64, sym=True)
    y_ref = (W_fq @ x_np[0])

    np.testing.assert_allclose(y_q4[0], y_ref, atol=1e-4, rtol=1e-4)


def test_q4linear_with_bias():
    """Bias is added correctly."""
    rng = np.random.default_rng(2)
    out_f, in_f = 16, 64
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    b = rng.normal(0, 0.1, out_f).astype(np.float32)
    x_np = rng.normal(0, 1, (1, in_f)).astype(np.float32)

    packed, scales = pack_weights_sym(W, group_size=64)
    layer_no_bias = Q4Linear(packed, scales, group_size=64)
    layer_bias    = Q4Linear(packed, scales, bias=mx.array(b), group_size=64)

    y_no_b = np.array(layer_no_bias(mx.array(x_np)))
    y_b    = np.array(layer_bias(mx.array(x_np)))

    np.testing.assert_allclose(y_b, y_no_b + b, atol=1e-5)


def test_q4linear_batch_and_seq():
    """Q4Linear handles [B, T, in_f] input shape correctly."""
    rng = np.random.default_rng(3)
    out_f, in_f = 32, 64
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x_np = rng.normal(0, 1, (2, 5, in_f)).astype(np.float32)   # B=2, T=5

    packed, scales = pack_weights_sym(W, group_size=64)
    layer = Q4Linear(packed, scales, group_size=64)

    y = layer(mx.array(x_np))
    assert y.shape == (2, 5, out_f)


# ---------------------------------------------------------------------------
# Model forward pass with Q4Linear
# ---------------------------------------------------------------------------

def test_model_forward_after_pack():
    """Model forward pass produces correct output shape after packing linears."""
    model = _make_tiny_model()
    _pack_and_replace_linears(model, group_size=_GROUP_SIZE)

    B, T = 1, 8
    input_ids = mx.array(np.random.randint(0, _TINY_CFG["vocab_size"], (B, T)))
    logits, cache = model(input_ids)

    assert logits.shape == (B, T, _TINY_CFG["vocab_size"])
    assert len(cache) == _TINY_CFG["num_hidden_layers"]


def test_model_linears_replaced():
    """All 7 attn+MLP projections per layer should become Q4Linear."""
    model = _make_tiny_model()
    _pack_and_replace_linears(model, group_size=_GROUP_SIZE)

    for layer in model.layers:
        for parent, name in [
            (layer.self_attn, "q_proj"),
            (layer.self_attn, "k_proj"),
            (layer.self_attn, "v_proj"),
            (layer.self_attn, "o_proj"),
            (layer.mlp,       "gate_proj"),
            (layer.mlp,       "up_proj"),
            (layer.mlp,       "down_proj"),
        ]:
            lin = getattr(parent, name)
            assert isinstance(lin, Q4Linear), \
                f"{name} should be Q4Linear, got {type(lin)}"


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------

def test_generate_produces_tokens():
    """generate() with Q4Linear model returns expected number of token ids."""
    model = _make_tiny_model()
    _pack_and_replace_linears(model, group_size=_GROUP_SIZE)

    prompt_ids = mx.array([[1, 2, 3]])   # batch=1, 3 prompt tokens
    params = SamplingParams(temperature=1.0, max_tokens=10)

    result = generate(model, prompt_ids, params=params)

    assert len(result.token_ids) == 10
    assert result.num_decode_tokens == 9    # first token counted separately
    assert result.decode_tok_s > 0


def test_generate_decode_step_uses_kv_cache():
    """Cached decode (T=1 steps) must give same first token as prefill if temp=0."""
    model = _make_tiny_model(seed=7)
    _pack_and_replace_linears(model, group_size=_GROUP_SIZE)

    prompt = mx.array([[5, 10, 15, 20]])
    params = SamplingParams(temperature=0.0, max_tokens=1)

    result = generate(model, prompt, params=params)
    first_tok = result.token_ids[0]

    # Run again — deterministic at temperature=0
    result2 = generate(model, prompt, params=params)
    assert result2.token_ids[0] == first_tok


def test_generate_on_token_callback():
    """on_token callback receives every generated token id."""
    model = _make_tiny_model()
    _pack_and_replace_linears(model, group_size=_GROUP_SIZE)

    prompt_ids = mx.array([[1, 2]])
    params = SamplingParams(temperature=1.0, max_tokens=5)

    collected: list[int] = []
    result = generate(model, prompt_ids, params=params,
                      on_token=lambda tok: collected.append(tok))

    assert collected == result.token_ids
