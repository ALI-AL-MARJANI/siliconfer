"""Phase 8 tests: mixed precision + speculative decoding.

All tests use the same tiny synthetic LlamaModel as test_integration.py
(hidden=64, 2 layers, vocab=256) — no model download needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel
from siliconfer.model.q4_linear import Q4Linear
from siliconfer.engine.q4_loader import _pack_and_replace_linears
from siliconfer.engine.generate import generate, SamplingParams
from siliconfer.engine.speculative import speculative_generate, SpeculativeResult


# ---------------------------------------------------------------------------
# Shared fixture: tiny synthetic model
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
    cfg = ModelConfig(**_TINY_CFG)
    model = LlamaModel(cfg)
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return mx.array(rng.normal(0, 0.02, shape).astype(np.float32))

    for layer in model.layers:
        a = layer.self_attn
        m = layer.mlp
        a.q_proj.weight = rand(64, 64)
        a.k_proj.weight = rand(32, 64)
        a.v_proj.weight = rand(32, 64)
        a.o_proj.weight = rand(64, 64)
        m.gate_proj.weight = rand(128, 64)
        m.up_proj.weight   = rand(128, 64)
        m.down_proj.weight = rand(64, 128)
        layer.input_layernorm.weight = rand(64)
        layer.post_attention_layernorm.weight = rand(64)

    model.embed_tokens.weight = rand(256, 64)
    mx.eval(model.parameters())
    return model


def _prompt(n: int = 8) -> mx.array:
    return mx.array([[i + 1 for i in range(n)]])


# ---------------------------------------------------------------------------
# Mixed-precision tests
# ---------------------------------------------------------------------------

class TestMixedPrecision:
    def test_skip_layer_stays_fp16(self):
        """Skipped layers keep their original nn.Linear weights."""
        import mlx.nn as nn
        model = _make_tiny_model()
        _pack_and_replace_linears(model, group_size=_GROUP_SIZE, skip_layers={0})
        # Layer 0 should still be nn.Linear (not Q4Linear)
        assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
        # Layer 1 should be quantized
        assert isinstance(model.layers[1].self_attn.q_proj, Q4Linear)

    def test_all_layers_skipped_leaves_all_fp16(self):
        """Skipping all layers results in no Q4Linear replacements."""
        import mlx.nn as nn
        model = _make_tiny_model()
        n = len(model.layers)
        _pack_and_replace_linears(model, group_size=_GROUP_SIZE, skip_layers=set(range(n)))
        for layer in model.layers:
            assert isinstance(layer.self_attn.q_proj, nn.Linear)
            assert isinstance(layer.mlp.gate_proj, nn.Linear)

    def test_skip_none_quantizes_all(self):
        """Default (skip_layers=None) quantizes every eligible layer."""
        model = _make_tiny_model()
        _pack_and_replace_linears(model, group_size=_GROUP_SIZE, skip_layers=None)
        for layer in model.layers:
            assert isinstance(layer.self_attn.q_proj, Q4Linear)

    def test_mixed_model_forward_runs(self):
        """A model with mixed fp16+Q4 layers can run a forward pass."""
        model = _make_tiny_model()
        _pack_and_replace_linears(model, group_size=_GROUP_SIZE, skip_layers={0})
        logits, _ = model(_prompt())
        mx.eval(logits)
        assert logits.shape == (1, 8, 256)

    def test_mixed_model_generates_tokens(self):
        """generate() works with a mixed-precision model."""
        model = _make_tiny_model()
        _pack_and_replace_linears(model, group_size=_GROUP_SIZE, skip_layers={0})
        params = SamplingParams(temperature=0.0, max_tokens=4)
        result = generate(model, _prompt(), params=params)
        assert len(result.token_ids) == 4


# ---------------------------------------------------------------------------
# Speculative decoding tests
# ---------------------------------------------------------------------------

class TestSpeculativeDecoding:
    def _same_model_spec(self, K: int, max_tokens: int, seed: int = 42):
        """Run speculative_generate with draft = target (same model instance).

        When draft == target, every draft token is produced by the same distribution
        as the target. With greedy (temp=0), argmax always matches → acceptance = 1.0.
        """
        model = _make_tiny_model(seed=0)
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        return speculative_generate(
            draft=model, target=model,
            prompt_ids=_prompt(),
            params=params, K=K, seed=seed,
        )

    def test_returns_speculative_result(self):
        res = self._same_model_spec(K=2, max_tokens=6)
        assert isinstance(res, SpeculativeResult)

    def test_token_count(self):
        """Generated token count equals max_tokens."""
        max_t = 8
        res = self._same_model_spec(K=3, max_tokens=max_t)
        assert len(res.token_ids) == max_t

    def test_same_model_greedy_full_acceptance(self):
        """With draft = target, greedy sampling → acceptance rate = 1.0."""
        res = self._same_model_spec(K=4, max_tokens=12)
        assert res.acceptance_rate == pytest.approx(1.0, abs=0.01)

    def test_same_model_matches_non_speculative(self):
        """Speculative with draft=target and greedy must match non-speculative output."""
        model = _make_tiny_model(seed=7)
        params = SamplingParams(temperature=0.0, max_tokens=10)
        prompt = _prompt(6)

        spec_res = speculative_generate(
            draft=model, target=model, prompt_ids=prompt,
            params=params, K=3, seed=0,
        )
        base_res = generate(model, prompt, params=params)

        assert spec_res.token_ids == base_res.token_ids, (
            f"Speculative: {spec_res.token_ids}\nNon-speculative: {base_res.token_ids}"
        )

    def test_different_K_values(self):
        """speculative_generate works for K in {1, 2, 4, 8}."""
        model = _make_tiny_model(seed=1)
        params = SamplingParams(temperature=0.0, max_tokens=6)
        for K in (1, 2, 4, 8):
            res = speculative_generate(
                draft=model, target=model, prompt_ids=_prompt(),
                params=params, K=K, seed=0,
            )
            assert len(res.token_ids) == 6, f"K={K} gave wrong token count"

    def test_eos_stops_generation(self):
        """Generation stops when eos_token_id is produced."""
        model = _make_tiny_model(seed=3)
        params = SamplingParams(temperature=0.0, max_tokens=50)
        # Use 1 as eos — model will produce some token including possibly 1
        res = speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=2, eos_token_id=1, seed=0,
        )
        # Either hit eos (shorter) or produced 50 tokens
        assert len(res.token_ids) <= 50

    def test_on_token_callback(self):
        """on_token callback is called for every generated token."""
        model = _make_tiny_model(seed=5)
        params = SamplingParams(temperature=0.0, max_tokens=8)
        received: list[int] = []
        speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=3,
            on_token=received.append, seed=0,
        )
        assert len(received) == 8

    def test_statistics_fields(self):
        """SpeculativeResult statistics fields are non-negative and consistent."""
        res = self._same_model_spec(K=3, max_tokens=9)
        assert res.total_rounds >= 1
        assert res.total_draft_tokens == res.total_rounds * 3
        assert 0.0 <= res.acceptance_rate <= 1.0
        assert res.effective_tok_s > 0.0
        assert res.prefill_tok_s > 0.0

    def test_temperature_sampling(self):
        """Temperature > 0 still produces valid token ids."""
        model = _make_tiny_model(seed=9)
        params = SamplingParams(temperature=1.0, max_tokens=8)
        res = speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=2, seed=42,
        )
        assert len(res.token_ids) == 8
        vocab = _TINY_CFG["vocab_size"]
        assert all(0 <= t < vocab for t in res.token_ids)
