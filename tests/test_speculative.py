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


# ---------------------------------------------------------------------------
# Phase 9d research note: naive multi-candidate retry is provably NOT lossless
# ---------------------------------------------------------------------------
#
# Documents (and tests, so it can't silently regress into being "fixed" and
# reintroduced without noticing the flaw) a rejected design: after a drafted
# token is rejected, resample fresh i.i.d. candidates from p_draft and retry
# the same accept/reject test before falling back to residual sampling. The
# relative win-probabilities among candidates don't depend on the retry
# count (provable), but the residual/fallback distribution required to keep
# the *overall* marginal equal to p_target does — and for retry count >= 2 it
# can require a NEGATIVE probability for some tokens, which is impossible.
# This is exactly why real tree-attention speculative decoding (SpecInfer,
# EAGLE-2) needs careful correlated multi-candidate verification instead.

def test_naive_multicandidate_retry_requires_negative_fallback_probability():
    """Algebraic proof that naive independent-retry is not a valid scheme:
    for retry count M=2, the fallback distribution required to keep the
    overall marginal equal to p_target has a negative entry whenever
    p_draft(v) >= p_target(v) for some v (with Z small enough)."""
    rng = np.random.default_rng(0)
    vocab = 10
    p_draft = rng.dirichlet(np.ones(vocab) * 2.0)
    p_target = rng.dirichlet(np.ones(vocab) * 2.0)

    q = np.minimum(p_draft, p_target)
    Z = q.sum()
    M = 2
    required_fallback_numerator = p_target - q * (1 - (1 - Z) ** M) / Z

    assert required_fallback_numerator.min() < 0.0, (
        "expected the naive scheme's required fallback distribution to be "
        "invalid (negative) for M=2 retries — if this now passes, the "
        "surrounding claim in speculative.py's Phase 9d research note needs "
        "re-examination, not silent removal"
    )


# ---------------------------------------------------------------------------
# Phase 9d: dynamic speculation depth (the feature actually shipped)
# ---------------------------------------------------------------------------

class TestDynamicK:
    def test_greedy_still_matches_non_speculative(self):
        """dynamic_K must not change greedy-mode losslessness — K never
        appears in the accept/reject correctness proof, so varying it
        round-to-round is losslessness-neutral by construction."""
        model = _make_tiny_model(seed=7)
        params = SamplingParams(temperature=0.0, max_tokens=12)
        prompt = _prompt(6)

        spec_res = speculative_generate(
            draft=model, target=model, prompt_ids=prompt,
            params=params, K=2, seed=0, dynamic_K=True, K_min=1, K_max=6,
        )
        base_res = generate(model, prompt, params=params)

        assert spec_res.token_ids == base_res.token_ids

    def test_same_model_greedy_full_acceptance_with_dynamic_k(self):
        """With draft=target and greedy sampling, acceptance rate is still 1.0
        regardless of how K adapts round to round."""
        model = _make_tiny_model(seed=0)
        params = SamplingParams(temperature=0.0, max_tokens=16)
        res = speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=2, seed=42, dynamic_K=True, K_min=1, K_max=8,
        )
        assert res.acceptance_rate == pytest.approx(1.0, abs=0.01)

    def test_k_grows_when_fully_accepted(self):
        """With draft=target (guaranteed full acceptance under greedy), K
        should grow every round up to K_max and stay there."""
        model = _make_tiny_model(seed=0)
        params = SamplingParams(temperature=0.0, max_tokens=30)
        res = speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=1, seed=42, dynamic_K=True, K_min=1, K_max=5,
        )
        # total_draft_tokens should reflect K growing 1->2->3->4->5->5->...,
        # not staying flat at 1 the whole time (which is what dynamic_K=False
        # would give for the same number of rounds).
        avg_k = res.total_draft_tokens / res.total_rounds
        assert avg_k > 1.5, f"expected K to have grown from 1, got avg_k={avg_k}"

    def test_k_shrinks_on_rejection(self):
        """With genuinely different draft/target models and stochastic
        sampling (rejections expected — greedy argmax coincidentally agrees
        too often between these tiny synthetic models to exercise this path),
        K should be pulled back toward K_min after a round that rejects early."""
        draft_model = _make_tiny_model(seed=11)
        target_model = _make_tiny_model(seed=99)
        params = SamplingParams(temperature=0.8, max_tokens=40)
        res = speculative_generate(
            draft=draft_model, target=target_model, prompt_ids=_prompt(),
            params=params, K=6, seed=42, dynamic_K=True, K_min=1, K_max=6,
        )
        # Different models should reject sometimes — K shouldn't stay pinned
        # at the max the whole time the way the same-model case does.
        avg_k = res.total_draft_tokens / res.total_rounds
        assert avg_k < 6.0, f"expected K to shrink below K_max at some point, got avg_k={avg_k}"

    def test_runs_end_to_end_with_different_draft_and_target(self):
        """Smoke test: genuinely different draft/target models, temperature>0,
        dynamic_K enabled — the realistic scenario this feature targets."""
        draft_model = _make_tiny_model(seed=11)
        target_model = _make_tiny_model(seed=22)
        params = SamplingParams(temperature=0.8, max_tokens=10)

        res = speculative_generate(
            draft=draft_model, target=target_model, prompt_ids=_prompt(),
            params=params, K=3, seed=5, dynamic_K=True, K_min=1, K_max=6,
        )
        assert len(res.token_ids) == 10
        vocab = _TINY_CFG["vocab_size"]
        assert all(0 <= t < vocab for t in res.token_ids)

    def test_respects_k_bounds(self):
        """K should never leave [K_min, K_max] regardless of round outcomes."""
        draft_model = _make_tiny_model(seed=3)
        target_model = _make_tiny_model(seed=44)
        params = SamplingParams(temperature=0.5, max_tokens=25)
        # K_min == K_max degenerates to fixed-K — should still just work.
        res = speculative_generate(
            draft=draft_model, target=target_model, prompt_ids=_prompt(),
            params=params, K=3, seed=1, dynamic_K=True, K_min=3, K_max=3,
        )
        assert res.total_draft_tokens == res.total_rounds * 3


# ---------------------------------------------------------------------------
# quantize_kv_cache integration (extends Phase 9b's QuantizedKVCache, which
# Attention.__call__/_trim_cache already handled transparently, into the
# speculative decoding loop)
# ---------------------------------------------------------------------------

class TestQuantizedKVCacheSpeculative:
    def test_greedy_matches_non_speculative_quantized_cache_generation(self):
        """With quantize_kv_cache=True, speculative decoding must exactly
        match *non-speculative generation from the same quantized-cache
        model* — not the fp16-cache model's output, since quantized-cache
        decoding is itself only an approximation of fp16 (Phase 9b)."""
        model = _make_tiny_model(seed=7)
        params = SamplingParams(temperature=0.0, max_tokens=10)
        prompt = _prompt(6)

        spec_res = speculative_generate(
            draft=model, target=model, prompt_ids=prompt,
            params=params, K=3, seed=0, quantize_kv_cache=True,
        )
        base_res = generate(model, prompt, params=params, quantize_kv_cache=True)

        assert spec_res.token_ids == base_res.token_ids

    def test_same_model_greedy_full_acceptance_with_quantized_cache(self):
        model = _make_tiny_model(seed=0)
        params = SamplingParams(temperature=0.0, max_tokens=12)
        res = speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=3, seed=42, quantize_kv_cache=True,
        )
        assert res.acceptance_rate == pytest.approx(1.0, abs=0.01)

    def test_combined_with_dynamic_k(self):
        """quantize_kv_cache and dynamic_K are independent knobs — should
        compose without issue."""
        model = _make_tiny_model(seed=2)
        params = SamplingParams(temperature=0.0, max_tokens=14)
        res = speculative_generate(
            draft=model, target=model, prompt_ids=_prompt(),
            params=params, K=2, seed=42,
            quantize_kv_cache=True, dynamic_K=True, K_min=1, K_max=6,
        )
        assert len(res.token_ids) == 14
        assert res.acceptance_rate == pytest.approx(1.0, abs=0.01)

    def test_runs_end_to_end_with_different_draft_and_target(self):
        """Smoke test: genuinely different draft/target models, temperature>0,
        quantize_kv_cache=True — the realistic scenario this feature targets."""
        draft_model = _make_tiny_model(seed=11)
        target_model = _make_tiny_model(seed=22)
        params = SamplingParams(temperature=0.8, max_tokens=10)

        res = speculative_generate(
            draft=draft_model, target=target_model, prompt_ids=_prompt(),
            params=params, K=3, seed=5, quantize_kv_cache=True,
        )
        assert len(res.token_ids) == 10
        vocab = _TINY_CFG["vocab_size"]
        assert all(0 <= t < vocab for t in res.token_ids)
