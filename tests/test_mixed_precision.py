"""Tests for mixed 2-bit/4-bit precision quantization (quant/mixed_precision.py).

The Shapley estimator tests use synthetic value functions with a CLOSED-FORM
ground truth (additive and pairwise-interaction games), not just "does it run"
checks — the whole point of using Shapley values is that they're an exact,
well-defined quantity, so the estimator should be checked against known exact
answers wherever possible, the same way SINQ's math was checked algebraically
before ever touching a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from siliconfer.quant.mixed_precision import (
    shapley_layer_sensitivity,
    assign_bitwidths,
)


# ---------------------------------------------------------------------------
# Shapley estimator: closed-form ground truth
# ---------------------------------------------------------------------------

def test_shapley_additive_value_function_exact():
    """For a purely additive game v(S) = sum_{i in S} c_i (no interaction
    between players), the marginal contribution of adding player i is
    EXACTLY c_i in every permutation, with zero variance — so even a single
    permutation should recover c_i essentially exactly."""
    c = np.array([1.0, -2.5, 3.0, 0.5, -1.0])
    n = len(c)

    def value_fn(S: frozenset[int]) -> float:
        return float(sum(c[i] for i in S))

    shapley = shapley_layer_sensitivity(value_fn, n_layers=n, n_permutations=3, seed=0)
    np.testing.assert_allclose(shapley, c, atol=1e-9)


def test_shapley_matches_closed_form_pairwise_game():
    """For a 2-additive game v(S) = sum_i c_i*[i in S] + sum_{i<j} d_ij*[i,j in S],
    the exact Shapley value is phi_i = c_i + sum_{j != i} d_ij / 2 (a standard
    result: each pairwise-interaction unanimity term splits equally between
    its two members). Verify the Monte Carlo estimator converges to this
    closed form."""
    rng = np.random.default_rng(0)
    n = 5
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, (n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)

    def value_fn(S: frozenset[int]) -> float:
        idx = sorted(S)
        total = sum(c[i] for i in idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                total += d[idx[a], idx[b]]
        return float(total)

    expected = c + d.sum(axis=1) / 2.0

    shapley = shapley_layer_sensitivity(value_fn, n_layers=n, n_permutations=3000, seed=1)
    np.testing.assert_allclose(shapley, expected, atol=0.03)


def test_shapley_symmetric_players_get_equal_value():
    """Interchangeable players (value function invariant to swapping them)
    must receive identical Shapley values — a basic symmetry axiom Shapley
    values are required to satisfy."""
    def value_fn(S: frozenset[int]) -> float:
        return float(len(S) ** 2)  # symmetric in all players

    shapley = shapley_layer_sensitivity(value_fn, n_layers=4, n_permutations=20000, seed=2)
    assert np.allclose(shapley, shapley[0], atol=0.1)


def test_shapley_efficiency_axiom():
    """Shapley values must sum to v(full coalition) - v(empty) (the
    'efficiency' axiom) — a direct algebraic identity, not an approximation."""
    rng = np.random.default_rng(3)
    n = 5
    c = rng.normal(0, 2, n)

    def value_fn(S: frozenset[int]) -> float:
        return float(sum(c[i] for i in S))

    shapley = shapley_layer_sensitivity(value_fn, n_layers=n, n_permutations=5, seed=4)
    assert abs(shapley.sum() - c.sum()) < 1e-9


def test_shapley_zero_layers():
    shapley = shapley_layer_sensitivity(lambda S: 0.0, n_layers=0, n_permutations=5)
    assert shapley.shape == (0,)


# ---------------------------------------------------------------------------
# Bit-width assignment
# ---------------------------------------------------------------------------

def test_assign_bitwidths_no_demotion_when_budget_sufficient():
    sensitivity = np.array([3.0, 1.0, 2.0, 0.5])
    bytes_high = np.full(4, 1000.0)
    bits = assign_bitwidths(sensitivity, bytes_high, memory_budget_bytes=4000.0)
    assert (bits == 4).all()


def test_assign_bitwidths_optimal_for_uniform_sizes():
    """With uniform layer sizes, demoting the k least-sensitive layers is the
    exact optimum (see module docstring) — verify the k actually demoted are
    exactly the k lowest-sensitivity layers, not just "some k of them"."""
    rng = np.random.default_rng(5)
    n = 10
    sensitivity = rng.normal(0, 1, n)
    bytes_high = np.full(n, 100.0)  # uniform size across all layers

    k_demote = 4
    # Budget for exactly (n - k_demote) layers at high_bits + k_demote at low_bits
    bytes_low = 100.0 * (2 / 4)
    budget = (n - k_demote) * 100.0 + k_demote * bytes_low

    bits = assign_bitwidths(sensitivity, bytes_high, memory_budget_bytes=budget)

    expected_demoted = set(np.argsort(sensitivity)[:k_demote].tolist())
    actual_demoted = set(np.where(bits == 2)[0].tolist())
    assert actual_demoted == expected_demoted
    assert (bits[list(expected_demoted)] == 2).all()


def test_assign_bitwidths_respects_budget():
    rng = np.random.default_rng(6)
    n = 12
    sensitivity = rng.normal(0, 1, n)
    bytes_high = rng.uniform(50, 150, n)
    budget = bytes_high.sum() * 0.6

    bits = assign_bitwidths(sensitivity, bytes_high, memory_budget_bytes=budget)
    bytes_low = bytes_high * (2 / 4)
    total_used = np.where(bits == 4, bytes_high, bytes_low).sum()
    assert total_used <= budget + 1e-6


def test_assign_bitwidths_only_two_values():
    rng = np.random.default_rng(7)
    n = 8
    sensitivity = rng.normal(0, 1, n)
    bytes_high = np.full(n, 100.0)
    bits = assign_bitwidths(sensitivity, bytes_high, memory_budget_bytes=500.0)
    assert set(bits.tolist()) <= {2, 4}


# ---------------------------------------------------------------------------
# Model-level integration: apply_mixed_precision on a tiny synthetic LlamaModel
# ---------------------------------------------------------------------------

def test_apply_mixed_precision_replaces_weights_and_forward_runs():
    import mlx.core as mx
    from siliconfer.model.config import ModelConfig
    from siliconfer.model.llama import LlamaModel
    from siliconfer.quant.mixed_precision import apply_mixed_precision

    config = ModelConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        tie_word_embeddings=True,
        hidden_act="silu",
    )
    model = LlamaModel(config)
    mx.eval(model.parameters())

    W_before = [np.array(layer.self_attn.q_proj.weight) for layer in model.layers]
    bits_per_block = [4, 2, 4, 2]
    apply_mixed_precision(model, bits_per_block, group_size=64, verbose=False)
    W_after = [np.array(layer.self_attn.q_proj.weight) for layer in model.layers]

    for before, after in zip(W_before, W_after):
        assert not np.allclose(before, after), "Every block should have changed weights"

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    logits, cache = model(input_ids)
    mx.eval(logits)
    assert logits.shape == (1, 5, config.vocab_size)
    assert np.isfinite(np.array(logits)).all()


def test_apply_mixed_precision_2bit_blocks_lossier_than_4bit_blocks():
    """A block assigned 2 bits should reconstruct its weights worse than one
    assigned 4 bits, on the same underlying random init — sanity-checks that
    the bit assignment argument actually reaches the right quantizer."""
    import mlx.core as mx
    from siliconfer.model.config import ModelConfig
    from siliconfer.model.llama import LlamaModel
    from siliconfer.quant.mixed_precision import apply_mixed_precision

    config = ModelConfig(
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
    model = LlamaModel(config)
    mx.eval(model.parameters())

    W_orig = [np.array(layer.self_attn.q_proj.weight) for layer in model.layers]
    apply_mixed_precision(model, [4, 2], group_size=64, verbose=False)
    W_after = [np.array(layer.self_attn.q_proj.weight) for layer in model.layers]

    mse_4bit = float(np.mean((W_orig[0] - W_after[0]) ** 2))
    mse_2bit = float(np.mean((W_orig[1] - W_after[1]) ** 2))
    assert mse_2bit > mse_4bit


def test_apply_mixed_precision_wrong_length_raises():
    import mlx.core as mx
    from siliconfer.model.config import ModelConfig
    from siliconfer.model.llama import LlamaModel
    from siliconfer.quant.mixed_precision import apply_mixed_precision

    config = ModelConfig(
        architectures=["LlamaForCausalLM"], hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=32, max_position_embeddings=128, rms_norm_eps=1e-5,
        rope_theta=500000.0, tie_word_embeddings=True, hidden_act="silu",
    )
    model = LlamaModel(config)
    mx.eval(model.parameters())
    with pytest.raises(ValueError, match="entries"):
        apply_mixed_precision(model, [4], group_size=32, verbose=False)
