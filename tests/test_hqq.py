"""Unit tests: HQQ-style clip-range search.

Random NumPy arrays only, plus one small synthetic model.

Two `k_grid` settings are used. `_AGGRESSIVE_K_GRID` clips at moderate
z-scores and shows that the search finds a better fit than min/max rounding
on outlier-heavy groups. The library default is far more conservative (see
hqq.py) and is used by the tests that only check shapes, bounds, fallbacks
and model integration.
"""

import numpy as np
import pytest

from siliconfer.quant.hqq import apply_hqq, hqq_quantize_weight
from siliconfer.quant.primitives import fake_quantize

_AGGRESSIVE_K_GRID = (30.0, 20.0, 15.0, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weight_mse(W_orig: np.ndarray, W_q: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared weight reconstruction error, optionally restricted to `mask`."""
    diff = (W_orig.astype(np.float64) - W_q.astype(np.float64)) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean())


def _make_outlier_weight(out_features: int, in_features: int, seed: int = 0):
    """Gaussian core with a few extreme outliers per row and group.

    Returns (W, outlier_mask). Outliers are 50x the core scale.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (out_features, in_features)).astype(np.float32)
    outlier_mask = np.zeros_like(W, dtype=bool)

    for i in range(out_features):
        idx = rng.choice(in_features, size=2, replace=False)
        W[i, idx] = rng.choice([-1, 1], size=2) * 50.0
        outlier_mask[i, idx] = True

    return W, outlier_mask


# ---------------------------------------------------------------------------
# Output shape / dtype / boundedness (safe default k_grid)
# ---------------------------------------------------------------------------

def test_hqq_output_shape():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (64, 128)).astype(np.float32)
    W_q = hqq_quantize_weight(W, group_size=128)
    assert W_q.shape == W.shape
    assert W_q.dtype == np.float32


def test_hqq_values_bounded():
    """HQQ fake-quant values should not explode, even with extreme outliers."""
    W, _ = _make_outlier_weight(32, 128, seed=11)
    W_q = hqq_quantize_weight(W, group_size=128)
    # Outliers are 50.0; allow a little headroom for rounding to the grid's last step
    assert np.abs(W_q).max() <= 55.0, "HQQ weights unexpectedly large"
    assert np.isfinite(W_q).all()


# ---------------------------------------------------------------------------
# Core correctness: HQQ beats RTN when a group has outliers (aggressive k_grid)
# ---------------------------------------------------------------------------

def test_hqq_beats_rtn_on_outliers():
    """With a grid that clips, the non-outlier bulk of a group is represented
    much better than with min/max asymmetric RTN.
    """
    W, outlier_mask = _make_outlier_weight(64, 128, seed=1)
    bulk_mask = ~outlier_mask

    W_rtn = fake_quantize(W, group_size=128, sym=False)
    W_hqq = hqq_quantize_weight(W, group_size=128, p=0.7, k_grid=_AGGRESSIVE_K_GRID)

    mse_rtn_bulk = _weight_mse(W, W_rtn, bulk_mask)
    mse_hqq_bulk = _weight_mse(W, W_hqq, bulk_mask)

    print(f"\n  bulk MSE: RTN={mse_rtn_bulk:.6f}, HQQ={mse_hqq_bulk:.6f} "
          f"(improvement {(mse_rtn_bulk - mse_hqq_bulk) / mse_rtn_bulk * 100:.1f}%)")
    assert mse_hqq_bulk < mse_rtn_bulk, (
        f"HQQ (bulk MSE={mse_hqq_bulk:.6f}) should beat RTN "
        f"(bulk MSE={mse_rtn_bulk:.6f}) on the non-outlier weights"
    )


def test_hqq_beats_rtn_multiple_groups():
    """Same outlier-robustness property should hold with 2 groups of 128."""
    W, outlier_mask = _make_outlier_weight(32, 256, seed=99)
    bulk_mask = ~outlier_mask

    W_rtn = fake_quantize(W, group_size=128, sym=False)
    W_hqq = hqq_quantize_weight(W, group_size=128, p=0.7, k_grid=_AGGRESSIVE_K_GRID)

    mse_rtn_bulk = _weight_mse(W, W_rtn, bulk_mask)
    mse_hqq_bulk = _weight_mse(W, W_hqq, bulk_mask)

    print(f"\n  2-group bulk MSE: RTN={mse_rtn_bulk:.6f}, HQQ={mse_hqq_bulk:.6f}")
    assert mse_hqq_bulk < mse_rtn_bulk


@pytest.mark.parametrize("p", [0.5, 0.7, 1.0])
def test_hqq_lower_p_more_robust(p):
    """Never worse than RTN on bulk MSE for p in (0, 1]; smaller p should be
    strictly better on this outlier-heavy data, p = 1 may only tie.
    """
    W, outlier_mask = _make_outlier_weight(32, 128, seed=5)
    bulk_mask = ~outlier_mask

    W_rtn = fake_quantize(W, group_size=128, sym=False)
    W_hqq = hqq_quantize_weight(W, group_size=128, p=p, k_grid=_AGGRESSIVE_K_GRID)

    mse_rtn_bulk = _weight_mse(W, W_rtn, bulk_mask)
    mse_hqq_bulk = _weight_mse(W, W_hqq, bulk_mask)

    print(f"\n  p={p}: RTN={mse_rtn_bulk:.6f}, HQQ={mse_hqq_bulk:.6f}")
    assert mse_hqq_bulk <= mse_rtn_bulk + 1e-6, f"p={p}: HQQ should not be worse than RTN on bulk MSE"
    if p < 1.0:
        assert mse_hqq_bulk < mse_rtn_bulk, f"p={p}: HQQ should strictly beat RTN when p<1"


def test_hqq_never_worse_than_rtn_in_lp_loss():
    """The L_p loss is never worse than min/max RTN's, on any data and any
    k_grid, because the untrimmed range is always a candidate.

    This is a statement about the L_p loss, not MSE: with p < 1 the search may
    trim part of an ordinary tail and raise MSE slightly.
    """
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1, (32, 128)).astype(np.float32)
    p = 0.7

    W_rtn = fake_quantize(W, group_size=128, sym=False)
    W_hqq = hqq_quantize_weight(W, group_size=128, p=p, k_grid=_AGGRESSIVE_K_GRID)

    loss_rtn = float((np.abs(W.astype(np.float64) - W_rtn.astype(np.float64)) ** p).mean())
    loss_hqq = float((np.abs(W.astype(np.float64) - W_hqq.astype(np.float64)) ** p).mean())

    print(f"\n  L_p loss: RTN={loss_rtn:.6f}, HQQ={loss_hqq:.6f}")
    assert loss_hqq <= loss_rtn + 1e-6


# ---------------------------------------------------------------------------
# Small fallback: in_features < group_size
# ---------------------------------------------------------------------------

def test_hqq_small_matrix_fallback():
    """When in_features < group_size, hqq_quantize_weight should not crash."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (8, 32)).astype(np.float32)
    W_q = hqq_quantize_weight(W, group_size=128)   # in_features=32 < group_size=128
    assert W_q.shape == W.shape


# ---------------------------------------------------------------------------
# Model-level integration: apply_hqq on a tiny synthetic LlamaModel
# ---------------------------------------------------------------------------

def test_apply_hqq_replaces_weights_and_forward_runs():
    import mlx.core as mx

    from siliconfer.model.config import ModelConfig
    from siliconfer.model.llama import LlamaModel

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

    W_before = np.array(model.layers[0].self_attn.q_proj.weight)
    apply_hqq(model, group_size=64, verbose=False)
    W_after = np.array(model.layers[0].self_attn.q_proj.weight)

    assert W_after.shape == W_before.shape
    assert not np.allclose(W_before, W_after), "HQQ should have changed the weights"

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    logits, cache = model(input_ids)
    mx.eval(logits)
    assert logits.shape == (1, 5, config.vocab_size)
    assert np.isfinite(np.array(logits)).all()
