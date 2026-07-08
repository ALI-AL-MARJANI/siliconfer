"""Phase 9a unit tests: HQQ (Half-Quadratic Quantization) core algorithm.

No calibration data and no model downloads — HQQ only touches the weight
tensor itself, so all tests use random numpy arrays (plus one small synthetic
LlamaModel for the model-level integration test).

Key point on test design: these tests use two different `k_grid` settings on
purpose.

- `_AGGRESSIVE_K_GRID` demonstrates that the search *mechanism* is correct:
  given a genuinely outlier-heavy group and a k_grid willing to clip at
  moderate z-scores, HQQ finds a materially better fit than plain min-max
  RTN for the non-outlier bulk of the group.
- The library *default* k_grid (imported as `hqq.hqq_quantize_weight`'s
  default) is deliberately far more conservative. Validating against real
  Qwen2.5-0.5B weights surfaced a "super weight" (see hqq.py's module
  docstring) at robust z-score ~58 whose clipping alone pushed WikiText-2 PPL
  from ~26 to 400+ — real trained weights can have lone, structurally
  critical outliers that look statistically identical to "safe to clip"
  under magnitude alone. The default grid is set high enough that no
  plausible real super-weight gets caught, at the cost of not doing much for
  ordinary (non-pathological) real weight tensors. Tests that only care about
  plumbing/safety (shape, boundedness, fallback, integration) use the safe
  default; tests that demonstrate the outlier-robustness mechanism itself use
  the aggressive grid explicitly.
"""

import numpy as np
import pytest

from siliconfer.quant.primitives import fake_quantize
from siliconfer.quant.hqq import hqq_quantize_weight, apply_hqq

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
    """Gaussian core with a handful of extreme outliers per row (per group).

    Returns (W, outlier_mask) where outlier_mask marks the injected outliers.

    Outliers are 50x the core scale (robust z ~ 50) — large enough to clearly
    demonstrate the mechanism under `_AGGRESSIVE_K_GRID`, but note this is
    deliberately *not* used with the library's conservative default grid in
    these tests (see module docstring).
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
    """HQQ's robust fit should give much better resolution on the non-outlier
    bulk of a group than naive min-max asymmetric RTN, at the cost of the
    (already-hopeless) outliers themselves — when configured to actually
    clip (see module docstring on why the library default is conservative)."""
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
    """HQQ should never do worse than RTN on bulk MSE for any p in (0, 1].
    Smaller p (heavier-tailed prior) is expected to find a strictly better
    fit on this outlier-heavy data; p=1.0 (plain L1) may only tie with RTN
    since it discounts large clipped errors less aggressively."""
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
    """HQQ's own objective (mean L_p reconstruction loss) must never be worse
    than plain min-max RTN, on any data and with any k_grid — by
    construction, since the untrimmed min-max range is always included as a
    search candidate (whether or not it ends up chosen).

    Note this is NOT an MSE/L2 claim: since HQQ optimizes L_p with p<1 (not
    L2), it will happily trim a bit of the natural sample tail even on clean
    Gaussian data if that lowers the *L_p* loss, which can slightly raise raw
    MSE without any pathological outliers present. That's expected, correct
    L_p-robust behavior, not a bug — so we verify the metric HQQ actually
    optimizes, not MSE, on this "no injected outliers" case. Uses the
    aggressive k_grid since that's the regime where trimming is most likely
    to be (mis)chosen.
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
