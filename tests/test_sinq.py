"""Phase (post-9d) unit tests: SINQ (dual-scale-inspired) quantization.

No calibration data and no model downloads — SINQ only touches the weight
tensor itself, so all tests use random numpy arrays (plus one small synthetic
LlamaModel for the model-level integration test).
"""

import numpy as np
import pytest

from siliconfer.quant.primitives import fake_quantize
from siliconfer.quant.sinq import sinq_quantize_weight, apply_sinq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weight_mse(W_orig: np.ndarray, W_q: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (W_orig.astype(np.float64) - W_q.astype(np.float64)) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean())


def _make_column_outlier_weight(out_features: int, in_features: int, seed: int = 0):
    """Gaussian core with a few COLUMNS (not individual elements) that are
    systematically 15x larger in magnitude than the rest, shared across all
    output rows — the structure SINQ's column rescaling targets, distinct
    from HQQ's per-element outlier structure."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (out_features, in_features)).astype(np.float32)
    n_hot = max(1, in_features // 40)
    hot_cols = rng.choice(in_features, size=n_hot, replace=False)
    W[:, hot_cols] *= 15.0

    col_mask = np.zeros(in_features, dtype=bool)
    col_mask[hot_cols] = True
    return W, col_mask


# ---------------------------------------------------------------------------
# Output shape / dtype / boundedness
# ---------------------------------------------------------------------------

def test_sinq_output_shape():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (64, 128)).astype(np.float32)
    W_q = sinq_quantize_weight(W, group_size=128)
    assert W_q.shape == W.shape
    assert W_q.dtype == np.float32


def test_sinq_values_finite_and_bounded():
    W, _ = _make_column_outlier_weight(32, 128, seed=11)
    W_q = sinq_quantize_weight(W, group_size=128)
    assert np.isfinite(W_q).all()
    # column rescaling is bounded ([0.1, 10] clip on s), so reconstruction
    # shouldn't blow up wildly past the original dynamic range
    assert np.abs(W_q).max() <= np.abs(W).max() * 2.0


# ---------------------------------------------------------------------------
# Core correctness: SINQ beats RTN on column-outlier data
# ---------------------------------------------------------------------------

def test_sinq_beats_rtn_on_cold_columns():
    """SINQ should give much better resolution on the non-outlier ('cold')
    columns than plain RTN, since those columns' shared group scale is no
    longer dominated by a few systematically-larger columns."""
    W, hot_col_mask = _make_column_outlier_weight(32, 128, seed=1)
    cold_mask = np.tile(~hot_col_mask, (W.shape[0], 1))

    W_rtn = fake_quantize(W, group_size=128, sym=True)
    W_sinq = sinq_quantize_weight(W, group_size=128, sym=True, n_iters=10, beta=0.5)

    mse_rtn_cold = _weight_mse(W, W_rtn, cold_mask)
    mse_sinq_cold = _weight_mse(W, W_sinq, cold_mask)

    print(f"\n  cold-column MSE: RTN={mse_rtn_cold:.6f}, SINQ={mse_sinq_cold:.6f} "
          f"(improvement {(mse_rtn_cold - mse_sinq_cold) / mse_rtn_cold * 100:.1f}%)")
    assert mse_sinq_cold < mse_rtn_cold * 0.5, (
        "SINQ should give at least 2x lower MSE on the crushed cold columns"
    )


def test_sinq_beats_rtn_overall_despite_hot_column_tradeoff():
    """Total MSE (including the hot columns, which do get worse — a real,
    expected tradeoff, not a bug) should still improve overall when only a
    small fraction of columns are hot."""
    W, hot_col_mask = _make_column_outlier_weight(32, 128, seed=2)

    W_rtn = fake_quantize(W, group_size=128, sym=True)
    W_sinq = sinq_quantize_weight(W, group_size=128, sym=True, n_iters=10, beta=0.5)

    mse_rtn = _weight_mse(W, W_rtn)
    mse_sinq = _weight_mse(W, W_sinq)

    print(f"\n  overall MSE: RTN={mse_rtn:.6f}, SINQ={mse_sinq:.6f}")
    assert mse_sinq < mse_rtn


def test_sinq_relative_error_signal_matters():
    """Regression guard for the real bug found during development: using
    ABSOLUTE reconstruction error as the column-scale update signal (instead
    of relative error) gives only a tiny improvement, because absolute error
    is roughly uniform across columns sharing one group's quantization step
    regardless of column magnitude. This test locks in that the *shipped*
    algorithm (relative error) clears a bar the broken absolute-error version
    never would."""
    W, hot_col_mask = _make_column_outlier_weight(32, 128, seed=3)
    cold_mask = np.tile(~hot_col_mask, (W.shape[0], 1))

    W_rtn = fake_quantize(W, group_size=128, sym=True)
    W_sinq = sinq_quantize_weight(W, group_size=128, sym=True, n_iters=10, beta=0.5)

    mse_rtn_cold = _weight_mse(W, W_rtn, cold_mask)
    mse_sinq_cold = _weight_mse(W, W_sinq, cold_mask)
    improvement = (mse_rtn_cold - mse_sinq_cold) / mse_rtn_cold
    assert improvement > 0.8, f"expected >80% cold-column improvement, got {improvement:.1%}"


def test_sinq_no_outlier_not_worse_than_rtn():
    """On well-behaved Gaussian data (no injected column outliers), SINQ
    should be roughly on par with or better than RTN — there's nothing
    pathological for it to fix, but it shouldn't actively hurt either."""
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1, (32, 128)).astype(np.float32)

    W_rtn = fake_quantize(W, group_size=128, sym=True)
    W_sinq = sinq_quantize_weight(W, group_size=128, sym=True, n_iters=10, beta=0.5)

    mse_rtn = _weight_mse(W, W_rtn)
    mse_sinq = _weight_mse(W, W_sinq)

    print(f"\n  no-outlier MSE: RTN={mse_rtn:.6f}, SINQ={mse_sinq:.6f}")
    assert mse_sinq < mse_rtn * 1.1  # allow up to 10% worse in the total absence of outliers


# ---------------------------------------------------------------------------
# Small fallback: in_features < group_size
# ---------------------------------------------------------------------------

def test_sinq_small_matrix_fallback():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (8, 32)).astype(np.float32)
    W_q = sinq_quantize_weight(W, group_size=128)   # in_features=32 < group_size=128
    assert W_q.shape == W.shape


# ---------------------------------------------------------------------------
# Model-level integration: apply_sinq on a tiny synthetic LlamaModel
# ---------------------------------------------------------------------------

def test_apply_sinq_replaces_weights_and_forward_runs():
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
    apply_sinq(model, group_size=64, n_iters=5, verbose=False)
    W_after = np.array(model.layers[0].self_attn.q_proj.weight)

    assert W_after.shape == W_before.shape
    assert not np.allclose(W_before, W_after), "SINQ should have changed the weights"

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    logits, cache = model(input_ids)
    mx.eval(logits)
    assert logits.shape == (1, 5, config.vocab_size)
    assert np.isfinite(np.array(logits)).all()
