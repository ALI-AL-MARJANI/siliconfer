"""Phase (post-9d) unit tests: SINQ (dual-scale-inspired) quantization.

No calibration data and no model downloads — SINQ only touches the weight
tensor itself, so all tests use random numpy arrays (plus one small synthetic
LlamaModel for the model-level integration test).
"""

import mlx.core as mx
import numpy as np

from siliconfer.kernels.neon import pack_weights_sym
from siliconfer.model.q4_linear import Q4Linear
from siliconfer.quant.primitives import fake_quantize
from siliconfer.quant.sinq import apply_sinq, sinq_quantize_weight, sinq_quantize_weight_components

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
    """Total MSE improves when only a small fraction of columns are hot, even
    though the hot columns themselves get worse."""
    W, hot_col_mask = _make_column_outlier_weight(32, 128, seed=2)

    W_rtn = fake_quantize(W, group_size=128, sym=True)
    W_sinq = sinq_quantize_weight(W, group_size=128, sym=True, n_iters=10, beta=0.5)

    mse_rtn = _weight_mse(W, W_rtn)
    mse_sinq = _weight_mse(W, W_sinq)

    print(f"\n  overall MSE: RTN={mse_rtn:.6f}, SINQ={mse_sinq:.6f}")
    assert mse_sinq < mse_rtn


def test_sinq_relative_error_signal_matters():
    """The column-scale update must use relative error. Absolute error is
    roughly uniform across the columns of a group, whatever their magnitude,
    and gives only a marginal improvement; the bar here is one that an
    absolute-error update does not reach."""
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
# Packed-kernel round trip — regression guard for the double-quantization bug
# (same class of bug as AWQ's: W_eff = Q(W*s)/s is off-grid, packing it
# directly silently re-quantizes and erases SINQ's benefit).
# ---------------------------------------------------------------------------

def test_sinq_packed_roundtrip_matches_weff():
    """Packing SINQ's grid-aligned components through Q4Linear must reproduce
    W_eff's output, to float32 tolerance."""
    W, _ = _make_column_outlier_weight(32, 128, seed=31)

    W_grid, s = sinq_quantize_weight_components(W, group_size=128, sym=True, n_iters=10, beta=0.5)
    W_eff = (W_grid.astype(np.float64) / s[None, :]).astype(np.float32)

    packed, scales = pack_weights_sym(W_grid, group_size=128)
    layer = Q4Linear(packed, scales, group_size=128, input_scale=(1.0 / s).astype(np.float32))

    rng = np.random.default_rng(32)
    x_np = rng.normal(0, 1, (1, 128)).astype(np.float32)
    y_packed = np.array(layer(mx.array(x_np)))[0]
    y_weff = W_eff @ x_np[0]

    np.testing.assert_allclose(y_packed, y_weff, atol=1e-3, rtol=1e-3)


def test_sinq_packing_weff_directly_is_wrong():
    """Documents the bug this fix replaces: packing W_eff directly (as the
    code did before) diverges substantially from SINQ's intended output on
    column-outlier data."""
    W, _ = _make_column_outlier_weight(32, 128, seed=31)

    W_grid, s = sinq_quantize_weight_components(W, group_size=128, sym=True, n_iters=10, beta=0.5)
    W_eff = (W_grid.astype(np.float64) / s[None, :]).astype(np.float32)

    packed_correct, scales_correct = pack_weights_sym(W_grid, group_size=128)
    layer_correct = Q4Linear(packed_correct, scales_correct, group_size=128,
                              input_scale=(1.0 / s).astype(np.float32))

    packed_bad, scales_bad = pack_weights_sym(W_eff, group_size=128)  # the old, buggy path
    layer_bad = Q4Linear(packed_bad, scales_bad, group_size=128)

    rng = np.random.default_rng(32)
    x_np = rng.normal(0, 1, (1, 128)).astype(np.float32)
    y_ref = W_eff @ x_np[0]
    y_correct = np.array(layer_correct(mx.array(x_np)))[0]
    y_bad = np.array(layer_bad(mx.array(x_np)))[0]

    err_correct = np.linalg.norm(y_correct - y_ref)
    err_bad = np.linalg.norm(y_bad - y_ref)
    assert err_bad > err_correct * 5, (
        f"expected the naive re-pack to be much worse (err_bad={err_bad:.4f}, "
        f"err_correct={err_correct:.4f})"
    )


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
