"""Unit tests: AWQ core algorithm on synthetic matrices.

No model weights downloaded — all tests use random numpy arrays.
"""

import mlx.core as mx
import numpy as np

from siliconfer.kernels.neon import pack_weights_sym
from siliconfer.model.q4_linear import Q4Linear
from siliconfer.quant.awq import (
    awq_clip_weight,
    awq_quantize_weight,
    awq_quantize_weight_components,
    awq_search_alpha,
    awq_search_alpha_by_loss,
)
from siliconfer.quant.primitives import fake_quantize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correlated_X(in_features: int, n_samples: int, phi: float = 0.9, seed: int = 0):
    """AR(1) correlated activations, [n_samples, in_features]."""
    rng = np.random.default_rng(seed)
    idx = np.arange(in_features)
    cov = phi ** np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    L   = np.linalg.cholesky(cov)
    Z   = rng.normal(0, 1, (in_features, n_samples))
    return (L @ Z).T.astype(np.float32)  # [n_samples, in_f]


def _output_error(W_orig, W_q, X):
    """||W_orig X^T - W_q X^T||_F  (X is [n_tok, in_f])."""
    Xt = X.T.astype(np.float64)
    return float(np.linalg.norm(
        W_orig.astype(np.float64) @ Xt - W_q.astype(np.float64) @ Xt, "fro"
    ))


# ---------------------------------------------------------------------------
# α = 0 should recover RTN
# ---------------------------------------------------------------------------

def test_awq_alpha_zero_is_rtn():
    """With α=0, s=1 (identity scale), AWQ reduces to RTN."""
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1, (32, 128)).astype(np.float32)
    act_scales = np.abs(rng.normal(0, 1, 128)).astype(np.float32) + 0.1

    W_awq = awq_quantize_weight(W, act_scales, alpha=0.0, group_size=128, sym=True)
    W_rtn = fake_quantize(W, group_size=128, sym=True)
    np.testing.assert_allclose(W_awq, W_rtn, atol=1e-5)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

def test_awq_output_shape():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (64, 128)).astype(np.float32)
    act_scales = (np.abs(rng.normal(0, 1, 128)) + 0.1).astype(np.float32)
    W_eff = awq_quantize_weight(W, act_scales, alpha=0.5)
    assert W_eff.shape == W.shape
    assert W_eff.dtype == np.float32


# ---------------------------------------------------------------------------
# AWQ beats RTN on calibration samples (the MSE AWQ minimises)
# ---------------------------------------------------------------------------

def test_awq_beats_rtn_on_calibration_samples():
    """AWQ with optimal α must achieve lower MSE than RTN on the calibration X."""
    rng = np.random.default_rng(7)
    out, in_f = 64, 128

    # Weights with skewed distribution — some input channels matter much more
    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)

    # Activations: channels have very different magnitudes (key for AWQ)
    act_scales = (np.abs(rng.normal(0, 2, in_f)) + 0.05).astype(np.float32)
    X = _make_correlated_X(in_f, n_samples=256, phi=0.8, seed=1)
    # Scale X by act_scales to match the magnitude pattern
    X = X * act_scales[None, :]

    # Recompute act_scales from this X
    act_scales_obs = np.abs(X).mean(axis=0).astype(np.float32)

    alpha = awq_search_alpha(W, act_scales_obs, X, group_size=128, sym=True, n_alpha=20)

    W_awq = awq_quantize_weight(W, act_scales_obs, alpha, group_size=128, sym=True)
    W_rtn = fake_quantize(W, group_size=128, sym=True)

    err_awq = _output_error(W, W_awq, X)
    err_rtn = _output_error(W, W_rtn, X)

    print(f"\n  RTN={err_rtn:.4f}, AWQ={err_awq:.4f}, α={alpha:.2f}")
    assert err_awq <= err_rtn, (
        f"AWQ (err={err_awq:.4f}) should beat RTN (err={err_rtn:.4f})"
    )


def test_awq_beats_rtn_multiple_groups():
    """AWQ with 2 groups of 128 should beat RTN."""
    rng = np.random.default_rng(99)
    out, in_f = 32, 256

    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)
    act_scales_raw = (np.abs(rng.normal(0, 3, in_f)) + 0.1).astype(np.float32)

    X = _make_correlated_X(in_f, n_samples=256, phi=0.8, seed=2)
    X = X * act_scales_raw[None, :]
    act_scales = np.abs(X).mean(axis=0).astype(np.float32)

    alpha = awq_search_alpha(W, act_scales, X, group_size=128, sym=True, n_alpha=20)

    W_awq = awq_quantize_weight(W, act_scales, alpha, group_size=128, sym=True)
    W_rtn = fake_quantize(W, group_size=128, sym=True)

    err_awq = _output_error(W, W_awq, X)
    err_rtn = _output_error(W, W_rtn, X)

    print(f"\n  2-group: RTN={err_rtn:.4f}, AWQ={err_awq:.4f}, α={alpha:.2f}")
    assert err_awq <= err_rtn


# ---------------------------------------------------------------------------
# AWQ picks nonzero α when activations are highly non-uniform
# ---------------------------------------------------------------------------

def test_awq_search_finds_nonzero_alpha():
    """When a few channels dominate activations, α>0 should be chosen."""
    rng = np.random.default_rng(123)
    out, in_f = 16, 128
    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)

    # Sparse activations: first 8 channels are 20× larger
    act_scales = np.ones(in_f, dtype=np.float32) * 0.1
    act_scales[:8] = 2.0

    X = rng.normal(0, 1, (256, in_f)).astype(np.float32)
    X = X * act_scales[None, :]

    alpha = awq_search_alpha(W, act_scales, X, group_size=128, n_alpha=20)
    assert alpha > 0.0, f"Expected α>0 for highly non-uniform activations, got {alpha}"


# ---------------------------------------------------------------------------
# Asymmetric AWQ
# ---------------------------------------------------------------------------

def test_awq_asym_beats_rtn():
    """Asymmetric AWQ should beat RTN on skewed + heterogeneous activations."""
    rng = np.random.default_rng(17)
    out, in_f = 32, 128

    W = (rng.normal(0, 1, (out, in_f)) + 1.0).astype(np.float32)
    act_scales_raw = (np.abs(rng.normal(0, 2, in_f)) + 0.1).astype(np.float32)
    X = rng.normal(0, 1, (256, in_f)).astype(np.float32)
    X = X * act_scales_raw[None, :]
    act_scales = np.abs(X).mean(axis=0).astype(np.float32)

    alpha = awq_search_alpha(W, act_scales, X, group_size=128, sym=False, n_alpha=20)
    W_awq = awq_quantize_weight(W, act_scales, alpha, group_size=128, sym=False)
    W_rtn = fake_quantize(W, group_size=128, sym=False)

    err_awq = _output_error(W, W_awq, X)
    err_rtn = _output_error(W, W_rtn, X)

    print(f"\n  asym: RTN={err_rtn:.4f}, AWQ={err_awq:.4f}, α={alpha:.2f}")
    assert err_awq <= err_rtn


# ---------------------------------------------------------------------------
# α=1 gives channel-normalised weights
# ---------------------------------------------------------------------------

def test_awq_alpha_one_scales_by_act():
    """With α=1, the weight is multiplied by act_scales (then dequantized back)."""
    rng = np.random.default_rng(55)
    W = rng.normal(0, 1, (8, 8)).astype(np.float32)
    act_scales = (np.abs(rng.normal(1, 0.5, 8)) + 0.1).astype(np.float32)

    W_eff = awq_quantize_weight(W, act_scales, alpha=1.0, group_size=8, sym=True)
    # W_eff = Q(W * s) / s ≈ W when quantization is fine
    # Just verify shape and that it differs from RTN
    assert W_eff.shape == W.shape
    W_rtn = fake_quantize(W, group_size=8, sym=True)
    assert not np.allclose(W_eff, W_rtn, atol=1e-3), \
        "α=1 should differ from α=0 (RTN) for non-uniform act_scales"


# ---------------------------------------------------------------------------
# Uniform activations: α=0 should be optimal (AWQ should not hurt RTN)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Packed-kernel round trip
#
# W_eff = Q(W*s)/s is not on a per-group int4 grid (s varies per column), so
# packing it re-quantizes it. The serving path packs W_grid = Q(W*s) and
# applies 1/s to the input (docs/packing.md).
# ---------------------------------------------------------------------------

def test_awq_packed_roundtrip_matches_weff():
    """Packing AWQ's grid-aligned components (W_grid + input_scale) through
    Q4Linear must reproduce W_eff's output, to float32 tolerance."""
    rng = np.random.default_rng(21)
    out_f, in_f = 32, 128
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)

    act_scales_raw = (np.abs(rng.normal(0, 2, in_f)) + 0.1).astype(np.float32)
    act_scales_raw[:8] *= 20  # a few salient channels
    X = rng.normal(0, 1, (256, in_f)).astype(np.float32) * act_scales_raw[None, :]
    act_scales = np.abs(X).mean(axis=0).astype(np.float32)

    alpha = awq_search_alpha(W, act_scales, X, group_size=128, sym=True, n_alpha=20)
    assert alpha > 0.0, "sanity: salient channels should pull alpha above 0"

    W_grid, s = awq_quantize_weight_components(W, act_scales, alpha, group_size=128, sym=True)
    W_eff = (W_grid.astype(np.float64) * (1.0 / s)[None, :]).astype(np.float32)

    packed, scales = pack_weights_sym(W_grid, group_size=128)
    layer = Q4Linear(packed, scales, group_size=128, input_scale=(1.0 / s).astype(np.float32))

    x_np = rng.normal(0, 1, (1, in_f)).astype(np.float32)
    y_packed = np.array(layer(mx.array(x_np)))[0]
    y_weff = W_eff @ x_np[0]

    np.testing.assert_allclose(y_packed, y_weff, atol=1e-3, rtol=1e-3)


def test_awq_packing_weff_directly_is_wrong():
    """Documents the bug this fix replaces: packing W_eff directly (ignoring
    the column scale, as the code did before) diverges substantially from
    AWQ's intended output on salient-channel data — this is why
    _pack_and_replace_linears now uses the grid-aligned components instead."""
    rng = np.random.default_rng(21)
    out_f, in_f = 32, 128
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)

    act_scales_raw = (np.abs(rng.normal(0, 2, in_f)) + 0.1).astype(np.float32)
    act_scales_raw[:8] *= 20
    X = rng.normal(0, 1, (256, in_f)).astype(np.float32) * act_scales_raw[None, :]
    act_scales = np.abs(X).mean(axis=0).astype(np.float32)

    alpha = awq_search_alpha(W, act_scales, X, group_size=128, sym=True, n_alpha=20)
    W_grid, s = awq_quantize_weight_components(W, act_scales, alpha, group_size=128, sym=True)
    W_eff = (W_grid.astype(np.float64) * (1.0 / s)[None, :]).astype(np.float32)

    packed_correct, scales_correct = pack_weights_sym(W_grid, group_size=128)
    layer_correct = Q4Linear(packed_correct, scales_correct, group_size=128,
                              input_scale=(1.0 / s).astype(np.float32))

    packed_bad, scales_bad = pack_weights_sym(W_eff, group_size=128)  # the old, buggy path
    layer_bad = Q4Linear(packed_bad, scales_bad, group_size=128)

    x_np = rng.normal(0, 1, (1, in_f)).astype(np.float32)
    y_ref = W_eff @ x_np[0]
    y_correct = np.array(layer_correct(mx.array(x_np)))[0]
    y_bad = np.array(layer_bad(mx.array(x_np)))[0]

    err_correct = np.linalg.norm(y_correct - y_ref)
    err_bad = np.linalg.norm(y_bad - y_ref)
    assert err_bad > err_correct * 5, (
        f"expected the naive re-pack to be much worse (err_bad={err_bad:.4f}, "
        f"err_correct={err_correct:.4f})"
    )


def test_awq_uniform_activations_alpha_zero():
    """With all-equal activation scales, α=0 is always optimal."""
    rng = np.random.default_rng(88)
    W = rng.normal(0, 1, (16, 128)).astype(np.float32)
    act_scales = np.ones(128, dtype=np.float32)  # perfectly uniform
    X = rng.normal(0, 1, (256, 128)).astype(np.float32)

    alpha = awq_search_alpha(W, act_scales, X, group_size=128, n_alpha=20)
    # Uniform scales → s = 1^alpha = 1 for any alpha; alpha returned can be anything
    # but AWQ error should == RTN error for all alpha
    W_awq = awq_quantize_weight(W, act_scales, alpha, group_size=128, sym=True)
    W_rtn = fake_quantize(W, group_size=128, sym=True)
    err_awq = _output_error(W, W_awq, X)
    err_rtn = _output_error(W, W_rtn, X)
    assert abs(err_awq - err_rtn) < 1e-3, \
        "Uniform activations: AWQ and RTN should give identical error"


# ---------------------------------------------------------------------------
# Weight clipping (llm-awq's auto_clip)
# ---------------------------------------------------------------------------

def _group_output_mse(W_ref, W_q, X, group_size):
    """Mean over samples of the squared per-group partial-output error, [out, n_groups]."""
    out_f, in_f = W_ref.shape
    n_groups = in_f // group_size
    D = (W_q.astype(np.float64) - W_ref.astype(np.float64)).reshape(out_f, n_groups, group_size)
    Xg = X.astype(np.float64).reshape(-1, n_groups, group_size)
    return (np.einsum("ogi,ngi->ong", D, Xg) ** 2).mean(axis=1)


def test_awq_clip_never_worse_than_unclipped():
    """i=0 of the search is "no clipping", so every (row, group) is at least as
    good as plain quantization on the samples it was searched on."""
    rng = np.random.default_rng(5)
    W = rng.normal(0, 1, (24, 256)).astype(np.float32)
    X = _make_correlated_X(256, 300, seed=6)

    W_clip = awq_clip_weight(W, X, group_size=128, sym=True)
    err_clip = _group_output_mse(W, fake_quantize(W_clip, 128, True), X, 128)
    err_plain = _group_output_mse(W, fake_quantize(W, 128, True), X, 128)

    assert np.all(err_clip <= err_plain + 1e-9)
    assert np.all(np.abs(W_clip) <= np.abs(W) + 1e-6), "clipping only shrinks magnitudes"


def test_awq_clip_helps_when_outlier_sits_on_a_quiet_channel():
    """A large weight on an input channel that carries almost no activation
    wastes the group's range; clipping it should reduce output error clearly."""
    rng = np.random.default_rng(7)
    W = rng.normal(0, 1, (16, 128)).astype(np.float32)
    W[:, 3] *= 12.0                                  # outlier column...
    X = rng.normal(0, 1, (400, 128)).astype(np.float32)
    X[:, 3] *= 1e-3                                  # ...that is nearly silent

    W_clip = awq_clip_weight(W, X, group_size=128, sym=True)
    err_clip = _group_output_mse(W, fake_quantize(W_clip, 128, True), X, 128).mean()
    err_plain = _group_output_mse(W, fake_quantize(W, 128, True), X, 128).mean()

    assert err_clip < 0.6 * err_plain, f"clip {err_clip:.4f} vs plain {err_plain:.4f}"
    assert np.abs(W_clip[:, 3]).max() < np.abs(W[:, 3]).max()


def test_awq_clipped_components_pack_losslessly():
    """Clipping happens before quantization, so W_grid is still on the int4 grid
    and the packed kernel must reproduce W_eff exactly as without clipping."""
    rng = np.random.default_rng(9)
    in_f = 128
    W = rng.normal(0, 1, (32, in_f)).astype(np.float32)
    W[:, 10] *= 8.0
    X = rng.normal(0, 1, (256, in_f)).astype(np.float32) * (np.abs(rng.normal(0, 2, in_f)) + 0.1)
    act_scales = np.abs(X).mean(axis=0).astype(np.float32)

    W_grid, s = awq_quantize_weight_components(W, act_scales, 0.5, 128, True, clip_samples=X)
    W_eff = (W_grid.astype(np.float64) / s[None, :]).astype(np.float32)

    packed, scales = pack_weights_sym(W_grid, group_size=128)
    layer = Q4Linear(packed, scales, group_size=128, input_scale=(1.0 / s).astype(np.float32))
    x_np = rng.normal(0, 1, (1, in_f)).astype(np.float32)
    np.testing.assert_allclose(np.array(layer(mx.array(x_np)))[0], W_eff @ x_np[0],
                               atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Block-output α search
# ---------------------------------------------------------------------------

def test_awq_search_by_loss_starts_from_rtn_and_picks_minimum():
    """The α returned is the argmin of the supplied loss, with α=0 meaning RTN."""
    rng = np.random.default_rng(11)
    W = rng.normal(0, 1, (8, 128)).astype(np.float32)
    act_scales = (np.abs(rng.normal(0, 1, 128)) + 0.1).astype(np.float32)
    W_rtn = fake_quantize(W, 128, True)

    seen = []

    def loss_fn(W_effs):
        seen.append(W_effs[0])
        return 1.0 if len(seen) != 8 else 0.25      # only the 8th call (α = 7/20) is better

    alpha = awq_search_alpha_by_loss([W], act_scales, loss_fn, 128, True, n_alpha=20)
    assert alpha == 7 / 20
    np.testing.assert_array_equal(seen[0], W_rtn)
    assert len(seen) == 21

    always_worse = lambda W_effs: 1.0
    assert awq_search_alpha_by_loss([W], act_scales, always_worse, 128, True, n_alpha=20) == 0.0
