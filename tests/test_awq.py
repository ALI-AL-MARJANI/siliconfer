"""Phase 4 unit tests: AWQ core algorithm on synthetic matrices.

No model weights downloaded — all tests use random numpy arrays.
"""

import numpy as np
import pytest

from siliconfer.quant.primitives import fake_quantize
from siliconfer.quant.awq import awq_search_alpha, awq_quantize_weight


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
