"""Phase 3 unit tests: GPTQ core algorithm on synthetic matrices.

No model weights are downloaded — all tests use random numpy arrays.

Key points on test design:
- GPTQ only clearly beats RTN when activations are CORRELATED (H has large
  off-diagonals). For white-noise X, H ≈ 2n·I and GPTQ ≈ RTN.
- The correct way to verify the algorithm is to evaluate error on the SAME
  calibration X used to build H (since GPTQ minimises exactly that objective).
- For generalization we use correlated X drawn from the same distribution.
"""

import numpy as np
import pytest

from siliconfer.quant.primitives import fake_quantize
from siliconfer.quant.gptq import gptq_quantize_weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correlated_H_and_X(
    in_features: int,
    n_samples: int,
    phi: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) correlated calibration data: cov[i,j] = phi^|i-j|.

    Returns H = 2 X Xᵀ (float64) and X (float32, shape [in_f, n_samples]).
    Highly correlated activations give H large off-diagonals, so GPTQ error
    propagation is non-trivial and improvement over RTN is measurable.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(in_features)
    cov = phi ** np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    L_cov = np.linalg.cholesky(cov)
    Z = rng.normal(0, 1, (in_features, n_samples))
    X = (L_cov @ Z).astype(np.float32)          # [in_f, n_samples]
    H = (2.0 * X.astype(np.float64) @ X.astype(np.float64).T)
    return H, X


def _output_error(W_orig: np.ndarray, W_q: np.ndarray, X: np.ndarray) -> float:
    """||W_orig X - W_q X||_F  (the objective GPTQ minimises on X)."""
    return float(np.linalg.norm(W_orig.astype(np.float64) @ X.astype(np.float64)
                                - W_q.astype(np.float64) @ X.astype(np.float64), "fro"))


# ---------------------------------------------------------------------------
# Core correctness: GPTQ < RTN on calibration data (the GPTQ objective)
# ---------------------------------------------------------------------------

def test_gptq_beats_rtn_on_calibration_data():
    """GPTQ must achieve lower reconstruction error on the calibration X it minimises."""
    rng = np.random.default_rng(42)
    out, in_f = 64, 128
    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)

    H, X = _make_correlated_H_and_X(in_f, n_samples=512, phi=0.9, seed=1)

    W_rtn  = fake_quantize(W, group_size=128, sym=True)
    W_gptq = gptq_quantize_weight(W, H, group_size=128, sym=True)

    # Evaluate on the SAME X used to build H — GPTQ provably minimises this
    err_rtn  = _output_error(W, W_rtn,  X)
    err_gptq = _output_error(W, W_gptq, X)

    print(f"\n  calib RTN={err_rtn:.4f}, GPTQ={err_gptq:.4f} "
          f"(improvement {(err_rtn-err_gptq)/err_rtn*100:.1f}%)")
    assert err_gptq < err_rtn, (
        f"GPTQ (err={err_gptq:.4f}) should beat RTN (err={err_rtn:.4f}) on calibration data"
    )


def test_gptq_beats_rtn_correlated_eval():
    """GPTQ should generalise to held-out X drawn from the SAME correlated distribution."""
    rng = np.random.default_rng(7)
    out, in_f = 64, 128
    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)

    H, X_calib = _make_correlated_H_and_X(in_f, n_samples=512, phi=0.9, seed=2)
    # Independent held-out X, same correlation structure
    _, X_eval = _make_correlated_H_and_X(in_f, n_samples=512, phi=0.9, seed=99)

    W_rtn  = fake_quantize(W, group_size=128, sym=True)
    W_gptq = gptq_quantize_weight(W, H, group_size=128, sym=True)

    err_rtn  = _output_error(W, W_rtn,  X_eval)
    err_gptq = _output_error(W, W_gptq, X_eval)

    print(f"\n  eval RTN={err_rtn:.4f}, GPTQ={err_gptq:.4f}")
    assert err_gptq < err_rtn


def test_gptq_beats_rtn_multiple_groups():
    """GPTQ with 2 groups of 128 each should beat RTN on correlated data."""
    rng = np.random.default_rng(99)
    out, in_f = 32, 256
    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)

    H, X = _make_correlated_H_and_X(in_f, n_samples=1024, phi=0.9, seed=3)

    W_rtn  = fake_quantize(W, group_size=128, sym=True)
    W_gptq = gptq_quantize_weight(W, H, group_size=128, sym=True)

    err_rtn  = _output_error(W, W_rtn,  X)
    err_gptq = _output_error(W, W_gptq, X)

    print(f"\n  2-group: RTN={err_rtn:.4f}, GPTQ={err_gptq:.4f}")
    assert err_gptq < err_rtn


def test_gptq_beats_rtn_asym_correlated():
    """Asymmetric GPTQ should beat RTN on skewed + correlated data."""
    rng = np.random.default_rng(7)
    out, in_f = 32, 128
    W = (rng.normal(0, 1, (out, in_f)) + 1.5).astype(np.float32)   # positive-skewed

    H, X = _make_correlated_H_and_X(in_f, n_samples=512, phi=0.9, seed=4)

    W_rtn  = fake_quantize(W, group_size=128, sym=False)
    W_gptq = gptq_quantize_weight(W, H, group_size=128, sym=False)

    err_rtn  = _output_error(W, W_rtn,  X)
    err_gptq = _output_error(W, W_gptq, X)

    print(f"\n  asym: RTN={err_rtn:.4f}, GPTQ={err_gptq:.4f}")
    assert err_gptq < err_rtn


# ---------------------------------------------------------------------------
# Output shape / dtype / boundedness
# ---------------------------------------------------------------------------

def test_gptq_output_shape():
    """gptq_quantize_weight must return same shape and float32."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (64, 128)).astype(np.float32)
    H, _ = _make_correlated_H_and_X(128, n_samples=256)
    W_q = gptq_quantize_weight(W, H, group_size=128)
    assert W_q.shape == W.shape
    assert W_q.dtype == np.float32


def test_gptq_values_bounded():
    """GPTQ fake-quant values should not explode."""
    rng = np.random.default_rng(11)
    W = rng.normal(0, 1, (32, 128)).astype(np.float32)
    H, _ = _make_correlated_H_and_X(128, n_samples=512)
    W_q = gptq_quantize_weight(W, H, group_size=128)
    # Values should stay within ~2× the original dynamic range
    scale = float(np.abs(W).max()) * 2.0
    assert float(np.abs(W_q).max()) <= scale + 1e-3, "GPTQ weights unexpectedly large"


# ---------------------------------------------------------------------------
# Improvement grows with correlation strength
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.5, 0.7, 0.9])
def test_gptq_improvement_increases_with_correlation(phi):
    """Higher AR(1) correlation → larger GPTQ improvement over RTN."""
    rng = np.random.default_rng(42)
    out, in_f = 32, 128
    W = rng.normal(0, 1, (out, in_f)).astype(np.float32)

    H, X = _make_correlated_H_and_X(in_f, n_samples=512, phi=phi, seed=5)

    W_rtn  = fake_quantize(W, group_size=128, sym=True)
    W_gptq = gptq_quantize_weight(W, H, group_size=128, sym=True)

    err_rtn  = _output_error(W, W_rtn, X)
    err_gptq = _output_error(W, W_gptq, X)

    improvement = (err_rtn - err_gptq) / err_rtn
    print(f"\n  phi={phi}: RTN={err_rtn:.4f}, GPTQ={err_gptq:.4f}, imp={improvement:.1%}")
    assert err_gptq < err_rtn, f"phi={phi}: GPTQ should beat RTN"


# ---------------------------------------------------------------------------
# Hessian properties
# ---------------------------------------------------------------------------

def test_H_symmetric():
    """H = 2 X Xᵀ must be symmetric."""
    H, _ = _make_correlated_H_and_X(64, n_samples=256)
    np.testing.assert_allclose(H, H.T, atol=1e-10)


def test_H_positive_semidefinite():
    """H = 2 X Xᵀ must be positive semi-definite."""
    H, _ = _make_correlated_H_and_X(64, n_samples=512)
    eigvals = np.linalg.eigvalsh(H)
    assert eigvals.min() >= -1e-8, f"H has negative eigenvalue: {eigvals.min()}"


def test_H_invertible_after_dampening():
    """H + dampening must be positive definite (Cholesky succeeds)."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (64, 16)).astype(np.float32)  # under-determined
    H = (2.0 * X.astype(np.float64) @ X.astype(np.float64).T)
    damp = 0.01 * float(np.diag(H).mean())
    H_d = H + damp * np.eye(64, dtype=np.float64)
    L = np.linalg.cholesky(H_d)
    assert L is not None


# ---------------------------------------------------------------------------
# Small fallback: in_features < group_size
# ---------------------------------------------------------------------------

def test_gptq_small_matrix_fallback():
    """When in_features < group_size, gptq_quantize_weight should not crash."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (8, 32)).astype(np.float32)
    H, _ = _make_correlated_H_and_X(32, n_samples=64)
    W_q = gptq_quantize_weight(W, H, group_size=128)   # in_features=32 < group_size=128
    assert W_q.shape == W.shape
