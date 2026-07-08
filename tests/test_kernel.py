"""Phase 5 kernel tests: NEON q4 GEMV/GEMM correctness.

These tests run against the compiled siliconfer_neon extension if it is built,
or against the numpy fallback if it is not. In both cases the results should
match the numpy reference within float32 rounding tolerance.
"""

import numpy as np
import pytest

from siliconfer.kernels.neon import (
    NEON_AVAILABLE,
    pack_weights_sym,
    pack_weights_asym,
    gemv_sym,
    gemv_scalar,
    gemv_asym,
    gemm_sym,
    gemm_asym,
)
from siliconfer.quant.primitives import fake_quantize, quantize_sym, dequantize_sym


# ---------------------------------------------------------------------------
# Reference: numpy GEMV using fake-quant weights
# ---------------------------------------------------------------------------

def _ref_gemv(W_fp32: np.ndarray, x: np.ndarray, group_size: int) -> np.ndarray:
    """Reference: fake-quantize W then compute W_q @ x in float32."""
    W_q = fake_quantize(W_fp32, group_size=group_size, sym=True).astype(np.float32)
    return (W_q @ x.astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Packing round-trip
# ---------------------------------------------------------------------------

def test_pack_unpack_round_trip():
    """Packing + kernel dequant should recover the fake-quant weights."""
    rng = np.random.default_rng(0)
    out_f, in_f, gs = 32, 128, 128
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)

    packed, scales = pack_weights_sym(W, group_size=gs)

    assert packed.shape == (out_f, in_f // 2)
    assert packed.dtype == np.uint8
    assert scales.shape == (out_f, in_f // gs)

    # Manually dequantize: lo nibble = even col, hi nibble = odd col
    lo = (packed & 0x0F).astype(np.int16)
    hi = (packed >> 4).astype(np.int16)
    lo_s = np.where(lo < 8, lo, lo - 16).astype(np.float32)
    hi_s = np.where(hi < 8, hi, hi - 16).astype(np.float32)

    W_rec = np.empty((out_f, in_f), dtype=np.float32)
    W_rec[:, 0::2] = lo_s * scales                  # even cols
    W_rec[:, 1::2] = hi_s * scales                  # odd cols

    W_ref = fake_quantize(W, group_size=gs, sym=True)
    np.testing.assert_allclose(W_rec, W_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# GEMV correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group_size", [64, 128])
def test_gemv_sym_matches_reference(group_size):
    """gemv_sym output should match fake_quantize + numpy matmul."""
    rng = np.random.default_rng(42)
    out_f, in_f = 128, 256
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x = rng.normal(0, 1, in_f).astype(np.float32)

    packed, scales = pack_weights_sym(W, group_size=group_size)
    y_kernel = gemv_sym(packed, scales, x, group_size)
    y_ref    = _ref_gemv(W, x, group_size)

    np.testing.assert_allclose(y_kernel, y_ref, atol=1e-4, rtol=1e-4)


def test_gemv_sym_scalar_matches_neon():
    """Scalar-C and NEON paths should produce identical results."""
    rng = np.random.default_rng(7)
    out_f, in_f = 64, 128
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x = rng.normal(0, 1, in_f).astype(np.float32)

    packed, scales = pack_weights_sym(W)
    y_neon   = gemv_sym(packed, scales, x)
    y_scalar = gemv_scalar(packed, scales, x)

    np.testing.assert_allclose(y_neon, y_scalar, atol=1e-5)


def test_gemv_large():
    """GEMV on a larger weight matrix (typical 7B layer size)."""
    rng = np.random.default_rng(1)
    out_f, in_f = 4096, 4096   # typical for 7B feed-forward
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x = rng.normal(0, 1, in_f).astype(np.float32)

    packed, scales = pack_weights_sym(W, group_size=128)
    y_kernel = gemv_sym(packed, scales, x, group_size=128)
    y_ref    = _ref_gemv(W, x, group_size=128)

    # Larger matrix: accumulation errors can grow, relax tolerance slightly
    np.testing.assert_allclose(y_kernel, y_ref, atol=5e-3, rtol=1e-3)


def test_gemv_zero_input():
    """Zero input should give zero output."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (32, 128)).astype(np.float32)
    packed, scales = pack_weights_sym(W)
    y = gemv_sym(packed, scales, np.zeros(128, dtype=np.float32))
    np.testing.assert_allclose(y, 0.0, atol=1e-6)


def test_gemv_single_output_row():
    """Works for out_features=1."""
    rng = np.random.default_rng(5)
    W = rng.normal(0, 1, (1, 128)).astype(np.float32)
    x = rng.normal(0, 1, 128).astype(np.float32)
    packed, scales = pack_weights_sym(W)
    y = gemv_sym(packed, scales, x)
    y_ref = _ref_gemv(W, x, 128)
    np.testing.assert_allclose(y, y_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# GEMM correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 4, 32])
def test_gemm_sym_matches_gemv_loop(T):
    """GEMM output should equal T independent GEMV calls."""
    rng = np.random.default_rng(99)
    out_f, in_f = 64, 128
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    X = rng.normal(0, 1, (T, in_f)).astype(np.float32)

    packed, scales = pack_weights_sym(W)
    Y_gemm = gemm_sym(packed, scales, X)

    # Reference: T independent GEMV calls
    Y_ref = np.stack([gemv_sym(packed, scales, X[t]) for t in range(T)])

    assert Y_gemm.shape == (T, out_f)
    np.testing.assert_allclose(Y_gemm, Y_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# GEMM correctness, asymmetric (Phase 9-follow-up: closes the gap documented
# in CLAUDE.md §9 — Q4Linear previously had no way to run prefill on
# asymmetrically-quantized weights at all, silently corrupting HQQ's output
# by re-packing it symmetric).
# ---------------------------------------------------------------------------

def _ref_gemv_asym(W_fp32: np.ndarray, x: np.ndarray, group_size: int) -> np.ndarray:
    """Reference: asymmetric fake-quantize W then compute W_q @ x in float32."""
    W_q = fake_quantize(W_fp32, group_size=group_size, sym=False).astype(np.float32)
    return (W_q @ x.astype(np.float32)).astype(np.float32)


@pytest.mark.parametrize("group_size", [64, 128])
def test_gemv_asym_matches_reference(group_size):
    rng = np.random.default_rng(42)
    out_f, in_f = 128, 256
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x = rng.normal(0, 1, in_f).astype(np.float32)

    packed, scales, zeros = pack_weights_asym(W, group_size=group_size)
    y_kernel = gemv_asym(packed, scales, zeros, x, group_size)
    y_ref = _ref_gemv_asym(W, x, group_size)

    np.testing.assert_allclose(y_kernel, y_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("T", [1, 4, 32])
def test_gemm_asym_matches_gemv_asym_loop(T):
    """GEMM (asymmetric) output should equal T independent GEMV(asym) calls."""
    rng = np.random.default_rng(99)
    out_f, in_f = 64, 128
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    X = rng.normal(0, 1, (T, in_f)).astype(np.float32)

    packed, scales, zeros = pack_weights_asym(W)
    Y_gemm = gemm_asym(packed, scales, zeros, X)

    Y_ref = np.stack([gemv_asym(packed, scales, zeros, X[t]) for t in range(T)])

    assert Y_gemm.shape == (T, out_f)
    np.testing.assert_allclose(Y_gemm, Y_ref, atol=1e-5)


def test_gemm_asym_matches_reference_matmul():
    rng = np.random.default_rng(3)
    out_f, in_f, T = 32, 128, 8
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    X = rng.normal(0, 1, (T, in_f)).astype(np.float32)

    packed, scales, zeros = pack_weights_asym(W, group_size=128)
    Y_kernel = gemm_asym(packed, scales, zeros, X, group_size=128)

    W_q = fake_quantize(W, group_size=128, sym=False).astype(np.float32)
    Y_ref = X @ W_q.T

    np.testing.assert_allclose(Y_kernel, Y_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Extension availability
# ---------------------------------------------------------------------------

def test_kernel_import():
    """Kernel loads (NEON or numpy fallback) without error."""
    from siliconfer.kernels import neon  # noqa: F401
    print(f"\n  NEON_AVAILABLE={NEON_AVAILABLE}")
    # Both paths must produce a result without crashing
    W = np.random.default_rng(0).normal(0, 1, (16, 64)).astype(np.float32)
    x = np.ones(64, dtype=np.float32)
    packed, scales = pack_weights_sym(W, group_size=64)
    y = gemv_sym(packed, scales, x, group_size=64)
    assert y.shape == (16,)
