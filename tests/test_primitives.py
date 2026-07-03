"""Phase 2 unit tests: group-wise int4 quantization primitives.

All tests run on synthetic numpy arrays — no model weights downloaded.
"""

import numpy as np
import pytest

from siliconfer.quant.primitives import (
    quantize_sym,
    dequantize_sym,
    quantize_asym,
    dequantize_asym,
    pack_int4,
    unpack_int4,
    fake_quantize,
)


# ---------------------------------------------------------------------------
# Symmetric round-trip
# ---------------------------------------------------------------------------

def test_sym_round_trip_values():
    """Symmetric quantize → dequantize values should be close to original."""
    rng = np.random.default_rng(42)
    w = rng.normal(0, 1, (64, 128)).astype(np.float32)
    q, scales = quantize_sym(w, group_size=128)
    w_r = dequantize_sym(q, scales, group_size=128)

    max_scale = float(scales.max())
    max_err = float(np.abs(w - w_r).max())
    rmse = float(np.sqrt(np.mean((w - w_r) ** 2)))

    print(f"\n  sym round-trip: max_err={max_err:.5f}, rmse={rmse:.5f}, max_scale={max_scale:.5f}")
    assert max_err <= max_scale + 1e-5, f"Max error {max_err} > max_scale {max_scale}"
    assert rmse < 0.15 * float(w.std()), f"RMSE {rmse} unreasonably large"


def test_sym_q_range():
    """Symmetric q values must sit in [-8, 7]."""
    rng = np.random.default_rng(0)
    w = rng.normal(0, 10, (32, 128)).astype(np.float32)
    q, _ = quantize_sym(w, group_size=128)
    assert int(q.min()) >= -8 and int(q.max()) <= 7


def test_sym_scales_shape():
    """Scale tensor shape must be (rows, n_groups)."""
    w = np.random.randn(32, 256).astype(np.float32)
    _, scales = quantize_sym(w, group_size=64)
    assert scales.shape == (32, 4), f"Expected (32,4), got {scales.shape}"

    _, scales128 = quantize_sym(w, group_size=128)
    assert scales128.shape == (32, 2)


def test_sym_zero_weight():
    """All-zero weight should round-trip exactly (scale = 1 fallback)."""
    w = np.zeros((8, 128), dtype=np.float32)
    q, scales = quantize_sym(w)
    w_r = dequantize_sym(q, scales)
    np.testing.assert_array_equal(q, np.zeros_like(q))
    np.testing.assert_allclose(w_r, w, atol=1e-6)


# ---------------------------------------------------------------------------
# Asymmetric round-trip
# ---------------------------------------------------------------------------

def test_asym_round_trip_values():
    """Asymmetric quantize → dequantize should be close to original."""
    rng = np.random.default_rng(7)
    w = rng.normal(2.0, 1.0, (64, 128)).astype(np.float32)   # non-zero mean
    q, scales, zeros = quantize_asym(w, group_size=128)
    w_r = dequantize_asym(q, scales, zeros, group_size=128)

    max_scale = float(scales.max())
    max_err = float(np.abs(w - w_r).max())
    print(f"\n  asym round-trip: max_err={max_err:.5f}, max_scale={max_scale:.5f}")
    assert max_err <= max_scale + 1e-5, f"Max error {max_err} > max_scale {max_scale}"


def test_asym_q_range():
    """Asymmetric q values must sit in [0, 15]."""
    rng = np.random.default_rng(0)
    w = rng.uniform(-5, 5, (32, 128)).astype(np.float32)
    q, _, _ = quantize_asym(w, group_size=128)
    assert int(q.min()) >= 0 and int(q.max()) <= 15


def test_asym_scales_shape():
    """Scale and zero tensors must both have shape (rows, n_groups)."""
    w = np.random.randn(16, 256).astype(np.float32)
    _, scales, zeros = quantize_asym(w, group_size=128)
    assert scales.shape == (16, 2)
    assert zeros.shape == (16, 2)


def test_asym_better_than_sym_skewed():
    """Asymmetric should give lower RMSE on heavily-skewed weight distributions."""
    rng = np.random.default_rng(99)
    w = rng.uniform(0, 4, (32, 128)).astype(np.float32)   # all positive

    w_sym = fake_quantize(w, group_size=128, sym=True)
    w_asym = fake_quantize(w, group_size=128, sym=False)

    rmse_sym = float(np.sqrt(np.mean((w - w_sym) ** 2)))
    rmse_asym = float(np.sqrt(np.mean((w - w_asym) ** 2)))
    print(f"\n  skewed: sym RMSE={rmse_sym:.5f}, asym RMSE={rmse_asym:.5f}")
    assert rmse_asym < rmse_sym, "Asymmetric should beat symmetric on skewed data"


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------

def test_pack_unpack_signed():
    """Pack → unpack must recover all signed int4 values in [-8, 7]."""
    rng = np.random.default_rng(7)
    original = rng.integers(-8, 8, size=(64,), dtype=np.int8)
    packed = pack_int4(original)
    assert packed.shape == (32,), f"Expected (32,), got {packed.shape}"
    recovered = unpack_int4(packed, n=64, signed=True)
    np.testing.assert_array_equal(original, recovered)


def test_pack_unpack_unsigned():
    """Pack → unpack must recover all unsigned int4 values in [0, 15]."""
    rng = np.random.default_rng(13)
    original = rng.integers(0, 16, size=(128,), dtype=np.uint8)
    packed = pack_int4(original)
    assert packed.shape == (64,)
    recovered = unpack_int4(packed, n=128, signed=False)
    np.testing.assert_array_equal(original, recovered)


def test_pack_unpack_boundary_values():
    """Boundary int4 values -8 and 7 must survive a round-trip."""
    values = np.array([-8, -1, 0, 1, 7, -8, 7, -1], dtype=np.int8)
    packed = pack_int4(values)
    recovered = unpack_int4(packed, n=8, signed=True)
    np.testing.assert_array_equal(values, recovered)


def test_pack_odd_count_raises():
    """pack_int4 must raise for an odd-length input."""
    with pytest.raises(ValueError, match="even"):
        pack_int4(np.array([1, 2, 3], dtype=np.int8))


def test_pack_size_halved():
    """Packed array must be exactly half the size of the input."""
    q = np.zeros(256, dtype=np.int8)
    packed = pack_int4(q)
    assert packed.size == 128


# ---------------------------------------------------------------------------
# Fake-quant / group-size sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_fake_quantize_group_sizes_sym(group_size):
    """fake_quantize should run without error for each standard group size."""
    rng = np.random.default_rng(123)
    w = rng.normal(0, 1, (64, 256)).astype(np.float32)
    w_q = fake_quantize(w, group_size=group_size, sym=True)

    assert w_q.shape == w.shape
    rmse = float(np.sqrt(np.mean((w - w_q) ** 2)))
    print(f"\n  sym group_size={group_size}: RMSE={rmse:.5f}")
    assert rmse < 0.25 * float(w.std()), f"group_size={group_size}: RMSE too large"


def test_group_size_ordering():
    """Smaller group size must yield strictly lower quantization RMSE."""
    rng = np.random.default_rng(456)
    w = rng.normal(0, 1, (64, 256)).astype(np.float32)

    errors: dict[int, float] = {}
    for gs in [256, 128, 64, 32]:
        w_q = fake_quantize(w, group_size=gs, sym=True)
        errors[gs] = float(np.sqrt(np.mean((w - w_q) ** 2)))

    print(f"\n  group size → RMSE: {errors}")
    assert errors[256] > errors[128] > errors[64] > errors[32], (
        f"Expected strictly decreasing RMSE but got: {errors}"
    )


def test_fake_quantize_shape_preserved():
    """fake_quantize must return same shape and dtype as input."""
    w = np.random.randn(16, 128).astype(np.float32)
    w_q = fake_quantize(w, group_size=128, sym=True)
    assert w_q.shape == w.shape
    assert w_q.dtype == np.float32


def test_group_size_not_divisible_raises():
    """quantize_sym should raise cleanly when cols is not divisible by group_size."""
    w = np.random.randn(8, 100).astype(np.float32)
    with pytest.raises(ValueError, match="not divisible"):
        quantize_sym(w, group_size=128)
