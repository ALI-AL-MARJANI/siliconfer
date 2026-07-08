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
    quantize_sym_int2,
    dequantize_sym_int2,
    quantize_asym_int2,
    dequantize_asym_int2,
    pack_int2,
    unpack_int2,
    quantize_sym_n,
    quantize_asym_n,
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


# ---------------------------------------------------------------------------
# int2 (mixed-precision groundwork — see quant/mixed_precision.py)
# ---------------------------------------------------------------------------

def test_sym_int2_q_range():
    """Symmetric int2 q values must sit in [-2, 1]."""
    rng = np.random.default_rng(0)
    w = rng.normal(0, 10, (32, 128)).astype(np.float32)
    q, _ = quantize_sym_int2(w, group_size=128)
    assert int(q.min()) >= -2 and int(q.max()) <= 1


def test_sym_int2_zero_weight():
    """All-zero weight should round-trip exactly under int2 too."""
    w = np.zeros((8, 128), dtype=np.float32)
    q, scales = quantize_sym_int2(w)
    w_r = dequantize_sym_int2(q, scales)
    np.testing.assert_array_equal(q, np.zeros_like(q))
    np.testing.assert_allclose(w_r, w, atol=1e-6)


def test_asym_int2_q_range():
    """Asymmetric int2 q values must sit in [0, 3]."""
    rng = np.random.default_rng(0)
    w = rng.uniform(-5, 5, (32, 128)).astype(np.float32)
    q, _, _ = quantize_asym_int2(w, group_size=128)
    assert int(q.min()) >= 0 and int(q.max()) <= 3


def test_int2_much_lossier_than_int4():
    """int2 must have strictly higher reconstruction error than int4 on the same data.

    This is the whole point of int2: it's a knowingly worse fallback, only meant
    to be assigned to layers a sensitivity estimate says can tolerate it. If this
    ever stopped holding, the mixed-precision memory/accuracy tradeoff would be
    nonsensical (paying nothing for the extra error).
    """
    rng = np.random.default_rng(42)
    w = rng.normal(0, 1, (64, 128)).astype(np.float32)
    w_int4 = fake_quantize(w, group_size=128, sym=True, bits=4)
    w_int2 = fake_quantize(w, group_size=128, sym=True, bits=2)

    rmse_int4 = float(np.sqrt(np.mean((w - w_int4) ** 2)))
    rmse_int2 = float(np.sqrt(np.mean((w - w_int2) ** 2)))
    print(f"\n  int4 RMSE={rmse_int4:.5f}, int2 RMSE={rmse_int2:.5f}")
    assert rmse_int2 > rmse_int4


def test_fake_quantize_invalid_bits_raises():
    """bits outside [2, 8] must raise — bits=3 itself is now valid (see
    quantize_sym_n/quantize_asym_n), unlike the earlier bits-in-{2,4}-only
    version of fake_quantize."""
    w = np.random.randn(8, 128).astype(np.float32)
    with pytest.raises(ValueError, match="bits must be"):
        fake_quantize(w, bits=1)
    with pytest.raises(ValueError, match="bits must be"):
        fake_quantize(w, bits=9)


def test_fake_quantize_bits_3_works_and_is_between_2_and_4_in_error():
    """bits=3 (8-level grid) should reconstruct better than bits=2 (4-level)
    and worse than bits=4 (16-level) — a basic monotonicity sanity check for
    the generalized arbitrary-bit-width quantizer."""
    rng = np.random.default_rng(11)
    w = rng.normal(0, 1, (32, 128)).astype(np.float32)

    w_2 = fake_quantize(w, group_size=128, sym=True, bits=2)
    w_3 = fake_quantize(w, group_size=128, sym=True, bits=3)
    w_4 = fake_quantize(w, group_size=128, sym=True, bits=4)

    rmse_2 = float(np.sqrt(np.mean((w - w_2) ** 2)))
    rmse_3 = float(np.sqrt(np.mean((w - w_3) ** 2)))
    rmse_4 = float(np.sqrt(np.mean((w - w_4) ** 2)))
    print(f"\n  bits=2 RMSE={rmse_2:.5f}, bits=3 RMSE={rmse_3:.5f}, bits=4 RMSE={rmse_4:.5f}")
    assert rmse_2 > rmse_3 > rmse_4


def test_quantize_sym_n_matches_quantize_sym_at_bits_4():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 1, (16, 128)).astype(np.float32)
    q_n, scales_n = quantize_sym_n(w, group_size=128, bits=4)
    q_ref, scales_ref = quantize_sym(w, group_size=128)
    np.testing.assert_array_equal(q_n, q_ref)
    np.testing.assert_allclose(scales_n, scales_ref)


def test_quantize_sym_n_matches_quantize_sym_int2_at_bits_2():
    rng = np.random.default_rng(2)
    w = rng.normal(0, 1, (16, 128)).astype(np.float32)
    q_n, scales_n = quantize_sym_n(w, group_size=128, bits=2)
    q_ref, scales_ref = quantize_sym_int2(w, group_size=128)
    np.testing.assert_array_equal(q_n, q_ref)
    np.testing.assert_allclose(scales_n, scales_ref)


def test_quantize_asym_n_matches_quantize_asym_at_bits_4():
    rng = np.random.default_rng(3)
    w = rng.normal(2.0, 1.0, (16, 128)).astype(np.float32)
    q_n, scales_n, zeros_n = quantize_asym_n(w, group_size=128, bits=4)
    q_ref, scales_ref, zeros_ref = quantize_asym(w, group_size=128)
    np.testing.assert_array_equal(q_n, q_ref)
    np.testing.assert_allclose(scales_n, scales_ref)
    np.testing.assert_allclose(zeros_n, zeros_ref)


def test_quantize_sym_n_bits_out_of_range_raises():
    w = np.random.randn(8, 128).astype(np.float32)
    with pytest.raises(ValueError, match="bits must be in"):
        quantize_sym_n(w, bits=1)
    with pytest.raises(ValueError, match="bits must be in"):
        quantize_sym_n(w, bits=9)


def test_pack_unpack_int2_signed():
    """Pack → unpack must recover all signed int2 values in [-2, 1]."""
    rng = np.random.default_rng(7)
    original = rng.integers(-2, 2, size=(64,), dtype=np.int8)
    packed = pack_int2(original)
    assert packed.shape == (16,), f"Expected (16,), got {packed.shape}"
    recovered = unpack_int2(packed, n=64, signed=True)
    np.testing.assert_array_equal(original, recovered)


def test_pack_unpack_int2_unsigned():
    """Pack → unpack must recover all unsigned int2 values in [0, 3]."""
    rng = np.random.default_rng(13)
    original = rng.integers(0, 4, size=(128,), dtype=np.uint8)
    packed = pack_int2(original)
    assert packed.shape == (32,)
    recovered = unpack_int2(packed, n=128, signed=False)
    np.testing.assert_array_equal(original, recovered)


def test_pack_unpack_int2_boundary_values():
    """Boundary int2 values -2 and 1 must survive a round-trip."""
    values = np.array([-2, -1, 0, 1, -2, 1, 0, -1], dtype=np.int8)
    packed = pack_int2(values)
    recovered = unpack_int2(packed, n=8, signed=True)
    np.testing.assert_array_equal(values, recovered)


def test_pack_int2_not_divisible_by_4_raises():
    with pytest.raises(ValueError, match="divisible by 4"):
        pack_int2(np.array([1, 2, 3], dtype=np.int8))


def test_pack_int2_size_quartered():
    """Packed int2 array must be exactly a quarter of the input size."""
    q = np.zeros(256, dtype=np.int8)
    packed = pack_int2(q)
    assert packed.size == 64


def test_int2_asym_round_trip_close():
    """Asymmetric int2 quantize -> dequantize error must be bounded by the group scale."""
    rng = np.random.default_rng(7)
    w = rng.normal(2.0, 1.0, (64, 128)).astype(np.float32)
    q, scales, zeros = quantize_asym_int2(w, group_size=128)
    w_r = dequantize_asym_int2(q, scales, zeros, group_size=128)

    max_scale = float(scales.max())
    max_err = float(np.abs(w - w_r).max())
    assert max_err <= max_scale + 1e-5, f"Max error {max_err} > max_scale {max_scale}"
