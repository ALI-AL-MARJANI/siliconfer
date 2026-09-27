"""Q4Linear: replacement for nn.Linear backed by packed int4 weights."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from siliconfer.kernels.neon import gemm_asym, gemm_sym


def to_mlx_quantized(
    packed: np.ndarray, scales: np.ndarray, zeros: np.ndarray | None = None
) -> tuple[mx.array, mx.array, mx.array]:
    """Convert packed int4 weights to MLX's quantized layout, without re-quantizing.

    MLX stores w = scale·u + bias with unsigned 4-bit u, eight per uint32,
    lowest bits first, which is this repo's byte layout viewed as uint32:

      asymmetric: (u − zero)·scale                  → bias = −zero·scale
      symmetric:  s·scale, s two's complement;
                  u = s + 8 = nibble XOR 8          → bias = −8·scale
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.shape[1] % 4 != 0:
        raise ValueError("in_features must be a multiple of 8 for the MLX backend")
    scales = np.asarray(scales, dtype=np.float32)
    if zeros is None:
        codes, biases = packed ^ 0x88, -8.0 * scales
    else:
        codes, biases = packed, -np.asarray(zeros, dtype=np.float32) * scales
    return mx.array(codes.view(np.uint32)), mx.array(scales), mx.array(biases.astype(np.float32))


class Q4Linear(nn.Module):
    """Linear layer over packed int4 weights.

    backend="neon": the weights are held as NumPy arrays and multiplied by the
    C++/NEON kernel on the CPU.

      _packed       uint8   [out_features, in_features // 2]
      _scales       float32 [out_features, n_groups]
      _zeros        float32 [out_features, n_groups], or None for a symmetric grid
      _input_scale  float32 [in_features], or None

    backend="mlx": the same codes are converted once to MLX's layout and
    multiplied on the GPU with `mx.quantized_matmul`, inside the lazy graph.

    `bias` is the only MLX parameter. `input_scale`, when set, multiplies the
    input before the product; AWQ and SINQ store Q(W·diag(s)) and pass 1/s here
    (docs/packing.md).
    """

    def __init__(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray | None = None,
        bias: mx.array | None = None,
        group_size: int = 128,
        input_scale: np.ndarray | None = None,
        backend: str = "neon",
    ) -> None:
        super().__init__()
        if backend not in ("neon", "mlx"):
            raise ValueError(f"Unknown backend {backend!r}. Choose 'neon' or 'mlx'.")
        self._backend = backend
        self._input_scale = (
            np.ascontiguousarray(input_scale, dtype=np.float32) if input_scale is not None else None
        )
        self.bias = bias
        self.group_size = group_size
        self.out_features = packed.shape[0]
        self.in_features = packed.shape[1] * 2

        if backend == "neon":
            self._packed = packed
            self._scales = scales
            self._zeros = zeros
        else:
            # Same codes in MLX's layout (see `to_mlx_quantized`); only the GPU
            # copy is kept. Underscore attributes stay out of the parameter tree.
            w, sc, b = to_mlx_quantized(packed, scales, zeros)
            self._packed = self._scales = self._zeros = None
            self._mx_w, self._mx_scales, self._mx_biases = w, sc, b
            self._mx_input_scale = (
                mx.array(self._input_scale) if self._input_scale is not None else None
            )
            mx.eval(self._mx_w, self._mx_scales, self._mx_biases)

    @property
    def nbytes(self) -> int:
        """Bytes held for the quantized weight (codes + per-group scales/zeros)."""
        if self._backend == "neon":
            n = self._packed.nbytes + self._scales.nbytes
            return n + (self._zeros.nbytes if self._zeros is not None else 0)
        return self._mx_w.nbytes + self._mx_scales.nbytes + self._mx_biases.nbytes

    def __call__(self, x: mx.array) -> mx.array:
        if self._backend == "mlx":
            # Stays in the lazy MLX graph: no evaluation, no CPU round trip.
            x = x.astype(mx.float32)
            if self._mx_input_scale is not None:
                x = x * self._mx_input_scale
            y = mx.quantized_matmul(x, self._mx_w, self._mx_scales, self._mx_biases,
                                    transpose=True, group_size=self.group_size, bits=4)
            return y + self.bias if self.bias is not None else y

        # np.array() forces MLX evaluation — explicit for clarity
        mx.eval(x)

        orig_shape = x.shape
        # [..., in_f] → [T, in_f]
        x_np = np.array(x.reshape(-1, self.in_features).astype(mx.float32))

        if self._input_scale is not None:
            x_np = x_np * self._input_scale[None, :]

        # NEON GEMM: Y[T, out_f] = X[T, in_f] @ W_q4.T
        if self._zeros is not None:
            y_np = gemm_asym(self._packed, self._scales, self._zeros, x_np, self.group_size)
        else:
            y_np = gemm_sym(self._packed, self._scales, x_np, self.group_size)

        y = mx.array(y_np).reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            y = y + self.bias

        return y
