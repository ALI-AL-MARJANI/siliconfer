"""Q4Linear: drop-in nn.Module replacement for nn.Linear backed by the NEON int4 kernel."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from siliconfer.kernels.neon import gemm_sym, gemm_asym


class Q4Linear(nn.Module):
    """MLX module wrapping packed int4 weights + NEON GEMM kernel.

    Stored layout:
      _packed  — uint8 [out_features, in_features // 2]  (numpy, not mx.array)
      _scales  — float32 [out_features, n_groups]         (numpy, not mx.array)
      _zeros   — float32 [out_features, n_groups] or None  (numpy, not mx.array)
      bias     — mx.array [out_features] or None

    Numpy arrays are invisible to MLX's parameter tree; only bias is tracked by MLX.

    `zeros=None` (the default) selects the symmetric NEON kernel. Passing
    `zeros` selects the asymmetric kernel instead — this is required for
    correctness with any method whose fake-quantized weights sit on an
    asymmetric grid (HQQ always; RTN/GPTQ/AWQ when run with `sym=False`).
    Packing an asymmetric weight with the symmetric kernel silently
    re-quantizes it onto the wrong grid — see CLAUDE.md §9 for the real bug
    this caused before `zeros` support existed here.
    """

    def __init__(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray | None = None,
        bias: mx.array | None = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self._packed = packed
        self._scales = scales
        self._zeros = zeros
        self.bias = bias
        self.group_size = group_size
        self.out_features = packed.shape[0]
        self.in_features = packed.shape[1] * 2

    def __call__(self, x: mx.array) -> mx.array:
        # np.array() forces MLX evaluation — explicit for clarity
        mx.eval(x)

        orig_shape = x.shape
        # [..., in_f] → [T, in_f]
        x_np = np.array(x.reshape(-1, self.in_features).astype(mx.float32))

        # NEON GEMM: Y[T, out_f] = X[T, in_f] @ W_q4.T
        if self._zeros is not None:
            y_np = gemm_asym(self._packed, self._scales, self._zeros, x_np, self.group_size)
        else:
            y_np = gemm_sym(self._packed, self._scales, x_np, self.group_size)

        y = mx.array(y_np).reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            y = y + self.bias

        return y
