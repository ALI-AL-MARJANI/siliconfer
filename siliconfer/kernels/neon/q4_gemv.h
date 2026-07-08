#pragma once
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// Symmetric q4 GEMV / GEMM
//
// Weight layout: W_packed[out_f, in_f/2]
//   byte j in row i holds: lo nibble = w(i, 2j), hi nibble = w(i, 2j+1)
//   nibble encoding: two's complement in [0,15]; values 0..7 map to 0..7,
//   values 8..15 map to -8..-1 (dequant sign-extends the nibble, then * scale)
//   (matches pack_int4 from siliconfer/quant/primitives.py)
//
// scales[out_f, n_groups], n_groups = in_f / group_size
// ---------------------------------------------------------------------------

// NEON-vectorised GEMV (batch=1 decode)
void q4_gemv_sym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
);

// Scalar reference (used for correctness tests and non-NEON fallback)
void q4_gemv_sym_scalar(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
);

// Asymmetric q4 GEMV (adds per-group zero-point)
void q4_gemv_asym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
);

// GEMM (prefill): X[T, in_f] → Y[T, out_f]
void q4_gemm_sym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int out_f, int in_f, int T, int group_size
);

// GEMM (prefill), asymmetric: X[T, in_f] → Y[T, out_f]
void q4_gemm_asym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int out_f, int in_f, int T, int group_size
);
