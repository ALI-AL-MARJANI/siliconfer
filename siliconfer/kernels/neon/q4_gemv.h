#pragma once
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// 4-bit weight kernels
//
// Weight layout: W_packed[out_f, in_f/2]
//   byte j in row i holds: lo nibble = w(i, 2j), hi nibble = w(i, 2j+1)
//   symmetric:  nibble is two's complement in [0,15]; 0..7 map to 0..7,
//               8..15 map to -8..-1; value = nibble_signed * scale
//   asymmetric: nibble is unsigned in [0,15]; value = (nibble - zero) * scale
//   (matches pack_int4 from siliconfer/quant/primitives.py)
//
// scales[out_f, n_groups] (and zeros, same shape), n_groups = in_f / group_size
// ---------------------------------------------------------------------------

// Number of threads used by the GEMV kernels and by row dequantization.
// Defaults to the number of performance cores. 1 disables threading.
// Small products stay single-threaded regardless (dispatch would cost more
// than the work).
void q4_set_num_threads(int n);
int  q4_get_num_threads();

// GEMV (batch=1 decode): y = dequant(W) · x
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

void q4_gemv_asym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
);

// Dequantize rows [r0, r1) to float32: out[(r - r0) * in_f + c].
// zeros == nullptr selects the symmetric encoding.
void q4_dequant_rows(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ out,
    int r0, int r1, int in_f, int group_size
);

// GEMM (prefill): X[T, in_f] → Y[T, out_f]
void q4_gemm_sym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int out_f, int in_f, int T, int group_size
);

void q4_gemm_asym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    const float*   __restrict__ X,
    float*         __restrict__ Y,
    int out_f, int in_f, int T, int group_size
);
