// q4_gemm.cpp — q4 GEMM for prefill (T > 1).
//
// Prefill is less bandwidth-bound than decode (larger batch amortises weight
// reads over many tokens), but still benefits from reduced weight traffic.
//
// Strategy: outer loop over tokens T, inner loop calls q4_gemv for each.
// This is correct and simple; a cache-tiled GEMM would be faster for large T
// but is deferred to Phase 8 (stretch).

#include "q4_gemv.h"

void q4_gemm_sym_neon(
    const uint8_t* __restrict__ W,      // [out_f, in_f/2]
    const float*   __restrict__ scales, // [out_f, n_groups]
    const float*   __restrict__ X,      // [T, in_f]
    float*         __restrict__ Y,      // [T, out_f]
    int out_f, int in_f, int T, int group_size
) {
    for (int t = 0; t < T; t++) {
        q4_gemv_sym_neon(
            W, scales,
            X + (size_t)t * in_f,     // x_t: row t of X
            Y + (size_t)t * out_f,    // y_t: row t of Y
            out_f, in_f, group_size
        );
    }
}
