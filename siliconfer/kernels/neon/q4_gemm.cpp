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

// q4_gemm_asym_neon — asymmetric counterpart, same T-loop-over-gemv strategy.
// Added to close the gap documented in CLAUDE.md §9: Q4Linear previously had
// no way to run prefill (T>1) on asymmetrically-quantized weights at all,
// forcing every method to be silently re-packed as symmetric regardless of
// how it was actually quantized (harmless for RTN/GPTQ/AWQ's default
// sym=True, but a real accuracy bug for HQQ, which is always asymmetric).
void q4_gemm_asym_neon(
    const uint8_t* __restrict__ W,      // [out_f, in_f/2]
    const float*   __restrict__ scales, // [out_f, n_groups]
    const float*   __restrict__ zeros,  // [out_f, n_groups]
    const float*   __restrict__ X,      // [T, in_f]
    float*         __restrict__ Y,      // [T, out_f]
    int out_f, int in_f, int T, int group_size
) {
    for (int t = 0; t < T; t++) {
        q4_gemv_asym_neon(
            W, scales, zeros,
            X + (size_t)t * in_f,
            Y + (size_t)t * out_f,
            out_f, in_f, group_size
        );
    }
}
