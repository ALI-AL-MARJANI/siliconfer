// q4_gemm.cpp — q4 GEMM for prefill (T > 1): Y[T, out_f] = X[T, in_f] · dequant(W)ᵀ.
//
// Unlike decode, prefill is compute-bound: every weight is used T times, so
// the cost of reading it is amortised and the float multiply-adds dominate.
// The kernel therefore works in row tiles:
//
//   1. dequantize a tile of weight rows to float32 with NEON (a tile is sized
//      to stay in L2), then
//   2. multiply X against that tile with the platform BLAS (Accelerate sgemm).
//
// Weights stay packed in memory; only one tile is ever expanded. For very
// small T the tile product is not worth it and the GEMV kernel is called per
// token instead.

#include "q4_gemv.h"

#include <algorithm>
#include <vector>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define Q4_HAVE_BLAS 1
#endif

// Below this many tokens, call the GEMV kernel per token.
static const int kMinTileT = 4;
// Float32 elements per dequantized tile (1 MB).
static const int kTileFloats = 1 << 18;

static void gemm_impl(
    const uint8_t* W, const float* scales, const float* zeros,
    const float* X, float* Y,
    int out_f, int in_f, int T, int group_size
) {
#ifdef Q4_HAVE_BLAS
    if (T >= kMinTileT) {
        int tile_rows = std::max(1, std::min(out_f, kTileFloats / in_f));
        static thread_local std::vector<float> tile;
        tile.resize((size_t)tile_rows * in_f);

        for (int r0 = 0; r0 < out_f; r0 += tile_rows) {
            int r1 = std::min(out_f, r0 + tile_rows);
            q4_dequant_rows(W, scales, zeros, tile.data(), r0, r1, in_f, group_size);
            // Y[:, r0:r1] = X · tileᵀ
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, r1 - r0, in_f,
                        1.0f, X, in_f, tile.data(), in_f,
                        0.0f, Y + r0, out_f);
        }
        return;
    }
#endif
    for (int t = 0; t < T; t++) {
        const float* x = X + (size_t)t * in_f;
        float* y = Y + (size_t)t * out_f;
        if (zeros) q4_gemv_asym_neon(W, scales, zeros, x, y, out_f, in_f, group_size);
        else       q4_gemv_sym_neon(W, scales, x, y, out_f, in_f, group_size);
    }
}

void q4_gemm_sym_neon(
    const uint8_t* __restrict__ W,      // [out_f, in_f/2]
    const float*   __restrict__ scales, // [out_f, n_groups]
    const float*   __restrict__ X,      // [T, in_f]
    float*         __restrict__ Y,      // [T, out_f]
    int out_f, int in_f, int T, int group_size
) {
    gemm_impl(W, scales, nullptr, X, Y, out_f, in_f, T, group_size);
}

void q4_gemm_asym_neon(
    const uint8_t* __restrict__ W,      // [out_f, in_f/2]
    const float*   __restrict__ scales, // [out_f, n_groups]
    const float*   __restrict__ zeros,  // [out_f, n_groups]
    const float*   __restrict__ X,      // [T, in_f]
    float*         __restrict__ Y,      // [T, out_f]
    int out_f, int in_f, int T, int group_size
) {
    gemm_impl(W, scales, zeros, X, Y, out_f, in_f, T, group_size);
}
