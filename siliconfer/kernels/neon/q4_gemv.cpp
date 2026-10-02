// q4_gemv.cpp — hand-written NEON 4-bit GEMV for Apple Silicon.
//
// GEMV (y = W_q4 x) is the decode kernel: one product per linear layer per
// generated token.
//
// Inner loop (per output row, per group, 16 packed bytes = 32 weights at a time):
//   1. Load 16 bytes. Sign-extend both nibbles with shifts:
//        hi = asr(byte, 4)            lo = asr(lsl(byte, 4), 4)
//   2. Interleave lo/hi (zip) so the 32 int8 weights are in column order —
//      cheaper than de-interleaving the 32 floats of x for every row.
//   3. Widen int8 → int16 → int32 → float32 (8 vectors of 4).
//   4. FMA each vector against x into its own accumulator. Eight independent
//      accumulators keep the loop throughput-bound rather than bound by the
//      latency of one FMA chain.
//   Reduce the accumulators, multiply by the group scale, add to y[i].
//
// Rows are independent, so large products are split across threads by row
// range (Grand Central Dispatch). The result for a row does not depend on the
// thread count.
//
// Group sizes that are not a multiple of 32 fall through to a scalar tail.

#include "q4_gemv.h"

#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

// Below this many weights a product runs on the calling thread: waking worker
// threads costs more than the product itself.
static const size_t kMinParallelWeights = 1u << 19;

static int default_num_threads() {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
#endif
    return 1;
}

static int g_num_threads = default_num_threads();

void q4_set_num_threads(int n) { g_num_threads = std::max(1, n); }
int  q4_get_num_threads()      { return g_num_threads; }

namespace {

struct RowJob {
    void (*fn)(void* ctx, int r0, int r1);
    void* ctx;
    int n_rows;
    int n_chunks;
};

void run_chunk(void* p, size_t c) {
    const RowJob* job = static_cast<const RowJob*>(p);
    int r0 = (int)((long long)c * job->n_rows / job->n_chunks);
    int r1 = (int)((long long)(c + 1) * job->n_rows / job->n_chunks);
    job->fn(job->ctx, r0, r1);
}

// Run fn(ctx, r0, r1) over [0, n_rows), split across threads when worthwhile.
void parallel_rows(int n_rows, size_t n_weights, void (*fn)(void*, int, int), void* ctx) {
#ifdef __APPLE__
    int n_chunks = std::min(g_num_threads, n_rows);
    if (n_chunks > 1 && n_weights >= kMinParallelWeights) {
        RowJob job{fn, ctx, n_rows, n_chunks};
        dispatch_apply_f((size_t)n_chunks,
                         dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                         &job, run_chunk);
        return;
    }
#endif
    fn(ctx, 0, n_rows);
}

struct GemvCtx {
    const uint8_t* W;
    const float* scales;
    const float* zeros;     // nullptr for symmetric
    const float* x;
    const float* x_gsum;    // per-group sum of x (asymmetric only)
    float* y;
    int in_f;
    int group_size;
};

struct DequantCtx {
    const uint8_t* W;
    const float* scales;
    const float* zeros;
    float* out;
    int r_base;
    int in_f;
    int group_size;
};

}  // namespace

#ifdef __ARM_NEON
#include <arm_neon.h>

namespace {

// 16 packed bytes → 32 weights as 8 float32x4, in column order.
// Symmetric: nibbles are two's complement.
inline void unpack32_sym(const uint8_t* p, float32x4_t f[8]) {
    int8x16_t b  = vreinterpretq_s8_u8(vld1q_u8(p));
    int8x16_t hi = vshrq_n_s8(b, 4);
    int8x16_t lo = vshrq_n_s8(vshlq_n_s8(b, 4), 4);
    int8x16_t wa = vzip1q_s8(lo, hi);     // columns 0..15
    int8x16_t wb = vzip2q_s8(lo, hi);     // columns 16..31
    int16x8_t a0 = vmovl_s8(vget_low_s8(wa)), a1 = vmovl_high_s8(wa);
    int16x8_t b0 = vmovl_s8(vget_low_s8(wb)), b1 = vmovl_high_s8(wb);
    f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0)));  f[1] = vcvtq_f32_s32(vmovl_high_s16(a0));
    f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1)));  f[3] = vcvtq_f32_s32(vmovl_high_s16(a1));
    f[4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b0)));  f[5] = vcvtq_f32_s32(vmovl_high_s16(b0));
    f[6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b1)));  f[7] = vcvtq_f32_s32(vmovl_high_s16(b1));
}

// Asymmetric: nibbles are unsigned [0, 15].
inline void unpack32_asym(const uint8_t* p, float32x4_t f[8]) {
    uint8x16_t b  = vld1q_u8(p);
    uint8x16_t hi = vshrq_n_u8(b, 4);
    uint8x16_t lo = vandq_u8(b, vdupq_n_u8(0x0F));
    uint8x16_t wa = vzip1q_u8(lo, hi);
    uint8x16_t wb = vzip2q_u8(lo, hi);
    uint16x8_t a0 = vmovl_u8(vget_low_u8(wa)), a1 = vmovl_high_u8(wa);
    uint16x8_t b0 = vmovl_u8(vget_low_u8(wb)), b1 = vmovl_high_u8(wb);
    f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a0)));  f[1] = vcvtq_f32_u32(vmovl_high_u16(a0));
    f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a1)));  f[3] = vcvtq_f32_u32(vmovl_high_u16(a1));
    f[4] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b0)));  f[5] = vcvtq_f32_u32(vmovl_high_u16(b0));
    f[6] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b1)));  f[7] = vcvtq_f32_u32(vmovl_high_u16(b1));
}

inline float nibble_sym(uint8_t u) { return (float)((u < 8) ? (int)u : (int)u - 16); }

// Σ_c nibble(c) · x[c] over one group; `sym` selects the nibble decoding.
template <bool SYM>
inline float group_dot(const uint8_t* wg, const float* xg, int half_gs) {
    int n_vec = half_gs / 16;
    float32x4_t acc[8];
    for (int k = 0; k < 8; k++) acc[k] = vdupq_n_f32(0.0f);

    for (int b = 0; b < n_vec; b++) {
        float32x4_t f[8];
        if (SYM) unpack32_sym(wg + 16 * b, f); else unpack32_asym(wg + 16 * b, f);
        const float* xp = xg + 32 * b;
        for (int k = 0; k < 8; k++) acc[k] = vfmaq_f32(acc[k], f[k], vld1q_f32(xp + 4 * k));
    }

    float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])),
                              vaddq_f32(vaddq_f32(acc[4], acc[5]), vaddq_f32(acc[6], acc[7])));
    float dot = vaddvq_f32(s);

    for (int c = n_vec * 16; c < half_gs; c++) {
        uint8_t byte = wg[c];
        float lo = SYM ? nibble_sym(byte & 0x0F) : (float)(byte & 0x0F);
        float hi = SYM ? nibble_sym(byte >> 4)   : (float)(byte >> 4);
        dot += lo * xg[2 * c] + hi * xg[2 * c + 1];
    }
    return dot;
}

void gemv_rows(void* p, int r0, int r1) {
    const GemvCtx& c = *static_cast<const GemvCtx*>(p);
    int n_groups = c.in_f / c.group_size;
    int half_gs  = c.group_size / 2;

    for (int i = r0; i < r1; i++) {
        const uint8_t* w_row = c.W + (size_t)i * (c.in_f / 2);
        const float*   s_row = c.scales + (size_t)i * n_groups;
        float y_val = 0.0f;

        if (c.zeros == nullptr) {
            for (int g = 0; g < n_groups; g++)
                y_val += group_dot<true>(w_row + g * half_gs, c.x + g * c.group_size, half_gs)
                         * s_row[g];
        } else {
            // (nibble − zero)·scale · x  =  scale · (Σ nibble·x − zero · Σ x)
            const float* z_row = c.zeros + (size_t)i * n_groups;
            for (int g = 0; g < n_groups; g++) {
                float dot = group_dot<false>(w_row + g * half_gs, c.x + g * c.group_size, half_gs);
                y_val += (dot - z_row[g] * c.x_gsum[g]) * s_row[g];
            }
        }
        c.y[i] = y_val;
    }
}

void dequant_rows(void* p, int r0, int r1) {
    const DequantCtx& c = *static_cast<const DequantCtx*>(p);
    int n_groups = c.in_f / c.group_size;
    int half_gs  = c.group_size / 2;
    int n_vec    = half_gs / 16;

    for (int i = r0; i < r1; i++) {
        const uint8_t* w_row = c.W + (size_t)i * (c.in_f / 2);
        const float*   s_row = c.scales + (size_t)i * n_groups;
        const float*   z_row = c.zeros ? c.zeros + (size_t)i * n_groups : nullptr;
        float* o_row = c.out + (size_t)(i - c.r_base) * c.in_f;

        for (int g = 0; g < n_groups; g++) {
            const uint8_t* wg = w_row + g * half_gs;
            float* og = o_row + g * c.group_size;
            float sc = s_row[g];
            float zp = z_row ? z_row[g] : 0.0f;
            float32x4_t vsc = vdupq_n_f32(sc), vzp = vdupq_n_f32(zp);

            for (int b = 0; b < n_vec; b++) {
                float32x4_t f[8];
                if (z_row) unpack32_asym(wg + 16 * b, f); else unpack32_sym(wg + 16 * b, f);
                for (int k = 0; k < 8; k++)
                    vst1q_f32(og + 32 * b + 4 * k, vmulq_f32(vsubq_f32(f[k], vzp), vsc));
            }
            for (int cb = n_vec * 16; cb < half_gs; cb++) {
                uint8_t byte = wg[cb];
                float lo = z_row ? (float)(byte & 0x0F) : nibble_sym(byte & 0x0F);
                float hi = z_row ? (float)(byte >> 4)   : nibble_sym(byte >> 4);
                og[2 * cb]     = (lo - zp) * sc;
                og[2 * cb + 1] = (hi - zp) * sc;
            }
        }
    }
}

}  // namespace

void q4_gemv_sym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
) {
    GemvCtx ctx{W, scales, nullptr, x, nullptr, y, in_f, group_size};
    parallel_rows(out_f, (size_t)out_f * in_f, gemv_rows, &ctx);
}

void q4_gemv_asym_neon(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
) {
    // Σ x per group is the same for every row: compute it once.
    int n_groups = in_f / group_size;
    std::vector<float> x_gsum(n_groups);
    for (int g = 0; g < n_groups; g++) {
        const float* xg = x + g * group_size;
        float32x4_t s = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 4 <= group_size; c += 4) s = vaddq_f32(s, vld1q_f32(xg + c));
        float t = vaddvq_f32(s);
        for (; c < group_size; c++) t += xg[c];
        x_gsum[g] = t;
    }
    GemvCtx ctx{W, scales, zeros, x, x_gsum.data(), y, in_f, group_size};
    parallel_rows(out_f, (size_t)out_f * in_f, gemv_rows, &ctx);
}

void q4_dequant_rows(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ out,
    int r0, int r1, int in_f, int group_size
) {
    DequantCtx ctx{W, scales, zeros, out, r0, in_f, group_size};
    // parallel_rows hands out ranges relative to 0; shift them by r0.
    struct Shift { DequantCtx* c; int r0; } shift{&ctx, r0};
    parallel_rows(r1 - r0, (size_t)(r1 - r0) * in_f,
                  [](void* p, int a, int b) {
                      Shift* s = static_cast<Shift*>(p);
                      dequant_rows(s->c, s->r0 + a, s->r0 + b);
                  },
                  &shift);
}

#else  // no NEON: scalar implementations with the same semantics

void q4_gemv_sym_neon(
    const uint8_t* W, const float* scales, const float* x, float* y,
    int out_f, int in_f, int group_size
) {
    q4_gemv_sym_scalar(W, scales, x, y, out_f, in_f, group_size);
}

void q4_dequant_rows(
    const uint8_t* W, const float* scales, const float* zeros, float* out,
    int r0, int r1, int in_f, int group_size
) {
    int n_groups = in_f / group_size;
    for (int i = r0; i < r1; i++) {
        const uint8_t* w_row = W + (size_t)i * (in_f / 2);
        for (int c = 0; c < in_f; c++) {
            uint8_t byte = w_row[c / 2];
            uint8_t u = (c & 1) ? (byte >> 4) : (byte & 0x0F);
            int g = c / group_size;
            float sc = scales[(size_t)i * n_groups + g];
            float v = zeros ? (float)u - zeros[(size_t)i * n_groups + g]
                            : (float)((u < 8) ? (int)u : (int)u - 16);
            out[(size_t)(i - r0) * in_f + c] = v * sc;
        }
    }
}

void q4_gemv_asym_neon(
    const uint8_t* W, const float* scales, const float* zeros,
    const float* x, float* y, int out_f, int in_f, int group_size
) {
    std::vector<float> row(in_f);
    for (int i = 0; i < out_f; i++) {
        q4_dequant_rows(W, scales, zeros, row.data(), i, i + 1, in_f, group_size);
        float acc = 0.0f;
        for (int c = 0; c < in_f; c++) acc += row[c] * x[c];
        y[i] = acc;
    }
}

#endif  // __ARM_NEON

// ---------------------------------------------------------------------------
// Scalar reference (always compiled; used for correctness tests)
// ---------------------------------------------------------------------------
void q4_gemv_sym_scalar(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ scales,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_f, int in_f, int group_size
) {
    int n_groups = in_f / group_size;
    int half_gs  = group_size / 2;

    for (int i = 0; i < out_f; i++) {
        const uint8_t* w_row = W + (size_t)i * (in_f / 2);
        const float*   s_row = scales + i * n_groups;
        float y_val = 0.0f;

        for (int g = 0; g < n_groups; g++) {
            const uint8_t* wg = w_row + g * half_gs;
            const float*   xg = x + g * group_size;
            float g_sum = 0.0f;

            for (int c = 0; c < half_gs; c++) {
                uint8_t byte = wg[c];
                uint8_t lo_u = byte & 0x0F;
                uint8_t hi_u = byte >> 4;
                // Sign-extend 4→8 bits: values 8..15 map to -8..-1
                int8_t lo_s = (lo_u < 8) ? (int8_t)lo_u : (int8_t)((int)lo_u - 16);
                int8_t hi_s = (hi_u < 8) ? (int8_t)hi_u : (int8_t)((int)hi_u - 16);
                g_sum += (float)lo_s * xg[2 * c];
                g_sum += (float)hi_s * xg[2 * c + 1];
            }

            y_val += g_sum * s_row[g];
        }
        y[i] = y_val;
    }
}
