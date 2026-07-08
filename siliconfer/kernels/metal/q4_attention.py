"""Fused quantized-KV attention for the decode phase (Phase 9c, tiled in 9f).

Closes a real gap in MLX: as of this writing MLX has no first-class fused
quantized-KV `scaled_dot_product_attention` (confirmed absent — several
"quantized SDPA" PRs on ml-explore/mlx exist but are closed/unmerged), so the
straightforward integration path (Phase 9b) dequantizes the whole
`QuantizedKVCache` back to fp16/fp32 before calling the standard SDPA op,
materializing a full-precision copy of the cache on every single decode step.

This kernel instead dequantizes int8 keys/values *inline*, running an online
(numerically-stable, single-pass) softmax so the whole attention output for a
head is produced without ever materializing a full-precision K/V array.
Scope: this targets the decode phase specifically (one new query token,
T_q=1) attending over the full existing `QuantizedKVCache`; prefill still
uses the standard path (T_q>1 attention is far less memory-bandwidth-bound,
so there's less to gain there — see CLAUDE.md's roofline argument for why
decode is the target).

Correctness notes:
- Uses `metal::precise::exp`, not the default fast-math `exp`. MSL's default
  `-ffast-math` mode does not guarantee `exp(-INFINITY) == 0`, and the first
  online-softmax update deliberately evaluates `exp(-INFINITY - score)` to
  seed the running correction factor at 0 — getting this wrong silently
  corrupts every subsequent step's normalization.
- No `simdgroup_matrix`/Metal TensorOps use — intentionally, since per Metal
  4 TensorOps' own tiling constraints, cooperative matrix instructions add
  heavy synchronization overhead for small, irregular per-head reductions
  like this one; plain scalar/parallel-reduction loops are the right shape
  here, not the wrong one.

Three iterations, in increasing order of parallelism (all verified to
float32 machine-epsilon precision against the reference dequant+native-SDPA
path before being trusted):

  v1 — one thread per (batch, head), fully sequential loop over T_cache.
       Correct; only launches B*n_heads threads total (14 for Qwen2.5-0.5B) —
       measured 4-30x *slower* than native SDPA (T=128-2048).
  v2 — one threadgroup per (batch, head), one thread per head_dim index
       within it (parallel dot-product reduction via threadgroup memory +
       barriers). Still only B*n_heads *threadgroups* — measured 6-16x
       slower (T=128-8192), an improvement but nowhere near parity, because
       threadgroup *count* — not per-threadgroup work — was still the
       bottleneck: 14 threadgroups can't keep an M4 GPU's execution units
       busy regardless of how efficient each one is internally.
  v3 (current) — real flash-attention-style tiling: split T_cache into
       `n_tiles` chunks, launch B*n_heads*n_tiles threadgroups (one per
       chunk), each running the same online-softmax loop as v2 but only over
       its own chunk, producing a *partial* (not-yet-normalized) result.
       Partial results are merged across the n_tiles axis with the standard
       flash-attention combine rule (§ below), done as a few cheap native
       `mx.array` ops (no second kernel dispatch needed — n_tiles is small).
       This multiplies threadgroup count by n_tiles, which is exactly the
       axis v1/v2 never touched, and it closed the gap: measured (Qwen2.5-0.5B
       dims: 14 heads, 2 kv heads, head_dim=64) **0.48-0.95x native SDPA at
       T=128-2048** — a large jump from v2's 0.06-0.16x at the same sizes —
       and **1.03-1.13x (i.e. actually faster) from T=4096 to T=16384**,
       exactly the long-context regime where avoiding full-precision KV
       materialization matters most. `n_tiles` defaults to a heuristic
       (`_choose_n_tiles`) but can be passed explicitly; very short contexts
       (T<=64) still don't have enough total work to amortize tiling
       overhead and stay below parity even with tuning.

Partial-result merge math (standard flash-attention online-softmax combine):
given N partial results (m_i, l_i, acc_i) — each tile's own running max,
sum-of-exp, and un-normalized weighted value sum — the combined result is
    M = max_i(m_i)
    L = sum_i( l_i · exp(m_i - M) )
    ACC[d] = sum_i( acc_i[d] · exp(m_i - M) )
    out[d] = ACC[d] / L
which is the same online-softmax update rule the single-tile kernels already
use internally, just applied once across tiles instead of once per timestep.
"""

from __future__ import annotations

import mlx.core as mx

_MAX_HEAD_DIM = 256   # compile-time upper bound for the kernel's local accumulator

_SOURCE_TILED = r"""
    // One threadgroup per (batch, head, tile); one thread per head_dim index
    // within it. Same per-timestep parallel-reduction structure as v2, but
    // now only responsible for a `tile_size`-timestep slice of the cache —
    // multiplying threadgroup count by n_tiles is the actual point of this
    // version (see module docstring for why v1/v2's threadgroup count was
    // the real bottleneck, not per-threadgroup efficiency).
    uint d = thread_position_in_threadgroup.x;
    uint group_id = threadgroup_position_in_grid.x;

    int B          = params[0];
    int n_heads    = params[1];
    int n_kv_heads = params[2];
    int head_dim   = params[3];
    int T_cache    = params[4];
    int gqa_groups = params[5];
    int n_tiles    = params[6];
    int tile_size  = params[7];

    int tile_idx = (int)group_id % n_tiles;
    int bh       = (int)group_id / n_tiles;
    int b = bh / n_heads;
    int h = bh % n_heads;
    int kv_h = h / gqa_groups;

    int t_start = tile_idx * tile_size;
    int t_end   = metal::min(t_start + tile_size, T_cache);

    device const float* q_ptr = q + (uint)(b * n_heads + h) * (uint)head_dim;
    uint kv_base = (uint)((b * n_kv_heads + kv_h) * T_cache) * (uint)head_dim;
    uint s_base  = (uint)(b * n_kv_heads + kv_h) * (uint)T_cache;

    float scale_factor = 1.0 / metal::sqrt((float)head_dim);

    threadgroup float partial[256];
    threadgroup float sh_m;
    threadgroup float sh_l;
    threadgroup float sh_correction;
    threadgroup float sh_p;

    bool active = d < (uint)head_dim;
    float q_d = active ? q_ptr[d] : 0.0;
    float acc_d = 0.0;

    if (d == 0) {
        sh_m = -INFINITY;
        sh_l = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = t_start; t < t_end; t++) {
        uint k_off = kv_base + (uint)(t * head_dim);
        partial[d] = active
            ? q_d * ((float)k_codes[k_off + d] * k_scales[s_base + (uint)t])
            : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d == 0) {
            float score = 0.0;
            for (int i = 0; i < head_dim; i++) {
                score += partial[i];
            }
            score *= scale_factor;
            float m_new = metal::max(sh_m, score);
            sh_correction = metal::precise::exp(sh_m - m_new);
            sh_p = metal::precise::exp(score - m_new);
            sh_m = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint v_off = kv_base + (uint)(t * head_dim);
            float v_val = (float)v_codes[v_off + d] * v_scales[s_base + (uint)t];
            acc_d = acc_d * sh_correction + sh_p * v_val;
        }
        if (d == 0) {
            sh_l = sh_l * sh_correction + sh_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write the UN-normalized partial result — merge happens across tiles
    // in Python, not here (this tile doesn't know the other tiles' scale).
    if (active) {
        uint acc_off = (uint)(b * n_heads + h) * (uint)(n_tiles * head_dim)
                      + (uint)tile_idx * (uint)head_dim + d;
        out_acc[acc_off] = acc_d;
    }
    if (d == 0) {
        uint ml_off = (uint)(b * n_heads + h) * (uint)n_tiles + (uint)tile_idx;
        out_m[ml_off] = sh_m;
        out_l[ml_off] = sh_l;
    }
"""

_kernel_tiled = mx.fast.metal_kernel(
    name="q4_attention_decode_tiled",
    input_names=["q", "k_codes", "k_scales", "v_codes", "v_scales", "params"],
    output_names=["out_acc", "out_m", "out_l"],
    source=_SOURCE_TILED,
)


def _choose_n_tiles(T_cache: int, min_tile_size: int = 16, max_tiles: int = 256) -> int:
    """Heuristic tile count: enough tiles to give the GPU real parallelism at
    long context, without over-splitting short contexts into tiles too small
    to be worth their own threadgroup's fixed overhead.

    Benchmarked (Qwen2.5-0.5B dims, M4): this default gives 0.48-0.95x native
    SDPA at short-to-medium context (T<=2048) — a large improvement over the
    untiled v2 kernel's 0.06-0.16x at the same sizes — and *exceeds* native
    SDPA from T=4096 onward (1.03x-1.13x, measured up to T=16384), which is
    exactly the long-context regime this kernel's "never materialize
    full-precision KV" design point matters most for.
    """
    if T_cache <= min_tile_size:
        return 1
    return min(max_tiles, -(-T_cache // min_tile_size))  # ceil division


def fused_quantized_attention_decode(
    q: mx.array,
    k_codes: mx.array,
    k_scales: mx.array,
    v_codes: mx.array,
    v_scales: mx.array,
    n_tiles: int | None = None,
) -> mx.array:
    """Fused decode-phase attention over an int8-quantized KV cache.

    Args:
        q:        [B, n_heads, head_dim] float32 — single decode-step query
                  (T_q=1 already squeezed out by the caller).
        k_codes:  [B, n_kv_heads, T_cache, head_dim] int8.
        k_scales: [B, n_kv_heads, T_cache] float32.
        v_codes:  [B, n_kv_heads, T_cache, head_dim] int8.
        v_scales: [B, n_kv_heads, T_cache] float32.
        n_tiles:  number of T_cache chunks to split across threadgroups
                  (v3's tiling parameter — see module docstring). Defaults to
                  a heuristic based on T_cache if not given.

    Returns:
        [B, n_heads, head_dim] float32 attention output (pre-o_proj).
    """
    B, n_heads, head_dim = q.shape
    n_kv_heads = k_codes.shape[1]
    T_cache = k_codes.shape[2]
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim={head_dim} exceeds kernel's compile-time max {_MAX_HEAD_DIM}")
    gqa_groups = n_heads // n_kv_heads

    if n_tiles is None:
        n_tiles = _choose_n_tiles(T_cache)
    n_tiles = max(1, min(n_tiles, T_cache))
    tile_size = -(-T_cache // n_tiles)  # ceil division

    params = mx.array(
        [B, n_heads, n_kv_heads, head_dim, T_cache, gqa_groups, n_tiles, tile_size],
        dtype=mx.int32,
    )

    q32 = q.astype(mx.float32)
    k_scales32 = k_scales.reshape(B, n_kv_heads, T_cache).astype(mx.float32)
    v_scales32 = v_scales.reshape(B, n_kv_heads, T_cache).astype(mx.float32)

    out_acc, out_m, out_l = _kernel_tiled(
        inputs=[q32, k_codes, k_scales32, v_codes, v_scales32, params],
        grid=(B * n_heads * n_tiles * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(B, n_heads, n_tiles, head_dim), (B, n_heads, n_tiles), (B, n_heads, n_tiles)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    # Flash-attention-style merge across the tile axis (cheap: n_tiles is
    # small, and this stays as native mx.array ops on the GPU — no host
    # round-trip, no second kernel dispatch).
    M = mx.max(out_m, axis=2, keepdims=True)              # [B, n_heads, 1]
    correction = mx.exp(out_m - M)                          # [B, n_heads, n_tiles]
    L = mx.sum(out_l * correction, axis=2)                   # [B, n_heads]
    ACC = mx.sum(out_acc * correction[..., None], axis=2)    # [B, n_heads, head_dim]
    L_safe = mx.maximum(L, 1e-20)
    return ACC / L_safe[..., None]
