"""Decode-step attention computed directly on an int8 KV cache (Metal).

MLX has no attention primitive that reads a quantized cache, so the plain
path dequantizes the whole cache before every attention call. This kernel
dequantizes keys and values inline and accumulates the output with a
single-pass online softmax, so the cache is never expanded to floats. It
handles one query token (decode); prefill uses the standard path.

The cache axis is split into `n_tiles` tiles. Each tile is a threadgroup of
`head_dim` threads that produces a partial result (m, l, acc): the running
maximum score, the sum of exp(score − m), and the weighted value sum. The
tiles are merged with the usual flash-attention rule:

    M   = max_i m_i
    L   = Σ_i l_i · exp(m_i − M)
    out = Σ_i acc_i · exp(m_i − M) / L

Tiling is what gives the GPU enough threadgroups to work with: without it
the kernel launches only batch × heads of them, whatever the cache length.

Notes:
- `metal::precise::exp` is required. The shading language's default
  fast-math `exp` does not guarantee exp(−inf) == 0, which the first
  online-softmax step relies on.
- Timings against MLX attention: scripts/bench_metal_attention.py.
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
    """Default tile count for a cache of length T_cache.

    Enough tiles to keep the GPU busy at long context, without splitting short
    contexts into tiles too small to pay for a threadgroup.
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
    """Attention for one decode step over an int8-quantized KV cache.

    Args:
        q:        [B, n_heads, head_dim] float32 query (one token).
        k_codes:  [B, n_kv_heads, T_cache, head_dim] int8.
        k_scales: [B, n_kv_heads, T_cache] float32.
        v_codes:  [B, n_kv_heads, T_cache, head_dim] int8.
        v_scales: [B, n_kv_heads, T_cache] float32.
        n_tiles:  number of tiles along T_cache; defaults to `_choose_n_tiles`.

    Returns:
        [B, n_heads, head_dim] float32 attention output (before o_proj).
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
