"""Decode-step attention over an int8 KV cache: fused Metal kernel vs MLX.

One decode step = one query against T cached key/value vectors. Three ways to
compute it, at Qwen2.5-0.5B's attention shape (14 heads, 2 KV heads, head
dimension 64):

  fused_metal       kernels/metal/q4_attention.py — reads the int8 codes
                    directly; the cache is never expanded to floats.
  dequant_then_sdpa dequantize the whole int8 cache to fp16, then
                    mx.fast.scaled_dot_product_attention. This is what the
                    int8 cache costs without the fused kernel.
  sdpa_fp16_cache   the same MLX attention on an unquantized fp16 cache
                    (reference: no quantization at all).

Reports p50 / p90 per context length and the fused kernel's max abs error
against dequant_then_sdpa.

    python scripts/bench_metal_attention.py

Writes results/bench/metal_attention.json. Run on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.eval.env_info import collect_env_info
from siliconfer.kernels.metal.q4_attention import _choose_n_tiles, fused_quantized_attention_decode
from siliconfer.model.kv_cache import dequantize_kv, quantize_kv


def _time(fn, n_reps: int, n_warmup: int) -> dict:
    for _ in range(n_warmup):
        mx.eval(fn())
    samples = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(samples)
    return {"p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)),
            "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)), "n": n_reps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context_lens", default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--n_heads", type=int, default=14)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--out", default="results/bench/metal_attention.json")
    args = parser.parse_args()

    H, KV, D = args.n_heads, args.n_kv_heads, args.head_dim
    rep = H // KV
    scale = D ** -0.5
    rows = []
    for T in (int(v) for v in args.context_lens.split(",")):
        mx.random.seed(T)
        q = mx.random.normal((1, H, D))
        k = mx.random.normal((1, KV, T, D)).astype(mx.float16)
        v = mx.random.normal((1, KV, T, D)).astype(mx.float16)
        k_codes, k_scale = quantize_kv(k)
        v_codes, v_scale = quantize_kv(v)
        k_s, v_s = k_scale[..., 0], v_scale[..., 0]
        mx.eval(q, k, v, k_codes, v_codes, k_s, v_s, k_scale, v_scale)
        q4 = q[:, :, None, :]                       # [1, H, 1, D] for SDPA

        def sdpa(kk, vv):
            # GQA: each KV head serves `rep` query heads.
            return mx.fast.scaled_dot_product_attention(
                q4.astype(kk.dtype), mx.repeat(kk, rep, axis=1), mx.repeat(vv, rep, axis=1),
                scale=scale, mask=None)[:, :, 0, :]

        fused = lambda: fused_quantized_attention_decode(q, k_codes, k_s, v_codes, v_s)
        dequant = lambda: sdpa(dequantize_kv(k_codes, k_scale, mx.float32),
                               dequantize_kv(v_codes, v_scale, mx.float32))
        plain = lambda: sdpa(k, v)

        err = float(mx.max(mx.abs(fused() - dequant())).item())
        row = {
            "T": T,
            "n_tiles": _choose_n_tiles(T),
            "fused_max_abs_err_vs_dequant_sdpa": err,
            "fused_metal": _time(fused, args.n_reps, args.n_warmup),
            "dequant_then_sdpa": _time(dequant, args.n_reps, args.n_warmup),
            "sdpa_fp16_cache": _time(plain, args.n_reps, args.n_warmup),
        }
        row["fused_speedup_vs_dequant_sdpa"] = (
            row["dequant_then_sdpa"]["p50_ms"] / row["fused_metal"]["p50_ms"])
        row["fused_speedup_vs_sdpa_fp16"] = (
            row["sdpa_fp16_cache"]["p50_ms"] / row["fused_metal"]["p50_ms"])
        rows.append(row)
        print(f"T={T:<6} fused {row['fused_metal']['p50_ms']:.3f} ms  "
              f"dequant+sdpa {row['dequant_then_sdpa']['p50_ms']:.3f} ms  "
              f"sdpa(fp16) {row['sdpa_fp16_cache']['p50_ms']:.3f} ms  "
              f"x{row['fused_speedup_vs_dequant_sdpa']:.2f} / x{row['fused_speedup_vs_sdpa_fp16']:.2f}  "
              f"err {err:.1e}")

    out = {"shape": {"n_heads": H, "n_kv_heads": KV, "head_dim": D, "batch": 1},
           "n_reps": args.n_reps, "n_warmup": args.n_warmup,
           "rows": rows, "env": collect_env_info()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench_metal_attention] wrote {out_path}")


if __name__ == "__main__":
    main()
