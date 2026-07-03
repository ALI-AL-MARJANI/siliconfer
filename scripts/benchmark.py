"""Phase 5 benchmark: NEON q4 GEMV throughput vs fp16 numpy baseline.

Measures:
  1. tok/s for simulated decode (batch=1 GEMV per layer × n_layers)
  2. Achieved memory bandwidth (GB/s) for weight reads
  3. Roofline: achieved fraction of M4 CPU memory bandwidth

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --out_f 4096 --in_f 4096 --n_reps 200
    python scripts/benchmark.py --model_size 1b   # simulate LLaMA-3.2-1B layer sizes
    python scripts/benchmark.py --model_size 7b   # simulate 7B hero layer sizes
"""

from __future__ import annotations
import argparse
import time
import numpy as np

from siliconfer.kernels.neon import NEON_AVAILABLE, pack_weights_sym, gemv_sym


# ---------------------------------------------------------------------------
# Bandwidth constants (M4 base, unified memory)
# ---------------------------------------------------------------------------
M4_BW_GB_S = 120.0   # ~120 GB/s peak unified memory bandwidth (M4 base)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()


def time_gemv(fn, n_reps=100, n_warmup=10):
    """Time fn() over n_reps calls, discard n_warmup. Returns mean seconds/call."""
    for _ in range(n_warmup):
        fn()
    t0 = _now()
    for _ in range(n_reps):
        fn()
    return (_now() - t0) / n_reps


# ---------------------------------------------------------------------------
# Model size presets (simulated LLaMA-style layers)
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    # (hidden, intermediate, n_layers)  — LLaMA-3.2 / Qwen2.5 approximate sizes
    "0.5b": (896,  4864,  24),
    "1b":   (2048, 8192,  16),
    "3b":   (3072, 8192,  28),
    "7b":   (4096, 14336, 32),
}


def _layer_shapes(model_size: str | None, out_f: int, in_f: int):
    """Returns list of (out, in) for each linear op in a simulated decode step."""
    if model_size and model_size in MODEL_PRESETS:
        h, ff, n_layers = MODEL_PRESETS[model_size]
        shapes = []
        for _ in range(n_layers):
            # Attention: q, k, v, o (simplify: all h×h)
            shapes += [(h, h), (h, h), (h, h), (h, h)]
            # MLP: gate, up, down
            shapes += [(ff, h), (ff, h), (h, ff)]
        return shapes
    # Single layer
    return [(out_f, in_f)]


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def benchmark_single(out_f: int, in_f: int, group_size: int, n_reps: int):
    """Benchmark one weight matrix."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (out_f, in_f)).astype(np.float32)
    x = rng.normal(0, 1, in_f).astype(np.float32)
    W_fp16 = W.astype(np.float16)

    # --- q4 NEON ---
    packed, scales = pack_weights_sym(W, group_size=group_size)

    def _q4():
        return gemv_sym(packed, scales, x, group_size)

    def _fp16():
        return (W_fp16.astype(np.float32) @ x)

    sec_q4  = time_gemv(_q4,  n_reps=n_reps)
    sec_fp16 = time_gemv(_fp16, n_reps=n_reps)

    # Bytes read per GEMV call
    bytes_q4  = out_f * in_f * 0.5     # 4 bits per weight = 0.5 bytes
    bytes_q4 += out_f * (in_f // group_size) * 4  # scales (float32)
    bytes_q4 += in_f * 4               # x (float32 input)
    bytes_fp16 = out_f * in_f * 2      # fp16 weights
    bytes_fp16 += in_f * 4             # x

    bw_q4  = bytes_q4  / sec_q4  / 1e9
    bw_fp16 = bytes_fp16 / sec_fp16 / 1e9

    speedup   = sec_fp16 / sec_q4
    roof_frac = bw_q4 / M4_BW_GB_S

    return {
        "out_f": out_f, "in_f": in_f,
        "sec_q4": sec_q4, "sec_fp16": sec_fp16,
        "bw_q4": bw_q4, "bw_fp16": bw_fp16,
        "speedup": speedup, "roof_frac": roof_frac,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default=None, choices=list(MODEL_PRESETS) + [None],
                        help="Simulate a full LLaMA-style decode step across all layers.")
    parser.add_argument("--out_f",     type=int, default=4096)
    parser.add_argument("--in_f",      type=int, default=4096)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--n_reps",    type=int, default=100)
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f" siliconfer NEON q4 benchmark")
    print(f"  NEON_AVAILABLE={NEON_AVAILABLE}, group_size={args.group_size}")
    print(f"  M4 peak bandwidth assumed: {M4_BW_GB_S:.0f} GB/s")
    print(f"{'='*64}\n")

    shapes = _layer_shapes(args.model_size, args.out_f, args.in_f)

    # Aggregate for model-level simulation
    total_sec_q4  = 0.0
    total_sec_fp16 = 0.0
    total_bytes_q4  = 0.0
    total_bytes_fp16 = 0.0

    print(f"{'out':>6} {'in':>6} | {'q4 ms':>8} {'fp16 ms':>9} | {'speedup':>8} | "
          f"{'q4 GB/s':>8} {'roof%':>7}")
    print("-" * 64)

    for (out_f, in_f) in shapes:
        r = benchmark_single(out_f, in_f, args.group_size, args.n_reps)
        total_sec_q4  += r["sec_q4"]
        total_sec_fp16 += r["sec_fp16"]
        total_bytes_q4  += r["bw_q4"] * r["sec_q4"]
        total_bytes_fp16 += r["bw_fp16"] * r["sec_fp16"]
        print(f"{out_f:>6} {in_f:>6} | "
              f"{r['sec_q4']*1e3:>7.3f}  {r['sec_fp16']*1e3:>8.3f}  | "
              f"{r['speedup']:>7.2f}×  | "
              f"{r['bw_q4']:>7.1f}  {r['roof_frac']*100:>6.1f}%")

    if len(shapes) > 1:
        # Model-level summary
        agg_bw_q4    = total_bytes_q4  / total_sec_q4  if total_sec_q4  > 0 else 0
        agg_speedup  = total_sec_fp16 / total_sec_q4
        print(f"\n{'─'*64}")
        print(f"  decode step (all {len(shapes)} ops):")
        print(f"    q4  total:  {total_sec_q4*1e3:.2f} ms  → "
              f"simulated {1/total_sec_q4:.1f} tok/s")
        print(f"    fp16 total: {total_sec_fp16*1e3:.2f} ms  → "
              f"simulated {1/total_sec_fp16:.1f} tok/s")
        print(f"    speedup:    {agg_speedup:.2f}×")
        print(f"    avg q4 bandwidth: {agg_bw_q4:.1f} GB/s  "
              f"({agg_bw_q4/M4_BW_GB_S*100:.1f}% of {M4_BW_GB_S:.0f} GB/s ceiling)")

    print()


if __name__ == "__main__":
    main()
