"""GEMV micro-benchmark: NEON int4 kernel vs optimized fp32/fp16/int4 baselines.

Compares, for one decode-shaped product y = W @ x (batch 1):

  neon_q4_1t        this repo's NEON int4 kernel (CPU, one thread)
  neon_q4_mt        the same kernel on all performance cores (products under
                    2**19 weights stay on one thread by design)
  accelerate_fp32   NumPy float32 matmul -> Apple Accelerate cblas_sgemv (CPU)
  numpy_fp16        float16 weights converted to float32 on every call, then
                    matmul (CPU). Unoptimized; kept only because earlier
                    numbers in this repo used it as the baseline.
  mlx_fp16          MLX float16 matmul (GPU)
  mlx_fp32          MLX float32 matmul (GPU)
  mlx_q4            mx.quantized_matmul, MLX's native 4-bit path (GPU)

Two cache regimes per shape:

  hot    one matrix, called repeatedly. Small matrices stay in L2, so this
         measures compute throughput, not memory bandwidth.
  cold   K distinct matrices visited round-robin, K chosen so the int4 copies
         alone exceed --cold_bytes. Each visit misses cache, which is what a
         real decode step sees (it walks every layer's weights once per token).

Memory bandwidth is measured on this machine (large-array copy, 1 and N
processes). Each contender's GB/s is the bytes of *its own* weight
representation read per second; `weights_per_s` is the format-independent
throughput.

Writes results/bench/gemv.json. Run with nothing else heavy on the machine:

    python scripts/bench_gemv.py
    python scripts/bench_gemv.py --shapes 896x896,4864x896 --n_reps 200
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.eval.env_info import collect_env_info
from siliconfer.kernels.neon import (
    _BACKEND,
    gemv_sym,
    get_num_threads,
    pack_weights_sym,
    set_num_threads,
)

# out_features x in_features. First four: every distinct projection shape in
# Qwen2.5-0.5B. Last two: a 7B-class attention and MLP projection.
DEFAULT_SHAPES = "896x896,128x896,4864x896,896x4864,4096x4096,14336x4096"


def _stats(samples_s: list[float]) -> dict:
    a = np.asarray(samples_s) * 1e3
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def _time_calls(fns: list, n_reps: int, n_warmup: int) -> list[float]:
    """Time n_reps calls, cycling round-robin through `fns`. Per-call seconds."""
    k = len(fns)
    for i in range(n_warmup):
        fns[i % k]()
    out = []
    for i in range(n_reps):
        fn = fns[i % k]
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _copy_worker(barrier, mb: int, n_reps: int, queue) -> None:
    n = mb * 1024 * 1024 // 4
    src = np.ones(n, dtype=np.float32)
    dst = np.empty(n, dtype=np.float32)
    np.copyto(dst, src)  # touch every page before timing
    barrier.wait()
    t0 = time.monotonic()
    for _ in range(n_reps):
        np.copyto(dst, src)
    queue.put((t0, time.monotonic(), 2 * n * 4 * n_reps))


def measure_copy_bandwidth(n_procs: int, mb_per_proc: int = 256, n_reps: int = 10) -> float:
    """Aggregate GB/s of `n_procs` concurrent large-array copies (read + write bytes).

    Uses processes, not threads: np.copyto holds the GIL, so a thread pool
    reports the single-thread figure regardless of thread count.
    """
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(n_procs), ctx.Queue()
    procs = [ctx.Process(target=_copy_worker, args=(barrier, mb_per_proc, n_reps, queue))
             for _ in range(n_procs)]
    for p in procs:
        p.start()
    spans = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    # Bytes moved while all workers overlap, over the span they were all running.
    start, end = max(s[0] for s in spans), min(s[1] for s in spans)
    return sum(s[2] * (end - start) / (s[1] - s[0]) for s in spans) / (end - start) / 1e9


def bench_shape(out_f: int, in_f: int, group_size: int, n_reps: int, n_warmup: int,
                cold_bytes: int, max_copies: int) -> dict:
    rng = np.random.default_rng(0)
    n_threads = get_num_threads()
    q4_bytes = out_f * in_f // 2 + out_f * (in_f // group_size) * 4
    k_cold = int(min(max_copies, max(2, -(-cold_bytes // q4_bytes))))

    Ws = [rng.normal(0, 1, (out_f, in_f)).astype(np.float32) for _ in range(k_cold)]
    x = rng.normal(0, 1, in_f).astype(np.float32)
    x_mx = mx.array(x)

    packed = [pack_weights_sym(W, group_size=group_size) for W in Ws]
    W16 = [W.astype(np.float16) for W in Ws]
    W_mx16 = [mx.array(W).astype(mx.float16) for W in Ws]
    W_mx32 = [mx.array(W) for W in Ws]
    W_mxq = [mx.quantize(mx.array(W), group_size=group_size, bits=4) for W in Ws]
    x_mx16 = x_mx.astype(mx.float16)
    mx.eval(W_mx16, W_mx32, W_mxq, x_mx, x_mx16)

    # name -> (bytes of weight representation read per call, one callable per copy)
    neon = [lambda p=p: gemv_sym(p[0], p[1], x, group_size) for p in packed]
    contenders = {
        "neon_q4_1t": (q4_bytes, neon),
        "neon_q4_mt": (q4_bytes, neon),
        "accelerate_fp32": (out_f * in_f * 4, [lambda W=W: W @ x for W in Ws]),
        "numpy_fp16": (out_f * in_f * 2, [lambda W=W: W.astype(np.float32) @ x for W in W16]),
        "mlx_fp16": (out_f * in_f * 2, [lambda W=W: mx.eval(W @ x_mx16) for W in W_mx16]),
        "mlx_fp32": (out_f * in_f * 4, [lambda W=W: mx.eval(W @ x_mx) for W in W_mx32]),
        "mlx_q4": (q4_bytes, [
            lambda q=q: mx.eval(mx.quantized_matmul(x_mx, q[0], q[1], q[2], transpose=True,
                                                    group_size=group_size, bits=4))
            for q in W_mxq
        ]),
    }

    # Correctness guard: a contender that computes something else is not a baseline.
    ref = Ws[0] @ x
    y_neon = gemv_sym(packed[0][0], packed[0][1], x, group_size)
    y_mxq = np.array(mx.quantized_matmul(x_mx, *W_mxq[0], transpose=True,
                                         group_size=group_size, bits=4))
    rel = lambda y: float(np.linalg.norm(y - ref) / np.linalg.norm(ref))

    result = {
        "out_f": out_f, "in_f": in_f, "group_size": group_size, "cold_copies": k_cold,
        "rel_err_vs_fp32": {"neon_q4": rel(y_neon), "mlx_q4": rel(y_mxq)},
        "neon_mt_threads": n_threads,
        "contenders": {},
    }
    for name, (nbytes, fns) in contenders.items():
        entry = {"weight_bytes": nbytes}
        set_num_threads(1 if name == "neon_q4_1t" else n_threads)
        for regime, regime_fns in (("hot", fns[:1]), ("cold", fns)):
            s = _stats(_time_calls(regime_fns, n_reps, n_warmup))
            s["gb_per_s"] = nbytes / (s["p50_ms"] / 1e3) / 1e9
            s["weights_per_s"] = out_f * in_f / (s["p50_ms"] / 1e3)
            entry[regime] = s
        result["contenders"][name] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help="Comma-separated OUTxIN list.")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--cold_bytes", type=int, default=64 * 1024 * 1024,
                        help="Minimum total int4 bytes across the cold-regime copies.")
    parser.add_argument("--max_copies", type=int, default=64)
    parser.add_argument("--out", default="results/bench/gemv.json")
    args = parser.parse_args()

    n_perf = int(os.popen("sysctl -n hw.perflevel0.physicalcpu").read().strip() or 1)
    bandwidth = {
        "method": "np.copyto on 256 MB float32 arrays, one process per core, "
                  "read+write bytes counted",
        "copy_1_proc_gb_s": measure_copy_bandwidth(1),
        f"copy_{n_perf}_procs_gb_s": measure_copy_bandwidth(n_perf),
    }
    print(f"[bench_gemv] kernel backend: {_BACKEND}")
    print(f"[bench_gemv] measured copy bandwidth: {bandwidth}")

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    results = []
    for out_f, in_f in shapes:
        r = bench_shape(out_f, in_f, args.group_size, args.n_reps, args.n_warmup,
                        args.cold_bytes, args.max_copies)
        results.append(r)
        print(f"\n{out_f}x{in_f}  (cold copies: {r['cold_copies']})")
        print(f"  {'contender':<16} {'hot p50 ms':>11} {'cold p50 ms':>12} {'cold p90 ms':>12} "
              f"{'cold GB/s':>10}")
        for name, e in r["contenders"].items():
            print(f"  {name:<16} {e['hot']['p50_ms']:>11.4f} {e['cold']['p50_ms']:>12.4f} "
                  f"{e['cold']['p90_ms']:>12.4f} {e['cold']['gb_per_s']:>10.2f}")

    out = {
        "config": vars(args),
        "kernel_backend": _BACKEND,
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "bandwidth": bandwidth,
        "shapes": results,
        "env": collect_env_info(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[bench_gemv] wrote {out_path}")


if __name__ == "__main__":
    main()
