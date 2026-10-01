# Kernels and the serving path

How packed int4 weights are multiplied, on the CPU and on the GPU, and what
that costs. All numbers quoted here come from `results/bench/` and are
tabulated in `results/SUMMARY.md`; the machine is an Apple M4 (4 performance
+ 6 efficiency cores, 16 GB).

## Two backends for the same codes

`Q4Linear` (`siliconfer/model/q4_linear.py`) replaces each quantized
projection. It takes the packed codes produced by any quantization method and
runs them on one of two backends:

| `backend` | Where | Kernel | Use |
|---|---|---|---|
| `"neon"` | CPU | this repo's C++/NEON kernel (`siliconfer/kernels/neon/`) | perplexity evaluation, CPU experiments |
| `"mlx"` | GPU | `mx.quantized_matmul`, MLX's built-in 4-bit product | generation |

The GPU backend does not re-quantize anything. MLX stores
`w = scale·u + bias` with unsigned 4-bit `u`, eight per `uint32`, lowest bits
first, which is this repo's byte layout viewed as `uint32`. So:

```
asymmetric:  (u − zero)·scale            →  bias = −zero·scale
symmetric:   s·scale, s two's complement →  u = s + 8 = nibble XOR 8,  bias = −8·scale
```

`tests/test_integration.py::test_to_mlx_quantized_is_lossless` checks the
conversion, and the two backends are checked against each other on single
layers and on a whole model.

## NEON GEMV (decode: one vector per call)

`q4_gemv.cpp`. Per output row, per group, 16 packed bytes (32 weights) at a
time:

1. sign-extend both nibbles with shifts: `hi = asr(b, 4)`,
   `lo = asr(lsl(b, 4), 4)`;
2. interleave `lo`/`hi` (`zip`) so the 32 int8 weights are in column order —
   cheaper than de-interleaving the 32 floats of `x` for every row;
3. widen int8 → int16 → int32 → float32 (eight vectors of four);
4. FMA each vector against `x` into its own accumulator. Eight independent
   accumulators keep the loop limited by SIMD throughput instead of by the
   latency of one FMA dependency chain.

The eight sums are reduced, multiplied by the group scale and added to the
row's output. The asymmetric kernel computes
`scale·(Σ u·x − zero·Σ x)`; `Σ x` per group is the same for every row and is
computed once per call.

Rows are independent, so products of at least 2¹⁹ weights are split by row
range across the performance cores with Grand Central Dispatch. The result
does not depend on the thread count
(`tests/test_kernel.py::test_gemv_result_independent_of_thread_count`).

The arithmetic is exact with respect to the dequantized weights: the
activation is not quantized. A kernel that also quantized the activation to
int8 and used `sdot` (as llama.cpp does) would do about four times more work
per SIMD instruction, at the price of a second approximation; that is not
implemented.

### What the micro-benchmark shows (`results/bench/gemv.json`)

- **One thread** processes about 14 billion weights per second regardless of
  matrix size, i.e. about 7 GB/s of packed bytes. Measured single-process copy
  bandwidth is about 85 GB/s, so one thread of this kernel is limited by
  computation (unpacking nibbles to floats), not by memory.
- **Four threads** reach 23–27 GB/s of packed bytes, about a quarter of the
  measured four-process copy bandwidth (102 GB/s).
- **Against optimized fp32 on the CPU** (Accelerate `sgemv`): once the matrix
  no longer fits in cache the threaded int4 kernel is 2.4× faster at the
  4864 × 896 MLP shape of Qwen2.5-0.5B and 3.0× faster at a 7B-class
  14336 × 4096 shape. Single-threaded it is slower than Accelerate (which is
  itself multi-threaded) at every shape. On a small matrix that stays in
  cache, Accelerate is faster by a wide margin.
- **Against the GPU, per call:** at the 0.5B shapes the threaded CPU kernel
  is the fastest contender, because one GPU product evaluated on its own
  pays a fixed dispatch cost of roughly 0.2 ms. At the 7B-class shape MLX's
  native 4-bit product is about 1.8× faster than this kernel.
- **NumPy fp16** (`numpy_fp16` in the table: fp16 weights converted to fp32
  on every call) is about 4–6× slower than Accelerate fp32 at the larger
  shapes. It is included only to show how much a weak baseline flatters a
  kernel; Accelerate is the baseline to compare against.

## NEON + BLAS GEMM (prefill: many vectors per call)

`q4_gemm.cpp`. Prefill reuses each weight for every token, so reading the
weights is amortised and the multiply-adds dominate. The kernel dequantizes a
tile of rows to float32 with NEON (1 MB per tile, so it stays in L2) and
multiplies the activations against the tile with Accelerate `sgemm`. Only one
tile is ever expanded; the weights stay packed in memory. Below four tokens
it calls the GEMV kernel per token instead.

The first version looped the GEMV kernel over tokens. Replacing it made a
16-window perplexity run about 13× shorter end to end with a bit-identical
RTN perplexity, which is what made the full-test-set evaluation tier
affordable.

## Why `backend="neon"` is slow for generation

A decode step calls 168 quantized projections. With the NEON backend each of
them leaves the MLX graph: the activation has to be evaluated on the GPU,
copied to the CPU, multiplied, and copied back. That is 168 forced
evaluations and round trips per token, and it costs far more than the
products themselves (see the decode table in the README: the NEON backend is
several times slower than fp16 end to end even though its kernel wins the
per-call comparison above).

The GPU backend keeps the whole step in one lazy graph and is the path to
use for generation. A decode loop running entirely on the CPU (normalisation,
attention and the output head in NumPy, no MLX in the loop) would remove the
round trips for the NEON kernel as well; it has not been written.

## Fused int8-KV attention (Metal)

`siliconfer/kernels/metal/q4_attention.py`, an `mx.fast.metal_kernel`. For
one decode step it computes attention directly over an int8 KV cache
(`model/kv_cache.py`): keys and values are dequantized inline and a
single-pass online softmax accumulates the output, so the cache is never
expanded to floats.

The cache axis is split into tiles; each tile is a threadgroup that produces
a partial result `(m, l, acc)` — running maximum, sum of `exp(score − m)`,
and the weighted value sum — and the tiles are merged with the usual
flash-attention rule:

```
M = max_i m_i      L = Σ_i l_i·exp(m_i − M)      out = Σ_i acc_i·exp(m_i − M) / L
```

It uses `metal::precise::exp`: the shading language's fast-math `exp` does
not guarantee `exp(−∞) = 0`, which the masked terms rely on. Correctness is
tested against "dequantize, then MLX attention" to float32 precision,
including tile counts that do not divide the cache length, and repeated runs
to catch races. Timings are in `results/bench/metal_attention.json`.

## Limitations

- The embedding / output head (136 M of the model's 494 M parameters) is not
  quantized, so every decode step still reads it in fp16. mlx-lm's 4-bit
  conversion quantizes it, which is a large part of why its decode speed is
  higher (see the reference rows in the README).
- No comparison against llama.cpp has been run.
- One model size has been measured end to end.
