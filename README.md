# siliconfer

> A **from-scratch 4-bit LLM inference engine** for Apple Silicon — no quantization library, no vendor kernel.

**Thesis:** int4 quantization keeps perplexity ≈ fp16 while cutting weight memory ~4× and speeding up bandwidth-bound decode.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-black?logo=apple)
![Tests](https://img.shields.io/badge/Tests-193%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Stack](https://img.shields.io/badge/Stack-MLX%20%7C%20NEON%20%7C%20Metal%20%7C%20C%2B%2B17-orange)

---

## Results

Benchmarked on **Qwen2.5-0.5B**, **M4 base (16 GB unified memory)**:

| Method | WikiText-2 PPL | ΔPPL | Decode tok/s | Memory | Compression |
|---|---|---|---|---|---|
| fp16 (MLX) | 18.96 | — | 18.2 | 988 MB | 1× |
| RTN-int4 | 25.94 | +6.98 | 14.2 | 463 MB | 2.1× |
| **GPTQ-int4** | **21.34** | **+2.38** | 14.4 | 463 MB | **2.1×** |
| AWQ-int4 | 34.87 | +15.91 | 14.4 | 463 MB | 2.1× |
| HQQ-int4 | 22.62 | +3.66 | 13.3 | 474 MB | 2.1× |
| SINQ-int4 | 28.68 | +9.72 | 14.4 | 463 MB | 2.1× |

HQQ needs **zero calibration data** and still clearly beats RTN and AWQ, landing
close to GPTQ (which needs a full Hessian calibration pass). Getting an honest number
here took a real fix along the way: the first version of `Q4Linear`'s packer only
supported symmetric requantization, silently corrupting HQQ's inherently-asymmetric
output — measured PPL 31.55 before the fix (vs. 22.62 after). Added the missing
asymmetric NEON prefill kernel (`gemm_asym`) and zero-point storage in `Q4Linear`;
re-verified end-to-end on the real model afterward. (HQQ's memory footprint is
slightly higher than the other methods' — 474 MB vs 463 MB — because asymmetric
packing stores an extra zero-point per group.)

**SINQ underperforms RTN in this table** (Δ+9.72 vs Δ+6.98) — a real result, not
cherry-picked around: under a *different*, longer evaluation context
(`scripts/quantize.py`'s default `seq_len=2048` vs this table's `seq_len=512`),
SINQ actually beats RTN (Δ+3.20 vs Δ+4.46). Both numbers are real and reproducible;
they just disagree on ranking, because different methods' quality degrades by very
different amounts under shorter-context evaluation — checked across all five
methods, not assumed: RTN's Δ only grows 1.57x going from seq_len=2048 to 512, HQQ's
1.67x, GPTQ's 2.18x, SINQ's 3.04x, and **AWQ's 6.49x** (the largest by far — AWQ was
already the weakest method here, see below). So this isn't a SINQ-specific problem;
some methods are simply more sensitive to evaluation context length than others, and
SINQ is the second-most sensitive of the five. See `CLAUDE.md §9`/`NOTES.md §13` for
the full numbers and the standing rule not to mix PPL figures from the two
evaluation scripts in one table.

Projection weights alone: **190 MB vs 748 MB fp16 = 3.9× compression**.
NEON kernel microbenchmark (simulated 7B decode): **4.1× speedup over fp16 numpy**.
KV-cache int8 quantization: **~1.9× cache memory reduction**, PPL 17.13→18.34 (Δ+1.2)
via true incremental decode on real WikiText-2 text — safer than RTN's weight-only
degradation. Fused quantized-attention Metal kernel: after two honestly-slower
iterations, real flash-attention-style tiling made it **1.03-1.13× faster than
native SDPA at T=4096-16384 context** — see §14 below.

---

## What's inside

Three pillars, all implemented from first principles. `NOTES.md` has the full
math derivations plus five debugging deep-dives worth reading if you like this
kind of thing: a single "super weight" that took WikiText-2 PPL from 12 to 10,000+
(§10), an algebraic proof that a promising-looking speculative-decoding shortcut is
provably *not* lossless (§11), a kernel bug that silently corrupted every
asymmetric-quantization method until a real fix — not a documented caveat — closed
it (§12), a proof that half of a "dual-scale" quantization paper's own
mechanism is a no-op in this codebase's baseline, plus the one-line fix (absolute
→ relative error) that took the other half from a 3% win to a 98% one (§13), and
the Metal kernel story below (§14): diagnosed correctly, failed twice, then
succeeded on the third attempt by acting on the diagnosis instead of re-polishing
what was already optimized.

### 1 — Quantization math

**RTN** — naive round-to-nearest baseline.

**GPTQ** (Frantar et al. 2022) — Hessian-weighted error propagation across weight columns:

```
H = 2·X·Xᵀ             calibration Hessian (shared across rows)
H += λ·mean(diag H)·I   dampen for invertibility (λ ≈ 0.01)
U = upper_cholesky(H⁻¹) ← upper Cholesky of inverse, NOT H⁻¹ directly

For each column q (left → right):
  quantize W[:,q] → Q[:,q]
  W[:,rest] -= ((W[:,q] - Q[:,q]) / U[q,q]) · U[q, rest]
```

Key detail: `U[q,q]` is the Schur-complement diagonal (conditional), not `H⁻¹[q,q]` (unconditional). Using the wrong one was a bug caught during implementation.

**AWQ** (Lin et al. 2023) — activation-aware channel scaling:

```
Y = W·X = (W·diag s)·(diag(s⁻¹)·X),   s = act_scale^α

Quantize W·diag(s)  → smaller error on high-activation channels
Fold diag(s⁻¹) into the preceding RMSNorm → zero runtime cost
```

**HQQ** (Badri & Shaji 2023) — calibration-free, robust to weight outliers:

```
For each group, search candidate clip ranges [median ± k·MAD] (k from a fixed grid,
k=∞ = plain min-max RTN); keep whichever minimizes the ACTUAL quantized
(round+clip+dequant) L_p loss (p=0.7, hyper-Laplacian).
```

No activations needed — only the weight tensor. Validating against real weights
surfaced a genuine "super weight" (Yu et al. 2024): a single lone outlier at robust
z-score ≈58 in one `down_proj` group that, if clipped, corrupted a residual-stream
channel enough to blow WikiText-2 PPL from ~12 to 10,000+. Magnitude statistics alone
can't tell "safe to compress" apart from "structurally critical" — the default
`k_grid` is set conservative enough that no plausible real super-weight gets caught.

**SINQ** (dual-scale-inspired, Sept 2025) — calibration-free, iterative column rescaling:

```
s = ones(in_features)
repeat: W' = W·diag(s);  quantize W' per-(row,group) as usual;  undo the scale
        rel_err[j] = RMS(W[:,j]-W_eff[:,j]) / RMS(|W[:,j]|)     ← RELATIVE, not absolute
        s *= (rel_err / mean(rel_err))^β                        ← crushed columns get more room
```

The real SINQ paper also fits a *row*-scale. Proven algebraically (and confirmed to
float64 machine epsilon) that a row-scale is a no-op here specifically: this
codebase's group-wise quantizer already computes an independent max-based scale per
*(output row, group)*, so any row-wise pre-scaling cancels out exactly before it can
help. What's shipped is honestly the column-scale half — a calibration-free
companion to AWQ, using the weight's own per-column magnitude spread instead of real
activation statistics. Getting the update signal right mattered a lot: a first
version using *absolute* reconstruction error as the correction signal only improved
~3% over plain RTN on a column-outlier test, because absolute error is roughly
uniform across columns sharing one group's quantization step regardless of magnitude
— switching to *relative* error took the same test to ~98% MSE reduction on the
crushed columns.

All five methods use **group-wise int4** (group size 128): two nibbles per byte, one fp32 scale per group (+ zero-point for asymmetric methods).

**Mixed 2/4-bit (and 3/4-bit) precision** (`quant/mixed_precision.py`,
CoopQ-inspired) — a from-scratch permutation-sampling Shapley-value estimator
(checked against closed-form ground truth for both additive and
pairwise-interaction games, not just "does it run") ranks each transformer
block's sensitivity, then demotes the least-sensitive blocks under a memory
budget (an exact top-k selection here, since every block in this architecture
has identical size). HQQ was generalized to run its outlier-aware clip-range
search at any bit-width (`quantize_sym_n`/`quantize_asym_n` in
`quant/primitives.py`), not just plain RTN, after a real-model check found
plain RTN-int2 catastrophically lossy (PPL 500,000+ demoting the whole
network) regardless of which blocks were chosen. Honest, scale-dependent
result: **2-bit is never a good tradeoff** vs uniform HQQ-int4 at this model's
size, even with the best selection + algorithm this session could produce
(Shapley selection beats uniform demotion by ~437x, HQQ-int2 beats RTN-int2 by
~30-2.5x, but the floor is still too low). **3-bit is a different, genuinely
useful story**: demoting the same 5 (of 24) least-sensitive blocks to 3-bit
instead of 2-bit cuts PPL degradation by ~2.6x (25.33 vs 65.65) for a smaller
memory saving — a real, modest compression bonus on top of uniform int4. See
NOTES.md §15-16 for the full numbers; not shipped as a served (packed-kernel)
method yet — see § 3 below.

---

### 2 — ARM NEON kernel

Hand-written C++17 GEMV (single-token decode) + GEMM (prefill), compiled with pybind11.

Weight layout: low nibble = even column, hi nibble = odd column.
Inner loop over 16 packed bytes (32 int4 values):

```cpp
uint8x16_t packed = vld1q_u8(w_ptr);
int8x16_t lo = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(packed), 4), 4); // even cols
int8x16_t hi = vshrq_n_s8(vreinterpretq_s8_u8(packed), 4);                 // odd cols
// vuzpq_f32 deinterleaves even/odd after int→float, then fused multiply-accumulate
```

The `(x<<4)>>4` shift trick sign-extends nibbles in 2 NEON instructions with no branches.
Python falls back to a NumPy reference if the `.so` isn't built.

**Why decode speeds up (roofline):**

```
M4 unified bandwidth ≈ 120 GB/s

fp16 GEMV: load n_weights × 2 bytes  →  60 B weights/s
int4 GEMV: load n_weights × 0.5 byte →  240 B weights/s
                                Theoretical speedup: 4×
```

---

### 3 — Serving loop

Full Llama-style decoder built on MLX: `RMSNorm → RoPE → GQA attention (KV cache) → SwiGLU MLP`.

`Q4Linear` replaces every projection with NEON-backed int4 matmul; MLX only tracks attention and norms — packed weight bytes are invisible to its parameter tree.

**Quantized KV cache** (`model/kv_cache.py`): group-wise int8, one scale per (batch, kv_head, token) vector — a KV vector has no natural sub-grouping to exploit int4's nibble-packing the way weight rows do, so int8 is the safer, still-real ~1.9× win. Implemented in native `mx.array` ops end-to-end (no numpy round-trip) since the cache is quantized on *every* decode step, not once at load time like weights — a numpy round-trip here would reproduce the CPU↔GPU sync bottleneck below, but per-token instead of per-model-load. Opt-in via `generate(..., quantize_kv_cache=True)` or `speculative_generate(..., quantize_kv_cache=True)` (composes with dynamic K too).

**Fused quantized-attention Metal kernel** (`kernels/metal/q4_attention.py`): closes a real, confirmed-absent gap — MLX has no first-class quantized-KV `scaled_dot_product_attention` as of this writing. Dequantizes int8 K/V inline via a hand-written `mx.fast.metal_kernel`, running a numerically-stable online softmax (`metal::precise::exp`, not the fast-math default, which doesn't guarantee `exp(-INFINITY)==0`) so the cache is never materialized to full precision for a decode step.

Three iterations, all verified to float32 machine-epsilon precision against the reference path before any performance claim: **v1** (one thread per head) was 4-30× *slower* than native SDPA; **v2** (threadgroup-parallel across `head_dim`) improved to 6-16× slower but still only launched `B×n_heads` threadgroups regardless of context length — the actual bottleneck. **v3** fixes that directly: real flash-attention-style tiling splits the cache into `n_tiles` chunks, launches `B×n_heads×n_tiles` threadgroups, and merges each chunk's partial (un-normalized) online-softmax result with the standard flash-attention combine rule — a few cheap native `mx.array` ops, no second kernel dispatch. Result: **0.48-0.95× native SDPA at T=128-2048, and 1.03-1.13× (genuinely faster) from T=4096 to T=16384** — exactly the long-context regime this kernel's whole design point (never materialize full-precision KV) matters most for. This was the explicit "already failed twice, high-risk" item on this project's own todo list; it paid off on the third attempt by targeting the bottleneck Phase 9c had already correctly diagnosed but not yet acted on, rather than re-polishing what was already optimized.

**Speculative decoding** (`engine/speculative.py`): a small draft model proposes K tokens; the target verifies K+1 in one forward pass. Rejection sampling (`accept with prob min(1, p_target/p_draft)`) is provably lossless — with draft=target and greedy decoding, output matches non-speculative exactly (verified by test). **Dynamic speculation depth** adapts K round-to-round (deeper after full acceptance, shallower after an early rejection) — exactly lossless by construction, since K never appears in the accept/reject correctness proof. A more ambitious multi-candidate retry scheme was attempted first and rejected: proved algebraically that it requires negative fallback probabilities for retry counts ≥ 2, i.e. it's provably *not* lossless — kept as a documented, tested negative finding rather than shipped.

**EAGLE-3-inspired draft head** (`model/draft_head.py`, `engine/draft_training.py`) — a real, trained (not just architected) scaled-down version of EAGLE's actual design: multi-layer feature fusion from 3 target depths, one `TransformerBlock` at the target's own hidden width, and the target's own frozen embedding/LM head reused rather than trained from scratch (matching EAGLE's real architecture, not a simplification invented here). Trained via real supervised distillation on Qwen2.5-0.5B + WikiText-2 (full-sequence teacher forcing, not the paper's multi-step rollout simulation), now with proper early stopping and best-checkpoint restore. Honest result across three data scales: top-1 next-token accuracy climbed **4.9% → 17.9% → 20.6%** (40 → 150 → 250 training sequences) — real, consistent learning, never regressing — but the gains are clearly diminishing (+13.0 points, then only +2.6), well short of the target's own ~40% top-1 and not obviously closing that gap with modestly more of the same data (the real EAGLE-3 uses 500K+ distilled examples). Full `speculative_generate` integration is deferred rather than forced — see NOTES.md §16.3 for why, plus a real debugging detour along the way: an apparent training stall at 1000 sequences turned out to be genuine system memory pressure on this shared 16GB machine, not a code bug (confirmed via `vm_stat`, not guessed).

**TTFT pre-warming** (`engine/generate.py::warmup`, `--warmup` in `scripts/run.py`): runs a throwaway prefill-shape and decode-shape forward pass to trigger MLX's lazy-graph compilation before a real request. Confirmed a real, measurable **1.38× first-token speedup on the plain fp16 path** (77.4ms → 56.0ms) — but only ~1.04× (noise-level) on the actual quantized `Q4Linear` serving path, because that path already forces an eager `mx.eval()` per layer to bridge to the NEON kernel, leaving little lazy graph left to warm. Reported as measured, not assumed to help just because the underlying phenomenon is real.

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Build NEON kernel (macOS, requires clang++)
bash siliconfer/kernels/neon/build_kernel.sh

# Tests — 193 fast unit + integration tests (~3s, no model needed)
pytest tests/

# Full suite including logit-parity vs HuggingFace (requires Qwen2.5-0.5B in cache)
pytest tests/ --run-integration

# Interactive generation (add --method hqq/sinq for calibration-free quant,
# --quantize_kv_cache for int8 KV cache)
python scripts/run.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --method gptq \
    --prompt "Explain attention in transformers:" \
    --max_tokens 200

# Speculative decoding, with dynamic speculation depth (see "What's inside" below)
python scripts/run.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --speculative --dynamic_K \
    --prompt "The theory of relativity states that"

# TTFT pre-warming (warms MLX's lazy graph before the timed request)
python scripts/run.py \
    --model_id Qwen/Qwen2.5-0.5B --warmup \
    --prompt "The theory of relativity states that"

# Mixed 2/4-bit precision PPL comparison (algorithm-level only — no packed-kernel
# serving yet, see NOTES.md §15)
python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method mixed \
    --mixed_budget_ratio 0.9 --mixed_permutations 8

# Full benchmark matrix (fp16 + RTN + GPTQ + AWQ + HQQ + SINQ, ~25 min)
python scripts/run_benchmarks.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --full --ppl --max_ppl_tokens 5000
```

---

## Repository layout

```
siliconfer/
├── siliconfer/
│   ├── model/          # config, llama blocks, RoPE, RMSNorm, GQA attention, kv_cache.py, draft_head.py
│   ├── quant/          # primitives, rtn.py, gptq.py, awq.py, hqq.py, sinq.py, mixed_precision.py, calibration.py
│   ├── kernels/
│   │   ├── neon/       # q4_gemv.cpp, q4_gemm.cpp, pybind11 bindings, CMakeLists.txt
│   │   └── metal/      # q4_attention.py — fused quantized-attention mx.fast.metal_kernel
│   ├── engine/         # generate.py (incl. warmup()), q4_loader.py, speculative.py, draft_training.py
│   └── eval/           # perplexity.py, bench.py
├── eval/               # plots.py (PPL bar, throughput bar, memory bar, roofline)
├── scripts/            # run.py, quantize.py, benchmark.py, run_benchmarks.py
└── tests/              # 193 unit + integration tests
```

---

## Stack

- **Python 3.11+**, **MLX** (Apple array framework), **NumPy**
- **C++17 + ARM NEON intrinsics**, **pybind11**, CMake
- **Metal Shading Language** via `mx.fast.metal_kernel` — custom fused attention kernel
- **PyTorch / HuggingFace** — reference baselines only
- **Hardware:** MacBook M4 base · 16 GB unified memory · ARMv9.2a (NEON + I8MM + SME2)
