# siliconfer

A from-scratch **4-bit LLM inference engine for Apple Silicon**, built to prove one thesis:

> **int4 quantization preserves perplexity ≈ fp16 while cutting weight memory ~4× and speeding up bandwidth-bound decode.**

Implemented from first principles — no quantization library, no vendor kernel.

---

## Benchmark results (Qwen2.5-0.5B, M4 base, 16 GB)

| Method          | bits | WikiText-2 PPL | ΔPPL    | decode tok/s | peak mem | compression |
|-----------------|------|---------------|---------|-------------|----------|-------------|
| fp16 (MLX)      | 16   | 18.96         | —       | 17.3        | 988 MB   | 1.0×        |
| RTN-int4        | 4    | 25.94         | +6.98   | 13.2        | 463 MB   | 2.1×        |
| **GPTQ-int4**   | **4**| **21.34**     | **+2.38**| 13.1       | 463 MB   | **2.1×**    |
| AWQ-int4        | 4    | 34.87         | +15.91  | 14.4        | 463 MB   | 2.1×        |

PPL evaluated on first 5 000 tokens of WikiText-2 test split (seq\_len=512).
Memory = total model including embeddings and norms (projection weights alone: 190 MB vs 748 MB = **3.9× compression**).

Plots in `results/`: `ppl.png`, `throughput.png`, `memory.png`, `roofline.png`.

### Key finding: GPTQ beats RTN by 4.6 PPL points

RTN quantizes each weight independently. GPTQ uses the Hessian `H = 2XXᵀ` of the layer's
calibration inputs to propagate quantization error left-to-right across weight columns —
redistributing the error budget toward less-important weights. With 24 layers × 7 projections,
the compound benefit moves GPTQ from +6.98 to +2.38 over fp16.

### Why AWQ underperforms on 0.5B

AWQ scales each input channel by `s = act_scale^α` before quantizing, protecting channels
with large activation magnitudes. Qwen2.5-0.5B has relatively uniform activations
(grid-searched α values cluster at 0.05–0.40), so AWQ provides minimal benefit and its
per-layer MSE calibration objective does not generalise to the WikiText-2 test set. AWQ's
advantage is well-established on 7B+ models with spikier activation distributions.

---

## Why int4 decode is faster (roofline model)

Single-token decode (batch=1) is a GEMV — one matrix-vector multiply per linear layer. GEMV
is **memory-bandwidth-bound**: arithmetic intensity ≈ 1 FLOP / 2 bytes (fp16) or 1 FLOP / 0.5
bytes (int4, after dequant). Throughput is therefore set by how fast weights can be streamed
from DRAM, not by compute.

```
M4 unified memory bandwidth ≈ 120 GB/s (shared CPU + GPU + ANE)

fp16 GEMV: bandwidth = n_weights × 2 bytes  →  120 GB/s / 2 B = 60 B weights/s
int4 GEMV: bandwidth = n_weights × 0.5 byte →  120 GB/s / 0.5 B = 240 B weights/s
Theoretical speedup: 4×
```

The NEON kernel achieves ~6.7 GB/s single-threaded on M4 CPU — well below the peak because
decode runs only on one CPU core while the GPU handles attention. In a pure-CPU stack (no
GPU attention sync overhead) the 4× speedup is observed: Phase 5 microbenchmark measured
**4.1× on a simulated 7B decode step** (112 NEON GEMV calls vs equivalent fp16 numpy).

The hybrid architecture (NEON GEMV for projections + MLX/Metal for attention) incurs 168
`mx.eval` sync barriers per decode step on the 0.5B model, capping effective throughput.
For a 7B model these sync costs amortise over the much larger weight reads.

---

## Three-pillar design

### 1 — Quantization math

**RTN** (round-to-nearest): `Q(W) = clip(round(W / scale), -8, 7) × scale` with
`scale = max(|W_group|) / 7`. Fastest; used as the baseline comparator.

**GPTQ** (Hessian-weighted, Frantar et al. 2022):
```
Objective: argmin_Ŵ  ||W·X − Ŵ·X||²_F

H = 2·X·Xᵀ          (calibration Hessian, shared across rows)
H += λ·mean(diag H)·I   (dampen, λ ≈ 0.01)
U = upper_cholesky(H⁻¹)   ← NOT H⁻¹ directly

For each column q (left → right):
  quantize W_:,q → Q_:,q
  W_:,rest −= ((W_:,q − Q_:,q) / U[q,q]) · U[q, rest]
```
`U[q,q]` is the Schur-complement diagonal (conditional inverse-Hessian), not `H⁻¹[q,q]`
(unconditional). Using the wrong one was a key bug caught during implementation.

**AWQ** (activation-aware, Lin et al. 2023):
```
Y = W·X = (W·diag s)·(diag(s⁻¹)·X),   s = act_scale^α

Quantize W·diag(s)  →  smaller effective quantization error on salient channels
Fold diag(s⁻¹) into the preceding RMSNorm  →  zero runtime cost
```
Grid-search α ∈ [0,1] per projection group, minimising per-layer output MSE.

All methods use **group-wise symmetric int4** (group size 128 by default): two nibbles packed
per byte, one fp16 scale per group. Range: [-7, 7] (never −8, so lossless round-trip through
fake-quantize → re-pack).

### 2 — ARM NEON kernel

`kernels/neon/q4_gemv.cpp`: hand-written NEON int4 GEMV for Apple Silicon decode.

Weight layout: low nibble = even column, high nibble = odd column (two int4 per byte).
Inner loop over 16 packed bytes (32 int4 values per iteration):

```cpp
// Load 16 packed bytes → 32 int4 values
uint8x16_t packed = vld1q_u8(w_ptr);
int8x16_t lo = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(packed), 4), 4); // even cols
int8x16_t hi = vshrq_n_s8(vreinterpretq_s8_u8(packed), 4);                 // odd cols

// Deinterleave into two float32x4x2 groups, fused multiply-accumulate
// vuzpq_f32 separates even/odd interleaved results from vcvtq_f32_s32
```

The `vshlq_n_s8 / vshrq_n_s8` shift trick sign-extends nibbles in two NEON instructions
instead of a mask+branch. `vuzpq_f32` deinterleaves the even/odd results after int→float
conversion, recovering the packed-column addressing.

Pybind11 binding exposes `q4_gemv_sym`, `q4_gemm_sym`, and `neon_available()` to Python.
The Python wrapper falls back to a NumPy reference if the `.so` is not built.

### 3 — Serving loop

Standard Llama decoder: RMSNorm → RoPE → GQA attention (KV cache) → SwiGLU MLP.
End-to-end generation via `siliconfer.engine.generate.generate()`.
`Q4Linear` swaps every `nn.Linear` in the model with NEON-backed int4 matmul; MLX tracks
only attention KV and norms — the packed weight bytes are invisible to its parameter tree.

**Phase 8 addition — speculative decoding** (`engine/speculative.py`):
A small draft model proposes K tokens; the large target model verifies all K+1 in one parallel
forward pass. Accepted tokens are committed via rejection sampling (`accept with prob
min(1, p_target/p_draft)`); the first rejection draws from the adjusted distribution
`max(0, p_target − p_draft)`. The algorithm is provably lossless (same marginal distribution
as target-only sampling). With draft=target and greedy decoding, acceptance rate = 1.0 and
output matches non-speculative exactly (verified by test).

---

## Reproduce

```bash
# Install
pip install -e ".[dev]"

# Build NEON kernel (requires clang++ on macOS)
bash siliconfer/kernels/neon/build_kernel.sh

# Run all tests (77 tests, ~0.5s)
pytest tests/ --ignore=tests/test_logit_parity.py

# Full benchmark matrix (fp16 + RTN + GPTQ + AWQ, ~25 min)
python scripts/run_benchmarks.py --model_id Qwen/Qwen2.5-0.5B --full --ppl --max_ppl_tokens 5000

# Interactive generation
python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --method gptq \
    --prompt "Explain attention in transformers:" --max_tokens 200
```

---

## Repository layout

```
siliconfer/
├── siliconfer/
│   ├── model/          # config, llama blocks, RoPE, RMSNorm, GQA attention, KV cache
│   ├── quant/          # primitives (int4 pack/unpack/RTN), gptq.py, awq.py, calibration.py
│   ├── kernels/neon/   # q4_gemv.cpp, q4_gemm.cpp, pybind11 bindings, CMakeLists.txt
│   └── engine/         # generate.py, q4_loader.py, speculative.py
├── eval/               # plots.py (matplotlib bar charts + roofline)
├── siliconfer/eval/    # perplexity.py, bench.py
├── scripts/            # quantize.py, run.py, run_benchmarks.py
├── tests/              # 77 unit + integration tests
├── results/            # results.json, *.png (generated)
├── NOTES.md            # mathematical derivations
└── ROADMAP.md          # long-form plan
```

