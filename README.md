# siliconfer

> A **from-scratch 4-bit LLM inference engine** for Apple Silicon — no quantization library, no vendor kernel.

**Thesis:** int4 quantization keeps perplexity ≈ fp16 while cutting weight memory ~4× and speeding up bandwidth-bound decode.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-black?logo=apple)
![Tests](https://img.shields.io/badge/Tests-81%20passing-brightgreen)
![Stack](https://img.shields.io/badge/Stack-MLX%20%7C%20NEON%20%7C%20C%2B%2B17-orange)

---

## Results

Benchmarked on **Qwen2.5-0.5B**, **M4 base (16 GB unified memory)**:

| Method | WikiText-2 PPL | ΔPPL | Decode tok/s | Memory | Compression |
|---|---|---|---|---|---|
| fp16 (MLX) | 18.96 | — | 17.3 | 988 MB | 1× |
| RTN-int4 | 25.94 | +6.98 | 13.2 | 463 MB | 2.1× |
| **GPTQ-int4** | **21.34** | **+2.38** | 13.1 | 463 MB | **2.1×** |
| AWQ-int4 | 34.87 | +15.91 | 14.4 | 463 MB | 2.1× |

Projection weights alone: **190 MB vs 748 MB fp16 = 3.9× compression**.
NEON kernel microbenchmark (simulated 7B decode): **4.1× speedup over fp16 numpy**.

---

## What's inside

Three pillars, all implemented from first principles:

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

All three methods use **group-wise symmetric int4** (group size 128): two nibbles per byte, one fp32 scale per group.

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

**Speculative decoding** (`engine/speculative.py`): a small draft model proposes K tokens; the target verifies K+1 in one forward pass. Rejection sampling (`accept with prob min(1, p_target/p_draft)`) is provably lossless — with draft=target and greedy decoding, output matches non-speculative exactly (verified by test).

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Build NEON kernel (macOS, requires clang++)
bash siliconfer/kernels/neon/build_kernel.sh

# Tests (81 tests, ~7s)
pytest tests/

# Interactive generation
python scripts/run.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --method gptq \
    --prompt "Explain attention in transformers:" \
    --max_tokens 200

# Full benchmark matrix (fp16 + RTN + GPTQ + AWQ, ~25 min)
python scripts/run_benchmarks.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --full --ppl --max_ppl_tokens 5000
```

---

## Repository layout

```
siliconfer/
├── siliconfer/
│   ├── model/          # config, llama blocks, RoPE, RMSNorm, GQA attention, KV cache
│   ├── quant/          # primitives, rtn.py, gptq.py, awq.py, calibration.py
│   ├── kernels/neon/   # q4_gemv.cpp, q4_gemm.cpp, pybind11 bindings, CMakeLists.txt
│   ├── engine/         # generate.py, q4_loader.py, speculative.py
│   └── eval/           # perplexity.py, bench.py
├── eval/               # plots.py (PPL bar, throughput bar, memory bar, roofline)
├── scripts/            # run.py, quantize.py, benchmark.py, run_benchmarks.py
└── tests/              # 81 unit + integration tests
```

---

## Stack

- **Python 3.11+**, **MLX** (Apple array framework), **NumPy**
- **C++17 + ARM NEON intrinsics**, **pybind11**, CMake
- **PyTorch / HuggingFace** — reference baselines only
- **Hardware:** MacBook M4 base · 16 GB unified memory · ARMv9.2a (NEON + I8MM + SME2)
