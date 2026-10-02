# RUNBOOK

Exact commands to reproduce every file under `results/`. Durations are taken
from the `elapsed_s` fields of the result files and from the job logs on the
reference machine.

**Reference machine:** MacBook, Apple M4 (4 performance + 6 efficiency cores),
16 GB unified memory, macOS 15.6. Nothing here needs a discrete GPU or more
than 16 GB. About 2 GB of free disk for the model and datasets.

**Rules**

- Run long jobs one at a time. Two model evaluations at once push a 16 GB
  machine into memory pressure and slow both by an order of magnitude.
- Keep the machine awake: prefix long commands with `caffeinate -i`.
- Speed benchmarks (section 3) need an otherwise idle machine. Perplexity
  runs are not timing-sensitive.
- Do not edit the code while a job is running: later steps of the same job
  would import a different version of it.
- Every script writes a JSON with its full configuration and an `env` block
  (git SHA, dirty flag, hardware, library versions, kernel backend).

## 0. Setup (about 5 minutes, plus a 1 GB model download on first run)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash siliconfer/kernels/neon/build_kernel.sh   # C++/NEON extension; needs clang++ (Xcode CLT)
python -m pytest tests -q                      # ~5 s, no model download
python -m pytest tests -q --run-integration    # ~10 s, needs the model in the HF cache
```

`Qwen/Qwen2.5-0.5B`, WikiText-2 and a streamed slice of C4 are downloaded
from the Hugging Face Hub on first use.

## 1. fp16 reference validation (about 9 minutes)

Checks this repo's forward pass against `transformers` on the full
WikiText-2 test split (298,938 tokens, same windows, float32).

```bash
python scripts/validate_fp16_reference.py --model_id Qwen/Qwen2.5-0.5B
```

Output: `results/ppl/fp16_reference_validation.json`.

## 2. Perplexity matrix

```bash
caffeinate -i bash scripts/run_ppl_matrix.sh            # quick tier  -> results/ppl/
FULL=1 caffeinate -i bash scripts/run_ppl_matrix.sh     # full tier   -> results/ppl_full/
```

Both tiers run the same 25 configurations on WikiText-2 (23 on C4):
fp16, RTN, SINQ-style, HQQ-style, GPTQ × 3 calibration seeds, AWQ × 3 seeds,
AWQ with block loss and clipping × 3 seeds (plus each option alone, one
seed, WikiText-2 only), the asymmetric variants of RTN / SINQ / GPTQ / AWQ,
and MLX's own quantizer as an external reference at group sizes 128 and 64.
Runs are sequential and resumable: a run whose output file exists is
skipped, so delete a file to force that run again.

| Tier | Tokens scored | Duration |
|---|---|---|
| quick | 32,768 per dataset | 2 h 41 min for the 48 runs |
| full | 296,960 (WikiText-2, 145 windows), 131,072 (C4) | 6 h 31 min for the 48 runs |

A single configuration:

```bash
python scripts/eval_ppl.py --method gptq --calib_seed 1 --max_tokens 32768
python scripts/eval_ppl.py --method awq --awq_block_loss --awq_clip --calib_seed 0 \
    --dataset c4 --c4_n_seqs 16 --c4_seed 0
```

`--c4_seed` selects the C4 test documents and must be the same for every run
being compared; each C4 result file records a SHA-256 of the scored token ids.

## 3. Speed benchmarks and per-claim evaluations

```bash
caffeinate -i bash scripts/run_all_benchmarks.sh
```

Runs, in order. The whole job took 57 minutes on the reference machine
(GEMV under 1 min, decode 10 min, mlx-lm reference 3 min, Metal attention and
TTFT 2 min, KV cache 2 min, speculative 9 min, mixed precision 5 min,
draft-head training 24 min):

| Step | Output |
|---|---|
| `scripts/bench_gemv.py` — one matrix-vector product, NEON int4 vs Accelerate / MLX | `results/bench/gemv.json` |
| `scripts/bench_decode.py` — end-to-end generation, fp16 vs int4 on both backends | `results/bench/decode_Qwen2.5-0.5B.json` |
| `scripts/bench_mlx_lm_reference.py` — same model through mlx-lm, in a throwaway environment via `uvx` | `results/bench/decode_mlx_lm_reference_Qwen2.5-0.5B.json` |
| `scripts/bench_metal_attention.py` — fused int8-KV attention kernel vs MLX attention | `results/bench/metal_attention.json` |
| `scripts/bench_ttft.py` — time to first token, cold vs warmed, fresh process per trial | `results/bench/ttft.json` |
| `scripts/eval_kv_cache.py` — int8 KV cache perplexity through incremental decode | `results/claims/kv_cache.json` |
| `scripts/eval_speculative.py` — distribution check, acceptance rate, throughput | `results/claims/speculative.json` |
| `scripts/quantize.py --method mixed` (2-bit and 3-bit low tier) | `results/claims/mixed_precision_{2,3}bit.json` |
| `scripts/train_draft_head.py` — 3 seeds | `results/claims/draft_head.json` |

The mlx-lm reference needs [`uv`](https://docs.astral.sh/uv/); without it
that step is skipped.

## 4. Tables and figures

```bash
python scripts/make_results_tables.py     # results/SUMMARY.md — every README table is copied from it
python eval/plots.py                      # results/figures/*.png
```

## 5. Not run (and why)

- **A second model size.** `python scripts/bench_decode.py --model_id Qwen/Qwen2.5-1.5B`
  and `MODEL=Qwen/Qwen2.5-1.5B bash scripts/run_ppl_matrix.sh` are ready to
  run; the model is a 3.1 GB download and the reference machine had 8 GB of
  free disk. Expect roughly 3× the durations above.
- **Group size 64** for this repo's methods: `--group_size 64` is supported by
  `scripts/eval_ppl.py` and writes `*-g64_*.json`; it is not in the matrix.
- **llama.cpp** as a CPU baseline, and **lm-evaluation-harness** downstream
  tasks: no script.
