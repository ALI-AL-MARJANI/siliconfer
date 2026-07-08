"""Phase 7 benchmark runner: produces the full results table.

Measures tok/s (prefill + decode), weight memory, and optionally PPL for:
  fp16 (MLX baseline), RTN-int4, GPTQ-int4 (--full), AWQ-int4 (--full),
  HQQ-int4 (--full), SINQ-int4 (--full)

Results are saved to results.json and plots are auto-generated.

Usage:
    # Fast: fp16 baseline + RTN-int4 (~2 min)
    python scripts/run_benchmarks.py --model_id Qwen/Qwen2.5-0.5B

    # With PPL (fast estimate, ~5 min extra):
    python scripts/run_benchmarks.py --model_id Qwen/Qwen2.5-0.5B --ppl --max_ppl_tokens 5000

    # Full matrix including GPTQ + AWQ + HQQ + SINQ PPL (~25 min):
    python scripts/run_benchmarks.py --model_id Qwen/Qwen2.5-0.5B --full

Output:
    results/results.json   — machine-readable results
    results/*.png          — bar charts + roofline plot
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.engine.q4_loader import load_q4_model
from siliconfer.eval.bench import measure_throughput, measure_memory
from siliconfer.kernels.neon import NEON_AVAILABLE


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_model_dir(model_id: str | None, model_dir: str | None) -> str:
    if model_dir is not None:
        return model_dir
    if model_id is None:
        model_id = "Qwen/Qwen2.5-0.5B"
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id} ...")
    return snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
    )


def _make_prompt(tokenizer_id: str, n_tokens: int = 64) -> mx.array:
    """Tokenize a fixed prompt padded to n_tokens."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    prompt = (
        "The history of artificial intelligence began in the mid-20th century, "
        "when researchers first started exploring the possibility of machines "
        "that could reason and learn. Early systems were rule-based, but modern "
        "neural networks have transformed the field entirely, enabling language "
        "models and vision systems of remarkable capability."
    )
    ids = tok.encode(prompt)[:n_tokens]
    return mx.array(ids)[None, :]   # [1, T]


def _run_ppl(model, config, tokenizer_id: str, max_tokens: int | None) -> float:
    from siliconfer.eval.perplexity import compute_perplexity
    return compute_perplexity(
        model, config, tokenizer_id,
        seq_len=512,            # shorter chunks → faster; still valid estimate
        max_tokens=max_tokens,
        verbose=False,
    )


def _bench_one(
    label: str,
    model: LlamaModel,
    prompt_ids: mx.array,
    n_decode: int,
    n_runs: int,
) -> dict:
    print(f"  measuring throughput ({n_runs} runs × {n_decode} decode tokens) ...", flush=True)
    tput = measure_throughput(model, prompt_ids, n_decode_tokens=n_decode, n_runs=n_runs)
    mem  = measure_memory(model)
    print(f"    decode {tput['decode_tok_s']:.1f} tok/s | "
          f"prefill {tput['prefill_tok_s']:.0f} tok/s | "
          f"total mem {mem['total_mb']:.0f} MB")
    return {**tput, **mem}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",   default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_dir",  default=None)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--n_decode",   type=int, default=100,
                        help="Decode tokens per throughput run.")
    parser.add_argument("--n_runs",     type=int, default=3,
                        help="Runs per method (median taken).")
    parser.add_argument("--ppl",        action="store_true",
                        help="Also measure WikiText-2 PPL for fp16 and RTN.")
    parser.add_argument("--full",       action="store_true",
                        help="Run GPTQ and AWQ methods too (slow).")
    parser.add_argument("--max_ppl_tokens", type=int, default=None,
                        help="Limit PPL evaluation tokens (e.g. 5000 for quick estimate).")
    parser.add_argument("--calib_seqs", type=int, default=64)
    parser.add_argument("--calib_len",  type=int, default=512)
    parser.add_argument("--out_dir",    default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    model_dir   = _resolve_model_dir(args.model_id, args.model_dir)
    tokenizer_id = args.model_id or "Qwen/Qwen2.5-0.5B"

    print(f"\n{'='*60}")
    print(f" siliconfer Phase 7 benchmarks")
    print(f"  model_id   = {tokenizer_id}")
    print(f"  group_size = {args.group_size}")
    print(f"  NEON_AVAILABLE = {NEON_AVAILABLE}")
    print(f"{'='*60}\n")

    prompt_ids = _make_prompt(tokenizer_id, n_tokens=64)
    results: dict = {}

    # ------------------------------------------------------------------ fp16
    print("=== fp16 (MLX baseline) ===")
    t0 = time.perf_counter()
    model_fp16, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
    mx.eval(model_fp16.parameters())
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    results["fp16"] = _bench_one("fp16", model_fp16, prompt_ids, args.n_decode, args.n_runs)

    if args.ppl:
        print(f"  measuring PPL (max_tokens={args.max_ppl_tokens}) ...", flush=True)
        ppl_fp16 = _run_ppl(model_fp16, config, tokenizer_id, args.max_ppl_tokens)
        results["fp16"]["ppl"] = ppl_fp16
        print(f"    WikiText-2 PPL = {ppl_fp16:.2f}")

    del model_fp16

    # ------------------------------------------------------------------ RTN
    print("\n=== RTN-int4 ===")
    t0 = time.perf_counter()
    model_rtn, _ = load_q4_model(
        model_dir, method="rtn", group_size=args.group_size, verbose=True,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    results["rtn"] = _bench_one("rtn", model_rtn, prompt_ids, args.n_decode, args.n_runs)

    if args.ppl:
        print(f"  measuring PPL ...", flush=True)
        ppl_rtn = _run_ppl(model_rtn, config, tokenizer_id, args.max_ppl_tokens)
        results["rtn"]["ppl"] = ppl_rtn
        print(f"    WikiText-2 PPL = {ppl_rtn:.2f}")

    del model_rtn

    # ------------------------------------------------------------------ GPTQ
    if args.full:
        print("\n=== GPTQ-int4 ===")
        t0 = time.perf_counter()
        model_gptq, _ = load_q4_model(
            model_dir, method="gptq", group_size=args.group_size,
            calib_model_id=tokenizer_id,
            n_calib_seqs=args.calib_seqs, calib_len=args.calib_len, verbose=True,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        results["gptq"] = _bench_one("gptq", model_gptq, prompt_ids, args.n_decode, args.n_runs)

        if args.ppl:
            print("  measuring PPL ...", flush=True)
            ppl_gptq = _run_ppl(model_gptq, config, tokenizer_id, args.max_ppl_tokens)
            results["gptq"]["ppl"] = ppl_gptq
            print(f"    WikiText-2 PPL = {ppl_gptq:.2f}")

        del model_gptq

    # ------------------------------------------------------------------ AWQ
    if args.full:
        print("\n=== AWQ-int4 ===")
        t0 = time.perf_counter()
        model_awq, _ = load_q4_model(
            model_dir, method="awq", group_size=args.group_size,
            calib_model_id=tokenizer_id,
            n_calib_seqs=args.calib_seqs, calib_len=args.calib_len, verbose=True,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        results["awq"] = _bench_one("awq", model_awq, prompt_ids, args.n_decode, args.n_runs)

        if args.ppl:
            print("  measuring PPL ...", flush=True)
            ppl_awq = _run_ppl(model_awq, config, tokenizer_id, args.max_ppl_tokens)
            results["awq"]["ppl"] = ppl_awq
            print(f"    WikiText-2 PPL = {ppl_awq:.2f}")

        del model_awq

    # ------------------------------------------------------------------ HQQ
    if args.full:
        print("\n=== HQQ-int4 ===")
        t0 = time.perf_counter()
        model_hqq, _ = load_q4_model(
            model_dir, method="hqq", group_size=args.group_size, verbose=True,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        results["hqq"] = _bench_one("hqq", model_hqq, prompt_ids, args.n_decode, args.n_runs)

        if args.ppl:
            print("  measuring PPL ...", flush=True)
            ppl_hqq = _run_ppl(model_hqq, config, tokenizer_id, args.max_ppl_tokens)
            results["hqq"]["ppl"] = ppl_hqq
            print(f"    WikiText-2 PPL = {ppl_hqq:.2f}")

        del model_hqq

    # ------------------------------------------------------------------ SINQ
    if args.full:
        print("\n=== SINQ-int4 ===")
        t0 = time.perf_counter()
        model_sinq, _ = load_q4_model(
            model_dir, method="sinq", group_size=args.group_size, verbose=True,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        results["sinq"] = _bench_one("sinq", model_sinq, prompt_ids, args.n_decode, args.n_runs)

        if args.ppl:
            print("  measuring PPL ...", flush=True)
            ppl_sinq = _run_ppl(model_sinq, config, tokenizer_id, args.max_ppl_tokens)
            results["sinq"]["ppl"] = ppl_sinq
            print(f"    WikiText-2 PPL = {ppl_sinq:.2f}")

        del model_sinq

    # ------------------------------------------------------------------ kernel stats
    results["kernel"] = {
        "neon_bw_gbs":    6.7,     # from Phase 5 benchmark
        "ai_q4":          0.5,     # approx: 1 FLOP per nibble byte
        "m4_peak_gbs":   120.0,
    }

    # ------------------------------------------------------------------ save + plot
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Generate plots via importlib (eval/ is not a package, load by path)
    print("\nGenerating plots ...")
    try:
        import sys
        import importlib.util
        _plots_path = Path(__file__).resolve().parent.parent / "eval" / "plots.py"
        _spec = importlib.util.spec_from_file_location("siliconfer_plots", _plots_path)
        _mod  = importlib.util.module_from_spec(_spec)
        sys.argv = ["plots.py", str(out_json), "--out_dir", str(out_dir)]
        _spec.loader.exec_module(_mod)
        _mod.main()
    except Exception as e:
        print(f"  [warn] plot generation failed: {e}")
        print("  Run manually: python eval/plots.py results/results.json")


if __name__ == "__main__":
    main()
