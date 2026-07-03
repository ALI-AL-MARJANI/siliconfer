"""End-to-end quantization script: fp16 / RTN / GPTQ / AWQ PPL on WikiText-2.

Usage:
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method gptq
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method awq
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method all --max_tokens 10000
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.quant.rtn import apply_rtn
from siliconfer.quant.calibration import load_calibration_sequences
from siliconfer.quant.gptq import apply_gptq
from siliconfer.quant.awq import apply_awq
from siliconfer.eval.perplexity import compute_perplexity


def _resolve_model_dir(model_id: str, model_dir: str | None) -> str:
    if model_dir is not None:
        return model_dir
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id} ...")
    return snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_dir", default=None,
                        help="Local path to model dir (skips HF download).")
    parser.add_argument("--method", default="rtn",
                        choices=["rtn", "gptq", "awq", "all"],
                        help="Quantization method(s) to run (default: rtn).")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--asym", action="store_true",
                        help="Use asymmetric quantization (default: symmetric).")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Limit total tokens evaluated (for quick runs).")
    parser.add_argument("--calib_seqs", type=int, default=128,
                        help="Number of calibration sequences for GPTQ (default: 128).")
    parser.add_argument("--calib_len", type=int, default=512,
                        help="Sequence length for GPTQ calibration (default: 512).")
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model_id, args.model_dir)
    sym = not args.asym
    quant_suffix = f"g{args.group_size} {'asym' if args.asym else 'sym'}"
    run_rtn  = args.method in ("rtn",  "all")
    run_gptq = args.method in ("gptq", "all")
    run_awq  = args.method in ("awq",  "all")

    results: list[tuple[str, float]] = []

    # ------------------------------------------------------------------ fp16
    print("\n=== Loading fp16 model ===")
    t0 = time.perf_counter()
    model_fp16, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
    mx.eval(model_fp16.parameters())
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print("\n--- fp16 WikiText-2 PPL ---")
    ppl_fp16 = compute_perplexity(
        model_fp16, config, args.model_id,
        seq_len=args.seq_len, max_tokens=args.max_tokens,
    )
    results.append(("fp16 (reference)", ppl_fp16))
    del model_fp16

    # ------------------------------------------------------------------ RTN
    if run_rtn:
        label = f"RTN-int4 {quant_suffix}"
        print(f"\n=== Applying {label} ===")
        model_rtn, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        apply_rtn(model_rtn, group_size=args.group_size, sym=sym)
        mx.eval(model_rtn.parameters())

        print(f"\n--- {label} WikiText-2 PPL ---")
        ppl_rtn = compute_perplexity(
            model_rtn, config, args.model_id,
            seq_len=args.seq_len, max_tokens=args.max_tokens,
        )
        results.append((label, ppl_rtn))
        del model_rtn

    # ------------------------------------------------------------------ GPTQ
    if run_gptq:
        label = f"GPTQ-int4 {quant_suffix}"
        print(f"\n=== Applying {label} ===")
        print(f"  Collecting calibration data ({args.calib_seqs} seqs × {args.calib_len} tokens) ...")
        calib_seqs = load_calibration_sequences(
            args.model_id, n_seqs=args.calib_seqs, seq_len=args.calib_len,
        )

        model_gptq, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        apply_gptq(
            model_gptq, calib_seqs,
            group_size=args.group_size, sym=sym, verbose=True,
        )
        mx.eval(model_gptq.parameters())

        print(f"\n--- {label} WikiText-2 PPL ---")
        ppl_gptq = compute_perplexity(
            model_gptq, config, args.model_id,
            seq_len=args.seq_len, max_tokens=args.max_tokens,
        )
        results.append((label, ppl_gptq))
        del model_gptq

    # ------------------------------------------------------------------- AWQ
    if run_awq:
        label = f"AWQ-int4  {quant_suffix}"
        print(f"\n=== Applying {label} ===")
        print(f"  Collecting calibration data ({args.calib_seqs} seqs × {args.calib_len} tokens) ...")
        calib_seqs = load_calibration_sequences(
            args.model_id, n_seqs=args.calib_seqs, seq_len=args.calib_len,
        )

        model_awq, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        apply_awq(
            model_awq, calib_seqs,
            group_size=args.group_size, sym=sym, fold_scales=False, verbose=True,
        )
        mx.eval(model_awq.parameters())

        print(f"\n--- {label} WikiText-2 PPL ---")
        ppl_awq = compute_perplexity(
            model_awq, config, args.model_id,
            seq_len=args.seq_len, max_tokens=args.max_tokens,
        )
        results.append((label, ppl_awq))
        del model_awq

    # ------------------------------------------------------------------ table
    ref_ppl = results[0][1]
    print("\n" + "=" * 55)
    print(f"{'Method':<34} {'PPL':>8} {'ΔPPL':>8}")
    print("-" * 55)
    for name, ppl in results:
        delta = f"+{ppl - ref_ppl:.2f}" if ppl != ref_ppl else "—"
        print(f"{name:<34} {ppl:>8.2f} {delta:>8}")
    print("=" * 55)


if __name__ == "__main__":
    main()
