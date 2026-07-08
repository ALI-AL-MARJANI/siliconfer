"""End-to-end quantization script: fp16 / RTN / GPTQ / AWQ / HQQ / SINQ PPL on WikiText-2.

Usage:
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method gptq
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method awq
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method hqq
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method sinq
    python scripts/quantize.py --model_id Qwen/Qwen2.5-0.5B --method all --max_tokens 10000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.quant.rtn import apply_rtn
from siliconfer.quant.calibration import load_calibration_sequences
from siliconfer.quant.gptq import apply_gptq
from siliconfer.quant.awq import apply_awq
from siliconfer.quant.hqq import apply_hqq
from siliconfer.quant.sinq import apply_sinq
from siliconfer.quant.mixed_precision import (
    shapley_layer_sensitivity,
    assign_bitwidths,
    make_block_nll_value_fn,
    apply_mixed_precision,
)
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
                        choices=["rtn", "gptq", "awq", "hqq", "sinq", "mixed", "all"],
                        help="Quantization method(s) to run (default: rtn).")
    parser.add_argument("--mixed_budget_ratio", type=float, default=0.75,
                        help="Fraction of full-4-bit packed footprint to target for "
                             "--method mixed (e.g. 0.75 = 25%% smaller than uniform int4).")
    parser.add_argument("--mixed_permutations", type=int, default=6,
                        help="Shapley permutation-sampling count for --method mixed. "
                             "Each permutation costs ~n_layers forward passes on the "
                             "small mixed-precision calibration set.")
    parser.add_argument("--mixed_calib_seqs", type=int, default=2,
                        help="Number of short sequences used only to *rank* layer "
                             "sensitivity for --method mixed (kept small — evaluated "
                             "many times, unlike GPTQ/AWQ's one-pass calibration).")
    parser.add_argument("--mixed_calib_len", type=int, default=128,
                        help="Sequence length for --method mixed's sensitivity calibration.")
    parser.add_argument("--mixed_low_bits", type=int, default=2,
                        help="Bit-width for demoted blocks in --method mixed (default: 2). "
                             "3 was found to meaningfully close the gap to uniform HQQ-int4 "
                             "vs plain 2-bit — see NOTES.md.")
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
    run_hqq  = args.method in ("hqq",  "all")
    run_sinq = args.method in ("sinq", "all")
    run_mixed = args.method in ("mixed", "all")

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

    # ------------------------------------------------------------------ HQQ
    if run_hqq:
        # HQQ is inherently asymmetric (its whole point is the zero-point fit),
        # so the sym/asym suffix used by the other methods doesn't apply here.
        label = f"HQQ-int4  g{args.group_size}"
        print(f"\n=== Applying {label} ===")
        print("  (no calibration data needed)")

        model_hqq, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        apply_hqq(model_hqq, group_size=args.group_size, verbose=True)
        mx.eval(model_hqq.parameters())

        print(f"\n--- {label} WikiText-2 PPL ---")
        ppl_hqq = compute_perplexity(
            model_hqq, config, args.model_id,
            seq_len=args.seq_len, max_tokens=args.max_tokens,
        )
        results.append((label, ppl_hqq))
        del model_hqq

    # ----------------------------------------------------------------- SINQ
    if run_sinq:
        label = f"SINQ-int4 {quant_suffix}"
        print(f"\n=== Applying {label} ===")
        print("  (no calibration data needed)")

        model_sinq, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        apply_sinq(model_sinq, group_size=args.group_size, sym=sym, verbose=True)
        mx.eval(model_sinq.parameters())

        print(f"\n--- {label} WikiText-2 PPL ---")
        ppl_sinq = compute_perplexity(
            model_sinq, config, args.model_id,
            seq_len=args.seq_len, max_tokens=args.max_tokens,
        )
        results.append((label, ppl_sinq))
        del model_sinq

    # ---------------------------------------------------------------- Mixed
    if run_mixed:
        low_bits = args.mixed_low_bits
        label = f"Mixed-{low_bits}/4bit g{args.group_size} ({args.mixed_budget_ratio:.0%} budget)"
        print(f"\n=== Applying {label} ===")

        model_mixed, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        mx.eval(model_mixed.parameters())
        n_layers = len(model_mixed.layers)

        print(f"  Loading {args.mixed_calib_seqs} short sequences "
              f"({args.mixed_calib_len} tokens) for layer-sensitivity ranking ...")
        sens_seqs = load_calibration_sequences(
            args.model_id, n_seqs=args.mixed_calib_seqs, seq_len=args.mixed_calib_len,
        )
        calib_input_ids = mx.concatenate(sens_seqs, axis=0)

        print(f"  Estimating per-block Shapley sensitivity "
              f"({args.mixed_permutations} permutations x {n_layers} layers, low_bits={low_bits}) ...")
        value_fn = make_block_nll_value_fn(
            model_mixed, calib_input_ids, group_size=args.group_size, low_bits=low_bits,
        )
        sensitivity = shapley_layer_sensitivity(
            value_fn, n_layers=n_layers, n_permutations=args.mixed_permutations,
        )
        print(f"  Sensitivity (higher = keep at 4-bit): "
              f"{np.array2string(sensitivity, precision=3)}")

        # Every decoder block has an identical parameter count in this architecture
        # (uniform hidden_size/intermediate_size across depth), so a single scalar
        # "bytes per block at 4-bit" applies to all of them — see mixed_precision.py.
        bytes_per_block_high = np.full(n_layers, 1.0)  # relative units; ratio is what matters
        budget = n_layers * args.mixed_budget_ratio
        bits_per_block = assign_bitwidths(
            sensitivity, bytes_per_block_high, memory_budget_bytes=budget, low_bits=low_bits,
        )
        n_demoted = int((bits_per_block == low_bits).sum())
        print(f"  Assigned: {n_layers - n_demoted} blocks at 4-bit, {n_demoted} blocks at {low_bits}-bit")

        apply_mixed_precision(
            model_mixed, bits_per_block, group_size=args.group_size, low_bits=low_bits, verbose=True,
        )
        mx.eval(model_mixed.parameters())

        print(f"\n--- {label} WikiText-2 PPL ---")
        ppl_mixed = compute_perplexity(
            model_mixed, config, args.model_id,
            seq_len=args.seq_len, max_tokens=args.max_tokens,
        )
        results.append((label, ppl_mixed))
        del model_mixed

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
