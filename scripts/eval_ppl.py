"""Standard-protocol perplexity evaluation.

Single consistent protocol for every number that goes in a results table:
full WikiText-2 raw test split (or a fixed C4 validation subset), context
length 2048, float32 NLL. Writes one JSON per run to results/ppl/, including
full config, git SHA, hardware, and library versions.

This replaces ad-hoc PPL numbers computed with different seq_len/max_tokens
settings in different scripts (see docs/eval-protocol.md for why that
happened and why it matters). scripts/quantize.py remains for algorithm-level
(fake-quant) experiments, but any perplexity reported in the README must come
from this script.

Usage:
    python scripts/eval_ppl.py --model_id Qwen/Qwen2.5-0.5B --method fp16
    python scripts/eval_ppl.py --model_id Qwen/Qwen2.5-0.5B --method awq --calib_seed 1
    python scripts/eval_ppl.py --model_id Qwen/Qwen2.5-0.5B --method gptq --dataset c4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from siliconfer.eval.env_info import collect_env_info
from siliconfer.eval.perplexity import compute_perplexity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_dir", default=None, help="Local model dir; downloads model_id if omitted.")
    parser.add_argument("--method", required=True, choices=["fp16", "rtn", "gptq", "awq", "hqq", "sinq", "mlx_native"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--sym", dest="sym", action="store_true", default=True)
    parser.add_argument("--asym", dest="sym", action="store_false")
    parser.add_argument("--calib_seed", type=int, default=42,
                         help="Calibration-sequence sampling seed (gptq/awq only). "
                              "Run with 3 different seeds and report mean +/- std.")
    parser.add_argument("--backend", default="neon", choices=["neon", "mlx"],
                         help="Kernel for the packed int4 layers: this repo's NEON CPU kernel, "
                              "or MLX's native quantized matmul on the GPU (same codes).")
    parser.add_argument("--awq_block_loss", action="store_true",
                         help="AWQ: score alpha on the enclosing block's output (as llm-awq does).")
    parser.add_argument("--awq_clip", action="store_true",
                         help="AWQ: per-group weight clipping after scaling (llm-awq's auto_clip).")
    parser.add_argument("--n_calib_seqs", type=int, default=128)
    parser.add_argument("--calib_len", type=int, default=512)
    parser.add_argument("--c4_n_seqs", type=int, default=256)
    parser.add_argument("--c4_seed", type=int, default=0,
                         help="Shuffle seed that selects the C4 test documents. Keep it fixed "
                              "across all methods/calibration seeds so every run is scored on "
                              "the same text (independent of --calib_seed).")
    parser.add_argument("--max_tokens", type=int, default=None,
                         help="Cap wikitext2 tokens for a quick smoke run; omit for the full test set.")
    parser.add_argument("--out_dir", default="results/ppl")
    args = parser.parse_args()

    t0 = time.time()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            repo_id=args.model_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
        )

    if args.method == "fp16":
        from siliconfer.model.llama import LlamaModel
        model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        mx.eval(model.parameters())
    elif args.method == "mlx_native":
        # External reference: MLX's own 4-bit quantizer (mx.quantize, affine min/max
        # per group) applied to the same projections this repo quantizes, evaluated
        # as fake-quant through the same fp16 forward pass. Not this repo's method.
        from siliconfer.model.llama import LlamaModel
        model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        for layer in model.layers:
            for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                  (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
                for name in names:
                    lin = getattr(parent, name)
                    w = lin.weight.astype(mx.float32)
                    w_q, scales, biases = mx.quantize(w, group_size=args.group_size, bits=4)
                    lin.weight = mx.dequantize(w_q, scales, biases, group_size=args.group_size,
                                               bits=4).astype(lin.weight.dtype)
        mx.eval(model.parameters())
    else:
        from siliconfer.engine.q4_loader import load_q4_model
        model, config = load_q4_model(
            model_dir,
            method=args.method,
            group_size=args.group_size,
            sym=args.sym,
            calib_model_id=args.model_id,
            n_calib_seqs=args.n_calib_seqs,
            calib_len=args.calib_len,
            calib_seed=args.calib_seed,
            awq_block_loss=args.awq_block_loss,
            awq_clip=args.awq_clip,
            backend=args.backend,
            verbose=True,
        )

    ppl, eval_info = compute_perplexity(
        model, config, args.model_id,
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        dataset=args.dataset,
        c4_n_seqs=args.c4_n_seqs,
        seed=args.c4_seed,
        verbose=True,
        return_info=True,
    )

    elapsed_s = round(time.time() - t0, 1)

    result = {
        "model_id": args.model_id,
        "method": args.method,
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "group_size": args.group_size,
        "sym": args.sym,
        "calib_seed": args.calib_seed if args.method in ("gptq", "awq") else None,
        "n_calib_seqs": args.n_calib_seqs if args.method in ("gptq", "awq") else None,
        "calib_len": args.calib_len if args.method in ("gptq", "awq") else None,
        "awq_block_loss": args.awq_block_loss if args.method == "awq" else None,
        "awq_clip": args.awq_clip if args.method == "awq" else None,
        "backend": args.backend if args.method not in ("fp16", "mlx_native") else None,
        "max_tokens": args.max_tokens,
        "c4_n_seqs": args.c4_n_seqs if args.dataset == "c4" else None,
        "c4_seed": args.c4_seed if args.dataset == "c4" else None,
        **eval_info,
        "ppl": ppl,
        "elapsed_s": elapsed_s,
        "env": collect_env_info(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{args.calib_seed}" if args.method in ("gptq", "awq") else ""
    variant = ""
    if args.method == "awq":
        variant = ("-blockloss" if args.awq_block_loss else "") + ("-clip" if args.awq_clip else "")
    if args.group_size != 128:
        variant += f"-g{args.group_size}"
    if not args.sym and args.method != "mlx_native":
        variant += "-asym"
    fname = f"{args.method}{variant}_{args.dataset}_seq{args.seq_len}{seed_tag}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n[eval_ppl] {args.method} / {args.dataset} / seq_len={args.seq_len}: "
          f"PPL={ppl:.2f}  ({elapsed_s:.0f}s)")
    print(f"[eval_ppl] wrote {out_path}")


if __name__ == "__main__":
    main()
