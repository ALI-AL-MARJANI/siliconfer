"""siliconfer run.py — end-to-end int4 text generation with the NEON kernel.

Loads a model, quantizes to int4, and generates streaming text.

Usage:
    # Fast: RTN quantization (~5s load)
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --prompt "The universe began"

    # Better quality: GPTQ (~2min calibration)
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --method gptq --prompt "The universe"

    # Use local model dir:
    python scripts/run.py --model_dir /path/to/model --prompt "Hello"

    # Speculative decoding (Phase 8/9d) — draft model defaults to the same
    # weights as the target if --draft_model_id/--draft_model_dir aren't
    # given (a correctness demo, not a speed win, since draft=target means
    # every draft token is exactly as expensive to verify as it was to
    # propose — pass a genuinely smaller model sharing the same tokenizer
    # for a real speedup):
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --speculative --K 4 \\
        --prompt "Explain attention in transformers:"

    # Dynamic speculation depth (adapts K round-to-round, still exactly
    # lossless — see CLAUDE.md §9d):
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --speculative --dynamic_K

    # Mixed precision (Phase 8): keep the first + last layers in fp16 for
    # lower PPL at a small memory cost. --mixed_precision is shorthand for
    # --skip_layers with the first and last layer indices; pass --skip_layers
    # directly for manual control (e.g. "--skip_layers 0,1,22,23").
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --mixed_precision
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

from siliconfer.engine.q4_loader import load_q4_model
from siliconfer.engine.generate import generate, SamplingParams
from siliconfer.engine.speculative import speculative_generate
from siliconfer.kernels.neon import NEON_AVAILABLE
from siliconfer.model.config import ModelConfig


def _resolve_model_dir(model_id: str | None, model_dir: str | None) -> str:
    if model_dir is not None:
        return model_dir
    if model_id is None:
        model_id = "Qwen/Qwen2.5-0.5B"
    from huggingface_hub import snapshot_download
    print(f"[run] Downloading {model_id} ...")
    return snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
    )


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="siliconfer int4 text generation")
    parser.add_argument("--model_id",   default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_dir",  default=None,
                        help="Local model directory (skips HF download).")
    parser.add_argument("--method",     default="rtn",
                        choices=["rtn", "gptq", "awq", "hqq", "sinq"],
                        help="Quantization method (default: rtn).")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--asym",       action="store_true")
    parser.add_argument("--prompt",     default="The history of artificial intelligence began",
                        help="Text prompt.")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p",      type=float, default=0.9)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--calib_seqs", type=int, default=64,
                        help="Calibration sequences for GPTQ/AWQ.")
    parser.add_argument("--calib_len",  type=int, default=512)
    parser.add_argument("--quantize_kv_cache", action="store_true",
                        help="Store the KV cache as group-wise int8 instead of fp16 (Phase 9b). "
                             "Works with --speculative too (applies to both draft and target).")
    parser.add_argument("--speculative", action="store_true",
                        help="Use speculative decoding (Phase 8/9d) instead of plain generation.")
    parser.add_argument("--draft_model_id", default=None,
                        help="HF model ID for the draft model (speculative only). "
                             "Defaults to --model_id/--model_dir (draft=target: a correctness "
                             "demo, not a speed win — pass a real smaller model for that).")
    parser.add_argument("--draft_model_dir", default=None,
                        help="Local draft model directory (speculative only).")
    parser.add_argument("--K", type=int, default=4,
                        help="Speculation depth per round (speculative only).")
    parser.add_argument("--dynamic_K", action="store_true",
                        help="Adapt K round-to-round based on acceptance (speculative only). "
                             "Exactly lossless by construction — see CLAUDE.md §9d.")
    parser.add_argument("--K_min", type=int, default=1)
    parser.add_argument("--K_max", type=int, default=8)
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Keep the first + last transformer layers in fp16 (Phase 8's "
                             "standard heuristic — those layers are most sensitive to "
                             "quantization). Shorthand for --skip_layers with those two "
                             "indices; --skip_layers overrides this if both are given.")
    parser.add_argument("--skip_layers", default=None,
                        help="Comma-separated layer indices to keep in fp16, e.g. '0,23'. "
                             "Overrides --mixed_precision if both are given.")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model_dir = _resolve_model_dir(args.model_id, args.model_dir)
    calib_model_id = args.model_id or "Qwen/Qwen2.5-0.5B"

    # skip_layers needs num_hidden_layers, which we only get back from
    # load_q4_model *after* loading — peek at config.json directly instead so
    # --mixed_precision's {0, n_layers-1} default can be resolved up front.
    skip_layers: set[int] | None = None
    if args.skip_layers:
        skip_layers = {int(i) for i in args.skip_layers.split(",")}
    elif args.mixed_precision:
        n_layers = ModelConfig.from_json(Path(model_dir) / "config.json").num_hidden_layers
        skip_layers = {0, n_layers - 1}

    print(f"\n{'='*60}")
    print(f" siliconfer — int4 inference on Apple Silicon")
    print(f"  method={args.method}  group_size={args.group_size}  "
          f"NEON={NEON_AVAILABLE}  kv_cache={'int8' if args.quantize_kv_cache else 'fp16'}")
    if skip_layers:
        print(f"  mixed precision: layers {sorted(skip_layers)} stay fp16")
    if args.speculative:
        k_desc = f"dynamic K∈[{args.K_min},{args.K_max}] (start {args.K})" if args.dynamic_K else f"K={args.K}"
        print(f"  speculative decoding: {k_desc}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ load
    t0 = time.perf_counter()
    model, config = load_q4_model(
        model_dir,
        method=args.method,
        group_size=args.group_size,
        sym=not args.asym,
        calib_model_id=calib_model_id,
        n_calib_seqs=args.calib_seqs,
        calib_len=args.calib_len,
        skip_layers=skip_layers,
        verbose=True,
    )
    load_sec = time.perf_counter() - t0
    print(f"[run] Target model ready in {load_sec:.1f}s\n")

    draft_model = None
    if args.speculative:
        draft_dir = _resolve_model_dir(args.draft_model_id, args.draft_model_dir) \
            if (args.draft_model_id or args.draft_model_dir) else model_dir
        if draft_dir == model_dir:
            print("[run] No --draft_model_id/--draft_model_dir given — using the target "
                  "model as its own draft (correctness demo, not a speed win).")
        t0 = time.perf_counter()
        draft_model, _ = load_q4_model(
            draft_dir,
            method=args.method,
            group_size=args.group_size,
            sym=not args.asym,
            calib_model_id=args.draft_model_id or calib_model_id,
            n_calib_seqs=args.calib_seqs,
            calib_len=args.calib_len,
            skip_layers=skip_layers,
            verbose=True,
        )
        print(f"[run] Draft model ready in {time.perf_counter()-t0:.1f}s\n")

    # ------------------------------------------------------------------ tokenize
    tokenizer = _load_tokenizer(model_dir)
    eos_id = tokenizer.eos_token_id

    encoded = tokenizer(args.prompt, return_tensors="np")
    prompt_ids = mx.array(encoded["input_ids"])   # [1, T]

    # ------------------------------------------------------------------ generate
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"Prompt: {args.prompt!r}\n")
    print("-" * 60)
    # Print prompt text first
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    # Decode each token as it arrives and stream to stdout
    def _on_token(tok_id: int) -> None:
        text = tokenizer.decode([tok_id], skip_special_tokens=False)
        sys.stdout.write(text)
        sys.stdout.flush()

    if args.speculative:
        result = speculative_generate(
            draft=draft_model,
            target=model,
            prompt_ids=prompt_ids,
            params=params,
            K=args.K,
            eos_token_id=eos_id,
            on_token=_on_token,
            dynamic_K=args.dynamic_K,
            K_min=args.K_min,
            K_max=args.K_max,
            quantize_kv_cache=args.quantize_kv_cache,
        )
    else:
        result = generate(
            model,
            prompt_ids,
            params=params,
            eos_token_id=eos_id,
            on_token=_on_token,
            quantize_kv_cache=args.quantize_kv_cache,
        )

    print("\n" + "-" * 60)

    # ------------------------------------------------------------------ stats
    total_tokens = result.num_decode_tokens
    print(f"\n[stats]")
    print(f"  prefill : {result.num_prefill_tokens} tok in "
          f"{result.prefill_time*1e3:.1f} ms  "
          f"({result.prefill_tok_s:.0f} tok/s)")
    if args.speculative:
        print(f"  decode  : {total_tokens} tok in "
              f"{result.decode_time:.2f} s  "
              f"({result.effective_tok_s:.1f} tok/s effective)")
        print(f"  speculative: {result.total_rounds} rounds, "
              f"acceptance_rate={result.acceptance_rate:.2f}, "
              f"avg_K={result.total_draft_tokens/max(result.total_rounds,1):.1f}")
    else:
        print(f"  decode  : {total_tokens} tok in "
              f"{result.decode_time:.2f} s  "
              f"({result.decode_tok_s:.1f} tok/s)")
    print(f"  method  : {args.method}-int4  group_size={args.group_size}")
    print()


if __name__ == "__main__":
    main()
