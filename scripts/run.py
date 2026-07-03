"""siliconfer run.py — end-to-end int4 text generation with the NEON kernel.

Loads a model, quantizes to int4, and generates streaming text.

Usage:
    # Fast: RTN quantization (~5s load)
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --prompt "The universe began"

    # Better quality: GPTQ (~2min calibration)
    python scripts/run.py --model_id Qwen/Qwen2.5-0.5B --method gptq --prompt "The universe"

    # Use local model dir:
    python scripts/run.py --model_dir /path/to/model --prompt "Hello"
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from siliconfer.engine.q4_loader import load_q4_model
from siliconfer.engine.generate import generate, SamplingParams
from siliconfer.kernels.neon import NEON_AVAILABLE


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
                        choices=["rtn", "gptq", "awq"],
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
    args = parser.parse_args()

    mx.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f" siliconfer — int4 inference on Apple Silicon")
    print(f"  method={args.method}  group_size={args.group_size}  "
          f"NEON={NEON_AVAILABLE}")
    print(f"{'='*60}\n")

    model_dir = _resolve_model_dir(args.model_id, args.model_dir)
    calib_model_id = args.model_id or "Qwen/Qwen2.5-0.5B"

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
        verbose=True,
    )
    load_sec = time.perf_counter() - t0
    print(f"[run] Model ready in {load_sec:.1f}s\n")

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

    result = generate(
        model,
        prompt_ids,
        params=params,
        eos_token_id=eos_id,
        on_token=_on_token,
    )

    print("\n" + "-" * 60)

    # ------------------------------------------------------------------ stats
    total_tokens = result.num_decode_tokens
    print(f"\n[stats]")
    print(f"  prefill : {result.num_prefill_tokens} tok in "
          f"{result.prefill_time*1e3:.1f} ms  "
          f"({result.prefill_tok_s:.0f} tok/s)")
    print(f"  decode  : {total_tokens} tok in "
          f"{result.decode_time:.2f} s  "
          f"({result.decode_tok_s:.1f} tok/s)")
    print(f"  method  : {args.method}-int4  group_size={args.group_size}")
    print()


if __name__ == "__main__":
    main()
