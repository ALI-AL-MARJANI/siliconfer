"""Cross-check the MLX fp16 forward pass against the real HuggingFace
transformers implementation on the same tokens, same chunking, same context
length.

This is the ground-truth check behind the "fp16 PPL validated against
transformers to within X" line in the README: it computes PPL twice --
once with siliconfer's from-scratch MLX decoder, once with
AutoModelForCausalLM -- on the identical non-overlapping seq_len windows,
and reports both plus the difference. It does not assert a threshold; it
reports the measured number.

Usage:
    python scripts/validate_fp16_reference.py --model_id Qwen/Qwen2.5-0.5B --max_tokens 20000
    python scripts/validate_fp16_reference.py --model_id Qwen/Qwen2.5-0.5B   # full test set (slower)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.eval.env_info import collect_env_info
from siliconfer.eval.perplexity import load_wikitext2_test_tokens
from siliconfer.model.llama import LlamaModel


def _torch_ppl(model_id: str, token_ids: np.ndarray, seq_len: int, device: str) -> float:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    n_chunks = (len(token_ids) - 1) // seq_len
    nlls = []
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_len
            chunk = token_ids[start : start + seq_len + 1]
            inp = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None, :]
            target = torch.tensor(chunk[1:], dtype=torch.long, device=device)

            logits = model(inp).logits[0].float()  # [T, vocab]
            nll = torch.nn.functional.cross_entropy(logits, target).item()
            nlls.append(nll)
            print(f"  [torch] chunk {i+1}/{n_chunks}  nll={nll:.4f}", flush=True)

    return math.exp(float(np.mean(nlls)))


def _mlx_ppl(model_dir: str, model_id: str, token_ids: np.ndarray, seq_len: int) -> float:
    model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float32)
    mx.eval(model.parameters())

    n_chunks = (len(token_ids) - 1) // seq_len
    nlls = []
    for i in range(n_chunks):
        start = i * seq_len
        chunk = token_ids[start : start + seq_len + 1]
        chunk_input = mx.array(chunk[:-1][None, :])
        logits, _ = model(chunk_input)
        mx.eval(logits)

        logits_np = np.array(logits[0], dtype=np.float32)
        targets = chunk[1:].astype(np.int64)
        max_l = logits_np.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(logits_np - max_l).sum(axis=-1)) + max_l.squeeze(-1)
        target_logit = logits_np[np.arange(len(targets)), targets]
        nll = float((log_sum_exp - target_logit).mean())
        nlls.append(nll)
        print(f"  [mlx]   chunk {i+1}/{n_chunks}  nll={nll:.4f}", flush=True)

    return math.exp(float(np.mean(nlls)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=None,
                         help="Cap tokens for a faster check; omit for the full WikiText-2 test set.")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--out_dir", default="results/ppl")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            repo_id=args.model_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
        )

    token_ids = load_wikitext2_test_tokens(args.model_id)
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens + 1]

    t0 = time.time()
    ppl_mlx = _mlx_ppl(model_dir, args.model_id, token_ids, args.seq_len)
    t1 = time.time()
    ppl_torch = _torch_ppl(args.model_id, token_ids, args.seq_len, args.device)
    t2 = time.time()

    diff = abs(ppl_mlx - ppl_torch)

    result = {
        "model_id": args.model_id,
        "seq_len": args.seq_len,
        "max_tokens": args.max_tokens,
        "n_tokens_used": int(len(token_ids)),
        "ppl_mlx_fp32": ppl_mlx,
        "ppl_transformers_fp32": ppl_torch,
        "abs_diff": diff,
        "elapsed_s": {"mlx": round(t1 - t0, 1), "transformers": round(t2 - t1, 1)},
        "env": collect_env_info(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fp16_reference_validation.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nMLX fp32 PPL:          {ppl_mlx:.4f}")
    print(f"transformers fp32 PPL: {ppl_torch:.4f}")
    print(f"abs diff:              {diff:.4f}")
    print(f"[validate_fp16_reference] wrote {out_path}")


if __name__ == "__main__":
    main()
