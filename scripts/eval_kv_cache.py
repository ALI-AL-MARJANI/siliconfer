"""int8 KV-cache quality and memory, measured through real incremental decode.

Perplexity is computed the way generation actually uses the cache: tokens are
fed one at a time and each new key/value vector is quantized as it is
appended. (Scoring a whole window in one forward pass would quantize the
cache once, after the fact, and is not representative.)

For each of --n_segments non-overlapping WikiText-2 test segments of
--segment_len tokens, the same fp16 model is run twice — plain cache and
int8 cache — and the next-token NLL is accumulated. Reports overall PPL for
both, the per-segment PPL difference (mean, std, paired bootstrap 95% CI),
and the cache's analytic memory footprint.

    python scripts/eval_kv_cache.py

Writes results/claims/kv_cache.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.eval.bench import measure_kv_cache_memory
from siliconfer.eval.env_info import collect_env_info
from siliconfer.eval.perplexity import load_wikitext2_test_tokens
from siliconfer.model.kv_cache import make_quantized_cache
from siliconfer.model.llama import LlamaModel


def incremental_nll(model: LlamaModel, tokens: np.ndarray, quantized: bool) -> float:
    """Mean next-token NLL over `tokens`, decoding one token at a time."""
    cache = make_quantized_cache(len(model.layers)) if quantized else None
    total = 0.0
    for t in range(len(tokens) - 1):
        logits, cache = model(mx.array([[int(tokens[t])]]), cache)
        row = logits[0, -1].astype(mx.float32)
        nll = mx.logsumexp(row) - row[int(tokens[t + 1])]
        mx.eval(nll)
        total += nll.item()
    return total / (len(tokens) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n_segments", type=int, default=8)
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", default="results/claims/kv_cache.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    )
    model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
    mx.eval(model.parameters())

    tokens = load_wikitext2_test_tokens(args.model_id)
    t0 = time.time()
    nll_plain, nll_int8 = [], []
    for s in range(args.n_segments):
        seg = tokens[s * args.segment_len:(s + 1) * args.segment_len]
        nll_plain.append(incremental_nll(model, seg, quantized=False))
        nll_int8.append(incremental_nll(model, seg, quantized=True))
        print(f"  segment {s + 1}/{args.n_segments}: PPL plain={math.exp(nll_plain[-1]):.2f} "
              f"int8={math.exp(nll_int8[-1]):.2f}")

    plain, int8 = np.array(nll_plain), np.array(nll_int8)
    seg_ppl_diff = np.exp(int8) - np.exp(plain)

    # Paired bootstrap over segments on the overall-PPL difference.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, args.n_segments, size=(args.bootstrap, args.n_segments))
    boot = np.exp(int8[idx].mean(axis=1)) - np.exp(plain[idx].mean(axis=1))

    result = {
        "model_id": args.model_id,
        "protocol": "incremental decode, one token per forward pass, fp16 weights",
        "n_segments": args.n_segments,
        "segment_len": args.segment_len,
        "n_tokens_scored": args.n_segments * (args.segment_len - 1),
        "ppl_plain_cache": math.exp(plain.mean()),
        "ppl_int8_cache": math.exp(int8.mean()),
        "delta_ppl": math.exp(int8.mean()) - math.exp(plain.mean()),
        "delta_ppl_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                     float(np.percentile(boot, 97.5))],
        "per_segment_ppl_plain": np.exp(plain).tolist(),
        "per_segment_ppl_int8": np.exp(int8).tolist(),
        "per_segment_delta_mean": float(seg_ppl_diff.mean()),
        "per_segment_delta_std": float(seg_ppl_diff.std(ddof=1)),
        "cache_memory_at_2048_tokens": measure_kv_cache_memory(config, 2048),
        "elapsed_s": round(time.time() - t0, 1),
        "env": collect_env_info(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[eval_kv_cache] PPL plain={result['ppl_plain_cache']:.2f} "
          f"int8={result['ppl_int8_cache']:.2f}  delta={result['delta_ppl']:+.2f} "
          f"(95% CI {result['delta_ppl_bootstrap_ci95'][0]:+.2f}..{result['delta_ppl_bootstrap_ci95'][1]:+.2f})")
    print(f"[eval_kv_cache] wrote {out_path}")


if __name__ == "__main__":
    main()
