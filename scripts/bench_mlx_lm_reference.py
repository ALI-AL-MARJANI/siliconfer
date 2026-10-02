"""External reference for scripts/bench_decode.py: the same model through mlx-lm.

mlx-lm is not a dependency of this repo. Run this script in a throwaway
environment so it cannot change the library versions the other results were
produced with:

    uvx --from mlx-lm python scripts/bench_mlx_lm_reference.py

Configurations: the model as stored (bf16), and MLX's own 4-bit quantization
(`mlx.nn.quantize`, group size 64, which also quantizes the embedding).
Same prompts (WikiText-2 test prefix), same token counts and greedy decoding
as bench_decode.py. Writes results/bench/decode_mlx_lm_reference_<model>.json.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def _dist(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p10": float(np.percentile(a, 10)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt_lens", default="128,512,2048")
    parser.add_argument("--n_decode", type=int, default=256)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = {}
    for name in ("mlx_lm-bf16", "mlx_lm-4bit"):
        mx.clear_cache()
        mx.reset_peak_memory()
        model, tokenizer = load(args.model_id)
        if name.endswith("4bit"):
            nn.quantize(model, group_size=64, bits=4)
            mx.eval(model.parameters())

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in dataset["text"] if t.strip())
        tokens = tokenizer.encode(text[:200_000])
        sampler = make_sampler(temp=0.0)

        entry = {"prompts": {}}
        for n in (int(v) for v in args.prompt_lens.split(",")):
            prompt = tokens[:n]
            decode, prefill = [], []
            for i in range(args.n_runs + 1):                 # first run is the warm-up
                last = None
                for last in stream_generate(model, tokenizer, prompt, max_tokens=args.n_decode,
                                            sampler=sampler):
                    pass
                if i > 0:
                    decode.append(last.generation_tps)
                    prefill.append(last.prompt_tps)
            entry["prompts"][str(n)] = {
                "decode_tok_s": _dist(decode),
                "prefill_tok_s": _dist(prefill),
                "n_generated": last.generation_tokens,
            }
            print(f"[mlx_lm_reference] {name:<12} prompt={n:<5} "
                  f"decode p50={np.percentile(decode, 50):.1f} tok/s")
        entry["mlx_peak_memory_mb"] = mx.get_peak_memory() / 1e6
        results[name] = entry
        del model

    out = {
        "model_id": args.model_id,
        "n_decode": args.n_decode,
        "n_runs": args.n_runs,
        "sampling": "greedy",
        "configs": results,
        "env": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm")},
        },
    }
    name = f"decode_mlx_lm_reference_{args.model_id.split('/')[-1]}.json"
    out_path = Path(args.out or f"results/bench/{name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[mlx_lm_reference] wrote {out_path}")


if __name__ == "__main__":
    main()
