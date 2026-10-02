"""Time to first token, cold vs warmed, per serving configuration.

A "cold" measurement must come from a process that has never run the model,
so every trial is its own subprocess: load the model, optionally call
`warmup()`, then time one prefill + first token.

    python scripts/bench_ttft.py

Writes results/bench/ttft.json. Run on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_CHILD = r"""
import json, sys, time
import mlx.core as mx
from huggingface_hub import snapshot_download
from siliconfer.engine.generate import SamplingParams, generate, warmup

model_id, config, do_warmup, prompt_len = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4])
model_dir = snapshot_download(repo_id=model_id,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
if config == "fp16":
    from siliconfer.model.llama import LlamaModel
    model, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
    mx.eval(model.parameters())
else:
    from siliconfer.engine.q4_loader import load_q4_model
    model, _ = load_q4_model(model_dir, method="rtn", backend=config.split("-")[1], verbose=False)

t0 = time.perf_counter()
if do_warmup:
    warmup(model)
warmup_ms = (time.perf_counter() - t0) * 1e3

prompt = mx.array([[(7 * i) % 1000 + 1000 for i in range(prompt_len)]])
t0 = time.perf_counter()
generate(model, prompt, params=SamplingParams(temperature=0.0, max_tokens=1))
print(json.dumps({"ttft_ms": (time.perf_counter() - t0) * 1e3, "warmup_ms": warmup_ms}))
"""


def _dist(values: list[float]) -> dict:
    a = np.asarray(values)
    return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "mean": float(a.mean()), "std": float(a.std(ddof=1)), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--configs", default="fp16,int4-neon,int4-mlx")
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--out", default="results/bench/ttft.json")
    args = parser.parse_args()

    results = {}
    for config in args.configs.split(","):
        entry = {}
        for label, flag in (("cold", "0"), ("warmed", "1")):
            ttft, warm = [], []
            for _ in range(args.n_trials):
                out = subprocess.run(
                    [sys.executable, "-c", _CHILD, args.model_id, config, flag, str(args.prompt_len)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip().splitlines()[-1]
                rec = json.loads(out)
                ttft.append(rec["ttft_ms"])
                warm.append(rec["warmup_ms"])
            entry[label] = {"ttft_ms": _dist(ttft)}
            if flag == "1":
                entry[label]["warmup_ms"] = _dist(warm)
        entry["speedup_p50"] = entry["cold"]["ttft_ms"]["p50"] / entry["warmed"]["ttft_ms"]["p50"]
        results[config] = entry
        print(f"[bench_ttft] {config:<10} cold p50 {entry['cold']['ttft_ms']['p50']:.1f} ms  "
              f"warmed p50 {entry['warmed']['ttft_ms']['p50']:.1f} ms  (x{entry['speedup_p50']:.2f}; "
              f"warm-up itself {entry['warmed']['warmup_ms']['p50']:.1f} ms)")

    from siliconfer.eval.env_info import collect_env_info
    out = {"model_id": args.model_id, "prompt_len": args.prompt_len, "n_trials": args.n_trials,
           "note": "each trial is a fresh process; TTFT = prefill + first sampled token",
           "configs": results, "env": collect_env_info()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench_ttft] wrote {out_path}")


if __name__ == "__main__":
    main()
