"""End-to-end generation benchmark: fp16 vs int4 on both kernels.

Configurations (same model, same prompts, greedy decoding):

  fp16        MLX float16, GPU
  int4-neon   packed int4 through this repo's NEON kernel (CPU). Every linear
              layer leaves the MLX graph, so each one forces an evaluation and
              a GPU -> CPU -> GPU copy of the activation.
  int4-mlx    the same int4 codes through MLX's native quantized matmul (GPU);
              the whole decode step stays in one lazy graph.

Weights are quantized with RTN: speed does not depend on which method chose
the codes, only on the storage format.

For each prompt length: one warm-up run, then --n_runs timed runs of
--n_decode generated tokens. Reports p50 / p90 / mean / std of decode tok/s,
prefill tok/s and time to first token, plus weight memory and MLX peak memory.

Writes results/bench/decode.json. Run on an otherwise idle machine:

    python scripts/bench_decode.py
    python scripts/bench_decode.py --model_id Qwen/Qwen2.5-1.5B   # second model size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.engine.generate import SamplingParams, generate
from siliconfer.eval.bench import measure_memory
from siliconfer.eval.env_info import collect_env_info
from siliconfer.eval.perplexity import load_wikitext2_test_tokens


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


def _load(config_name: str, model_dir: str):
    if config_name == "fp16":
        from siliconfer.model.llama import LlamaModel
        model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
        mx.eval(model.parameters())
        return model, config
    from siliconfer.engine.q4_loader import load_q4_model
    backend = config_name.split("-")[1]
    return load_q4_model(model_dir, method="rtn", backend=backend, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--configs", default="fp16,int4-neon,int4-mlx")
    parser.add_argument("--prompt_lens", default="128,512,2048")
    parser.add_argument("--n_decode", type=int, default=256)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    )

    tokens = load_wikitext2_test_tokens(args.model_id)
    prompt_lens = [int(v) for v in args.prompt_lens.split(",")]
    prompts = {n: mx.array(tokens[:n].tolist())[None, :] for n in prompt_lens}
    params = SamplingParams(temperature=0.0, max_tokens=args.n_decode)

    results = {}
    reference_tokens: dict[int, list[int]] = {}
    for name in args.configs.split(","):
        mx.clear_cache()
        mx.reset_peak_memory()
        model, _ = _load(name, model_dir)
        entry = {"weights_mb": measure_memory(model), "prompts": {}}

        for n, prompt in prompts.items():
            generate(model, prompt, params=params)          # warm-up, not timed
            decode, prefill, ttft = [], [], []
            for _ in range(args.n_runs):
                r = generate(model, prompt, params=params)
                decode.append(r.decode_tok_s)
                prefill.append(r.prefill_tok_s)
                ttft.append(r.prefill_time * 1e3)
            entry["prompts"][str(n)] = {
                "decode_tok_s": _dist(decode),
                "prefill_tok_s": _dist(prefill),
                "ttft_ms": _dist(ttft),
                "n_generated": r.num_decode_tokens,
            }
            # The two int4 kernels run the same codes: their greedy output must agree.
            if name.startswith("int4"):
                ref = reference_tokens.setdefault(n, r.token_ids)
                entry["prompts"][str(n)]["tokens_match_first_int4_config"] = r.token_ids == ref
            d = entry["prompts"][str(n)]["decode_tok_s"]
            print(f"[bench_decode] {name:<10} prompt={n:<5} decode p50={d['p50']:.1f} "
                  f"p90={d['p90']:.1f} tok/s  "
                  f"prefill p50={entry['prompts'][str(n)]['prefill_tok_s']['p50']:.0f} tok/s")

        entry["mlx_peak_memory_mb"] = mx.get_peak_memory() / 1e6
        results[name] = entry
        del model

    out = {
        "model_id": args.model_id,
        "n_decode": args.n_decode,
        "n_runs": args.n_runs,
        "sampling": "greedy",
        "quantization": "rtn, group_size=128, symmetric",
        "configs": results,
        "env": collect_env_info(),
    }
    default_name = f"decode_{args.model_id.split('/')[-1]}.json"
    out_path = Path(args.out or f"results/bench/{default_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench_decode] wrote {out_path}")


if __name__ == "__main__":
    main()
