"""Speculative decoding: distribution check, acceptance rate, and throughput.

Part 1 — is the output distribution really the target's?
  Two small random models (draft != target) and temperature 1. The first three
  generated tokens are sampled --n_samples times with speculative decoding
  and compared, by chi-square goodness of fit, with the target model's exact
  distribution (obtained by enumerating every prefix): the marginals of
  tokens 1, 2 and 3 and the joint of (token 1, token 2). Low-expected-count
  cells are pooled. A lossless sampler should give
  unremarkable p-values; the same test is also run on plain (non-speculative)
  sampling as a control for the test itself.

Part 2 — acceptance rate and speed on the real model
  target = fp16 model, draft = the same model with int4 weights (GPU backend).
  This is the only draft available without downloading a second model; it is
  not much cheaper than the target, so no speedup should be expected — the
  point is the measured acceptance rate and the cost accounting.

    python scripts/eval_speculative.py

Writes results/claims/speculative.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from siliconfer.engine.generate import SamplingParams, generate
from siliconfer.engine.speculative import speculative_generate
from siliconfer.eval.env_info import collect_env_info
from siliconfer.eval.perplexity import load_wikitext2_test_tokens
from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel

_TINY = dict(
    architectures=["LlamaForCausalLM"], hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=24,
    max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=10000.0,
    tie_word_embeddings=True, hidden_act="silu",
)


def _tiny_model(seed: int) -> LlamaModel:
    mx.random.seed(seed)
    model = LlamaModel(ModelConfig(**_TINY))
    # Larger weights than the default init: the two models then disagree enough
    # that about 40% of draft tokens are accepted, so both the accept path and
    # the reject-and-resample path are exercised heavily.
    model.update(_scaled(model.parameters(), 1.5))
    mx.eval(model.parameters())
    return model


def _scaled(tree, factor):
    if isinstance(tree, mx.array):
        return tree * factor if tree.ndim >= 2 else tree
    if isinstance(tree, dict):
        return {k: _scaled(v, factor) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_scaled(v, factor) for v in tree]
    return tree


def _probs(model: LlamaModel, ids: list[int]) -> np.ndarray:
    logits, _ = model(mx.array([ids]))
    p = mx.softmax(logits[0, -1].astype(mx.float32))
    return np.array(p, dtype=np.float64)


def _chi_square(observed: np.ndarray, expected_p: np.ndarray, min_expected: float = 5.0) -> dict:
    """Pearson chi-square with cells of expected count < min_expected pooled into one."""
    n = observed.sum()
    expected = expected_p * n
    small = expected < min_expected
    obs = np.append(observed[~small], observed[small].sum())
    exp = np.append(expected[~small], expected[small].sum())
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    stat = float(((obs - exp) ** 2 / exp).sum())
    df = len(obs) - 1
    p_value = float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64),
                                            torch.tensor(stat / 2.0, dtype=torch.float64)))
    return {"chi2": stat, "df": df, "p_value": p_value, "n": int(n)}


def distribution_check(n_samples: int, K: int) -> dict:
    target, draft = _tiny_model(1), _tiny_model(2)
    V = _TINY["vocab_size"]
    prompt = [3, 7, 1, 5]
    prompt_mx = mx.array([prompt])
    params = SamplingParams(temperature=1.0, max_tokens=3)

    # Exact target distribution of the first three tokens, by enumeration.
    p1 = _probs(target, prompt)
    p2_given = np.stack([_probs(target, prompt + [a]) for a in range(V)])            # [a, b]
    joint12 = p1[:, None] * p2_given
    p3 = np.zeros(V)
    for a in range(V):
        for b in range(V):
            p3 += joint12[a, b] * _probs(target, prompt + [a, b])
    expected = {"token1": p1, "token2": joint12.sum(axis=0), "token3": p3,
                "token1_token2_joint": joint12.reshape(-1)}
    tv_draft_target = 0.5 * float(np.abs(_probs(draft, prompt + [0]) - p2_given[0]).sum())

    out = {"vocab_size": V, "K": K, "n_samples": n_samples,
           "note": "token1 is sampled from the target right after prefill in every sampler; "
                   "tokens 2 and 3 are the ones produced by draft-then-verify rounds",
           "total_variation_draft_vs_target_example": tv_draft_target}

    def spec(seed, **kw):
        r = speculative_generate(draft, target, prompt_mx, params=params, K=K, seed=seed, **kw)
        return r.token_ids, r.total_accepted, r.total_draft_tokens

    samplers = {
        "speculative": lambda seed: spec(seed),
        "speculative_dynamic_K": lambda seed: spec(seed, dynamic_K=True),
        "plain_sampling_control": lambda seed: (_plain(target, prompt_mx, params, seed), 0, 0),
    }
    for name, sample in samplers.items():
        counts = {"token1": np.zeros(V), "token2": np.zeros(V), "token3": np.zeros(V),
                  "token1_token2_joint": np.zeros(V * V)}
        accepted = drafted = 0
        for i in range(n_samples):
            (a, b, c), acc, n_draft = sample(i)
            counts["token1"][a] += 1
            counts["token2"][b] += 1
            counts["token3"][c] += 1
            counts["token1_token2_joint"][a * V + b] += 1
            accepted += acc
            drafted += n_draft
        out[name] = {k: _chi_square(counts[k], expected[k]) for k in counts}
        out[name]["draft_acceptance_rate"] = accepted / drafted if drafted else None
        print(f"  {name:<24} p-values: " + "  ".join(
            f"{k}={out[name][k]['p_value']:.3f}" for k in counts)
            + (f"  (acceptance {accepted / drafted:.2f})" if drafted else ""))
    return out


def _plain(model, prompt_mx, params, seed):
    mx.random.seed(seed)
    return generate(model, prompt_mx, params=params).token_ids


def real_model_run(model_id: str, n_prompts: int, prompt_len: int, n_decode: int) -> dict:
    from huggingface_hub import snapshot_download

    from siliconfer.engine.q4_loader import load_q4_model

    model_dir = snapshot_download(
        repo_id=model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
    target, _ = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
    mx.eval(target.parameters())
    draft, _ = load_q4_model(model_dir, method="rtn", backend="mlx", verbose=False)

    tokens = load_wikitext2_test_tokens(model_id)
    prompts = [mx.array(tokens[i * 4096:i * 4096 + prompt_len].tolist())[None, :]
               for i in range(n_prompts)]

    out = {"target": "fp16", "draft": "rtn int4 (same model), mlx backend",
           "n_prompts": n_prompts, "prompt_len": prompt_len, "n_decode": n_decode, "settings": {}}
    for temp in (0.0, 1.0):
        params = SamplingParams(temperature=temp, max_tokens=n_decode)
        generate(target, prompts[0], params=params)                     # warm-up
        base = [generate(target, p, params=params).decode_tok_s for p in prompts]
        for label, kw in (("K=4", dict(K=4)), ("dynamic_K", dict(K=4, dynamic_K=True))):
            speculative_generate(draft, target, prompts[0], params=params, seed=0, **kw)
            acc, tps, rounds = [], [], []
            for i, p in enumerate(prompts):
                r = speculative_generate(draft, target, p, params=params, seed=i, **kw)
                acc.append(r.acceptance_rate)
                tps.append(r.effective_tok_s)
                rounds.append(r.num_decode_tokens / max(r.total_rounds, 1))
            out["settings"][f"temperature={temp},{label}"] = {
                "acceptance_rate_mean": float(np.mean(acc)),
                "acceptance_rate_std": float(np.std(acc, ddof=1)),
                "tokens_per_target_call_mean": float(np.mean(rounds)),
                "speculative_tok_s_mean": float(np.mean(tps)),
                "speculative_tok_s_std": float(np.std(tps, ddof=1)),
                "plain_target_tok_s_mean": float(np.mean(base)),
                "plain_target_tok_s_std": float(np.std(base, ddof=1)),
                "speedup": float(np.mean(tps) / np.mean(base)),
            }
            s = out["settings"][f"temperature={temp},{label}"]
            print(f"  T={temp} {label:<10} acceptance {s['acceptance_rate_mean']:.3f}  "
                  f"{s['speculative_tok_s_mean']:.1f} tok/s vs plain {s['plain_target_tok_s_mean']:.1f} "
                  f"(x{s['speedup']:.2f})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n_samples", type=int, default=20_000)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--n_decode", type=int, default=128)
    parser.add_argument("--skip_real_model", action="store_true")
    parser.add_argument("--out", default="results/claims/speculative.json")
    args = parser.parse_args()

    t0 = time.time()
    print("[eval_speculative] distribution check (tiny random models)")
    result = {"distribution_check": distribution_check(args.n_samples, args.K)}
    if not args.skip_real_model:
        print("[eval_speculative] acceptance rate and throughput (real model)")
        result["real_model"] = real_model_run(args.model_id, args.n_prompts, args.prompt_len,
                                              args.n_decode)
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["env"] = collect_env_info()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[eval_speculative] wrote {out_path}")


if __name__ == "__main__":
    main()
