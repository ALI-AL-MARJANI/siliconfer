"""Build results/SUMMARY.md from the raw result files.

Every table in the README is copied from the output of this script, so a
number in the README can always be traced to a file under results/. Missing
inputs produce a "run pending" line rather than an error.

    python scripts/make_results_tables.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PENDING = "_run pending_"

# (method, awq_block_loss, awq_clip) -> label, in display order
_METHOD_ORDER = [
    (("fp16", False, False), "fp16 (unquantized)"),
    (("rtn", False, False), "RTN"),
    (("sinq", False, False), "SINQ-style column rescaling"),
    (("awq", False, False), "AWQ (layer-output loss, no clip)"),
    (("awq", True, False), "AWQ + block loss"),
    (("awq", False, True), "AWQ + clip"),
    (("awq", True, True), "AWQ + block loss + clip"),
    (("hqq", False, False), "HQQ-style clip search"),
    (("gptq", False, False), "GPTQ"),
    (("mlx_native", False, False), "MLX `mx.quantize` (external reference)"),
]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _pm(values: list[float], digits: int = 2) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def _ppl_rows(ppl_dir: Path) -> dict:
    """{(method, block, clip, grid, group_size): {dataset: [ppl per seed]}}"""
    rows: dict = defaultdict(lambda: defaultdict(list))
    meta: dict = {}
    for path in sorted(ppl_dir.glob("*.json")):
        r = _load(path)
        if not r or "method" not in r:
            continue
        m = r["method"]
        if m == "fp16":
            grid = "—"
        elif m in ("hqq", "mlx_native"):
            grid = "asym"                       # these quantizers are always asymmetric
        else:
            grid = "sym" if r.get("sym", True) else "asym"
        key = (m, bool(r.get("awq_block_loss")), bool(r.get("awq_clip")), grid,
               r.get("group_size", 128))
        rows[key][r["dataset"]].append(r["ppl"])
        meta[r["dataset"]] = r.get("n_tokens_scored") or r.get("max_tokens") or meta.get(r["dataset"])
    return {"rows": rows, "tokens": meta}


def ppl_table(ppl_dir: Path, title: str) -> list[str]:
    data = _ppl_rows(ppl_dir)
    rows = data["rows"]
    out = [f"### {title}", "", f"Source: `{ppl_dir}/*.json`.", ""]
    if not rows:
        return out + [PENDING, ""]

    fp16 = {d: v[0] for d, v in rows.get(("fp16", False, False, "—", 128), {}).items()}
    out += ["| Method | Grid | Bits/weight | WikiText-2 PPL | Δ vs fp16 | C4 PPL | Δ vs fp16 | Seeds |",
            "|---|---|---|---|---|---|---|---|"]
    for (m, block, clip), label in _METHOD_ORDER:
        for grid in ("—", "sym", "asym"):
            for group_size in (128, 64):
                key = (m, block, clip, grid, group_size)
                if key not in rows:
                    continue
                cells = []
                n_seeds = 1
                for d in ("wikitext2", "c4"):
                    v = rows[key].get(d)
                    if not v:
                        cells += ["—", "—"]
                        continue
                    n_seeds = max(n_seeds, len(v))
                    delta = f"+{np.mean(v) - fp16[d]:.2f}" if m != "fp16" and d in fp16 else "—"
                    cells += [_pm(v), delta]
                if m == "fp16":
                    bits = "16"
                else:
                    bits = f"{4 + (32 if grid == 'sym' else 64) / group_size:.2f}"
                name = label + (f", group {group_size}" if group_size != 128 else "")
                out.append(f"| {name} | {grid} | {bits} | " + " | ".join(cells) + f" | {n_seeds} |")
    out += ["",
            "± is the sample standard deviation over calibration seeds; rows without ± have no "
            "random component. Bits/weight counts the 4-bit code plus the per-group float32 scale "
            "(and zero-point for asymmetric grids).", ""]
    return out


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def decode_table(bench_dir: Path) -> list[str]:
    out = ["### End-to-end generation", ""]
    files = sorted(p for p in bench_dir.glob("decode_*.json") if "mlx_lm_reference" not in p.name)
    if not files:
        return out + [PENDING, ""]
    for path in files:
        r = _load(path)
        ref = _load(bench_dir / path.name.replace("decode_", "decode_mlx_lm_reference_"))
        out += [f"`{path}` — {r['model_id']}, greedy, {r['n_decode']} generated tokens, "
                f"{r['n_runs']} runs after one warm-up.", "",
                "| Configuration | Prompt | Decode tok/s p50 (p10–p90) | Prefill tok/s p50 | "
                "TTFT ms p50 | Weights MB | MLX peak MB |", "|---|---|---|---|---|---|---|"]
        for name, e in r["configs"].items():
            for n, p in e["prompts"].items():
                d = p["decode_tok_s"]
                out.append(f"| {name} | {n} | {d['p50']:.1f} ({d['p10']:.1f}–{d['p90']:.1f}) | "
                           f"{p['prefill_tok_s']['p50']:.0f} | {p['ttft_ms']['p50']:.1f} | "
                           f"{e['weights_mb']['total_mb']:.0f} | {e['mlx_peak_memory_mb']:.0f} |")
        if ref:
            out += ["", f"External reference (`{bench_dir / path.name.replace('decode_', 'decode_mlx_lm_reference_')}`, "
                        f"mlx-lm {ref['env']['versions']['mlx-lm']}, mlx {ref['env']['versions']['mlx']}):", "",
                    "| Configuration | Prompt | Decode tok/s p50 (p10–p90) | Prefill tok/s p50 | MLX peak MB |",
                    "|---|---|---|---|---|"]
            for name, e in ref["configs"].items():
                for n, p in e["prompts"].items():
                    d = p["decode_tok_s"]
                    out.append(f"| {name} | {n} | {d['p50']:.1f} ({d['p10']:.1f}–{d['p90']:.1f}) | "
                               f"{p['prefill_tok_s']['p50']:.0f} | {e['mlx_peak_memory_mb']:.0f} |")
        out.append("")
    return out


def gemv_table(bench_dir: Path) -> list[str]:
    out = ["### GEMV micro-benchmark (one matrix-vector product)", ""]
    r = _load(bench_dir / "gemv.json")
    if not r:
        return out + [PENDING, ""]
    bw = {k: v for k, v in r["bandwidth"].items() if k != "method"}
    out += [f"`{bench_dir / 'gemv.json'}`. Measured copy bandwidth: "
            + ", ".join(f"{k} = {v:.0f} GB/s" for k, v in bw.items())
            + f" ({r['bandwidth']['method']}).", "",
            "Cold-cache p50 in ms (each call touches a matrix that is not in cache); "
            "in parentheses, GB/s of that contender's own weight bytes.", ""]
    names = list(r["shapes"][0]["contenders"])
    out += ["| out × in | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in r["shapes"]:
        cells = [f"{s['contenders'][n]['cold']['p50_ms']:.3f} ({s['contenders'][n]['cold']['gb_per_s']:.0f})"
                 for n in names]
        out.append(f"| {s['out_f']} × {s['in_f']} | " + " | ".join(cells) + " |")
    out += ["", "Cache-hot p50 in ms (the same matrix called repeatedly):", "",
            "| out × in | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in r["shapes"]:
        out.append(f"| {s['out_f']} × {s['in_f']} | "
                   + " | ".join(f"{s['contenders'][n]['hot']['p50_ms']:.3f}" for n in names) + " |")
    out.append("")
    return out


def metal_table(bench_dir: Path) -> list[str]:
    out = ["### Decode-step attention over an int8 KV cache", ""]
    r = _load(bench_dir / "metal_attention.json")
    if not r:
        return out + [PENDING, ""]
    out += [f"`{bench_dir / 'metal_attention.json'}`, p50 ms over {r['n_reps']} calls.", "",
            "| Cached tokens | fused Metal kernel | dequantize + MLX SDPA | MLX SDPA on fp16 cache | "
            "fused vs dequant+SDPA | fused vs fp16 SDPA | max abs err |", "|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        out.append(f"| {row['T']} | {row['fused_metal']['p50_ms']:.3f} | "
                   f"{row['dequant_then_sdpa']['p50_ms']:.3f} | {row['sdpa_fp16_cache']['p50_ms']:.3f} | "
                   f"{row['fused_speedup_vs_dequant_sdpa']:.2f}× | {row['fused_speedup_vs_sdpa_fp16']:.2f}× | "
                   f"{row['fused_max_abs_err_vs_dequant_sdpa']:.1e} |")
    out.append("")
    return out


def ttft_table(bench_dir: Path) -> list[str]:
    out = ["### Time to first token, cold vs warmed", ""]
    r = _load(bench_dir / "ttft.json")
    if not r:
        return out + [PENDING, ""]
    out += [f"`{bench_dir / 'ttft.json'}`, {r['n_trials']} fresh processes per cell, "
            f"{r['prompt_len']}-token prompt.", "",
            "| Configuration | Cold TTFT ms p50 (p90) | Warmed TTFT ms p50 (p90) | Speedup | Warm-up cost ms |",
            "|---|---|---|---|---|"]
    for name, e in r["configs"].items():
        c, w = e["cold"]["ttft_ms"], e["warmed"]["ttft_ms"]
        out.append(f"| {name} | {c['p50']:.1f} ({c['p90']:.1f}) | {w['p50']:.1f} ({w['p90']:.1f}) | "
                   f"{e['speedup_p50']:.2f}× | {e['warmed']['warmup_ms']['p50']:.1f} |")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Per-claim evaluations
# ---------------------------------------------------------------------------

def claims(claims_dir: Path) -> list[str]:
    out = []

    out += ["### int8 KV cache", ""]
    r = _load(claims_dir / "kv_cache.json")
    if r:
        lo, hi = r["delta_ppl_bootstrap_ci95"]
        mem = r["cache_memory_at_2048_tokens"]
        out += [f"`{claims_dir / 'kv_cache.json'}` — {r['protocol']}; {r['n_segments']} segments × "
                f"{r['segment_len']} tokens ({r['n_tokens_scored']} scored).", "",
                f"- PPL with plain cache: {r['ppl_plain_cache']:.2f}; with int8 cache: "
                f"{r['ppl_int8_cache']:.2f}; difference {r['delta_ppl']:+.2f} "
                f"(paired bootstrap 95% CI {lo:+.2f} to {hi:+.2f}).",
                f"- Cache size at 2,048 tokens: {mem['fp16_mb']:.1f} MB fp16 → {mem['int8_mb']:.1f} MB int8 "
                f"({mem['compression']:.2f}×, analytic).", ""]
    else:
        out += [PENDING, ""]

    out += ["### Speculative decoding", ""]
    r = _load(claims_dir / "speculative.json")
    if r:
        dc = r["distribution_check"]
        out += [f"`{claims_dir / 'speculative.json'}`. Distribution check: {dc['n_samples']} samples, "
                f"vocabulary {dc['vocab_size']}, chi-square p-values against the exact target distribution.", "",
                "| Sampler | token 1 | token 2 | token 3 | (token 1, token 2) joint | draft acceptance |",
                "|---|---|---|---|---|---|"]
        for name in ("speculative", "speculative_dynamic_K", "plain_sampling_control"):
            e = dc[name]
            acc = f"{e['draft_acceptance_rate']:.2f}" if e.get("draft_acceptance_rate") is not None else "—"
            out.append(f"| {name} | {e['token1']['p_value']:.3f} | {e['token2']['p_value']:.3f} | "
                       f"{e['token3']['p_value']:.3f} | {e['token1_token2_joint']['p_value']:.3f} | {acc} |")
        if "real_model" in r:
            rm = r["real_model"]
            out += ["", f"Real model: target {rm['target']}, draft {rm['draft']}; {rm['n_prompts']} prompts × "
                        f"{rm['n_decode']} tokens.", "",
                    "| Setting | Acceptance rate | Tokens per target call | Speculative tok/s | Plain target tok/s | Ratio |",
                    "|---|---|---|---|---|---|"]
            for name, s in rm["settings"].items():
                out.append(f"| {name} | {s['acceptance_rate_mean']:.3f} ± {s['acceptance_rate_std']:.3f} | "
                           f"{s['tokens_per_target_call_mean']:.2f} | {s['speculative_tok_s_mean']:.1f} ± "
                           f"{s['speculative_tok_s_std']:.1f} | {s['plain_target_tok_s_mean']:.1f} ± "
                           f"{s['plain_target_tok_s_std']:.1f} | {s['speedup']:.2f}× |")
        out.append("")
    else:
        out += [PENDING, ""]

    out += ["### Mixed precision (fake-quant only)", ""]
    found = False
    for path in sorted(claims_dir.glob("mixed_precision_*.json")):
        r = _load(path)
        if not r:
            continue
        found = True
        out += [f"`{path}`:", ""] + [f"- {name}: PPL {ppl:.2f}" for name, ppl in r["ppl"].items()] + [""]
    if not found:
        out += [PENDING, ""]

    out += ["### Draft head", ""]
    r = _load(claims_dir / "draft_head.json")
    if r:
        out += [f"`{claims_dir / 'draft_head.json'}` — {r['config']['n_train']} training sequences of "
                f"{r['config']['seq_len']} tokens, {len(r['runs'])} seeds, top-1 next-token accuracy on "
                f"{r['n_test_tokens']} held-out WikiText-2 test tokens.", "",
                f"- Draft head: {100 * r['draft_top1_test_mean']:.2f}% ± {100 * r['draft_top1_test_std']:.2f}% "
                f"(per seed: " + ", ".join(f"{100 * x['draft_top1_test']:.2f}%" for x in r["runs"]) + ").",
                f"- Target model on the same tokens: {100 * r['target_top1_test']:.2f}%.",
                "- Untrained head: " + ", ".join(f"{100 * x['draft_top1_test_untrained']:.2f}%" for x in r["runs"]) + ".",
                ""]
    else:
        out += [PENDING, ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    root = Path(args.results_dir)

    lines = ["# Results summary", "",
             "Generated by `scripts/make_results_tables.py` from the files under `results/`. "
             "Do not edit by hand.", ""]

    v = _load(root / "ppl" / "fp16_reference_validation.json")
    lines += ["## Perplexity", ""]
    if v:
        lines += [f"fp16 reference check (`{root / 'ppl' / 'fp16_reference_validation.json'}`, full WikiText-2 "
                  f"test, {v['n_tokens_used']} tokens, float32): this repo {v['ppl_mlx_fp32']:.4f}, "
                  f"`transformers` {v['ppl_transformers_fp32']:.4f}, absolute difference {v['abs_diff']:.1e}.", ""]
    lines += ppl_table(root / "ppl_full", "Full tier — whole WikiText-2 test split, 64 C4 windows")
    lines += ppl_table(root / "ppl", "Quick tier — 32,768 tokens per dataset")

    lines += ["## Speed", ""]
    lines += decode_table(root / "bench") + gemv_table(root / "bench")
    lines += metal_table(root / "bench") + ttft_table(root / "bench")

    lines += ["## Other measurements", ""] + claims(root / "claims")

    out_path = root / "SUMMARY.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
