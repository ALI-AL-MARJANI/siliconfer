"""Generate the README figures from the raw result files.

Reads results/ppl_full (falling back to results/ppl), results/bench/decode_*.json
and results/bench/gemv.json; writes PNGs to results/figures/. Nothing is
hard-coded: a figure whose input file is missing is skipped.

Usage:
    python eval/plots.py
    python eval/plots.py --results_dir results --out_dir results/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Light chart surface, ink colours, and the first two categorical slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"     # blue
SERIES_2 = "#eb6834"     # orange
NEUTRAL = "#a3a29b"      # baselines / external references

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": GRID,
    "xtick.color": INK_SECONDARY, "ytick.color": INK, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "ytick.left": False,
})

_PPL_LABELS = [
    (("rtn", False, False), "RTN"),
    (("sinq", False, False), "SINQ-style"),
    (("awq", False, False), "AWQ"),
    (("awq", True, True), "AWQ + block loss + clip"),
    (("hqq", False, False), "HQQ-style clip search"),
    (("gptq", False, False), "GPTQ"),
    (("mlx_native", False, False), "MLX mx.quantize (reference)"),
]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 1. Perplexity increase over fp16, per method, symmetric vs asymmetric grid
# ---------------------------------------------------------------------------

def plot_ppl(ppl_dir: Path, out_dir: Path) -> None:
    runs: dict = defaultdict(lambda: defaultdict(list))
    for path in sorted(ppl_dir.glob("*.json")):
        r = _load(path)
        if not r or "method" not in r or r.get("group_size", 128) != 128:
            continue
        m = r["method"]
        grid = "asym" if m in ("hqq", "mlx_native") or not r.get("sym", True) else "sym"
        runs[(m, bool(r.get("awq_block_loss")), bool(r.get("awq_clip")), grid)][r["dataset"]].append(r["ppl"])
    fp16 = {d: v[0] for d, v in runs.get(("fp16", False, False, "sym"), {}).items()}
    if not fp16:
        print("  ppl: no fp16 result, skipped")
        return

    datasets = [d for d in ("wikitext2", "c4") if d in fp16]
    titles = {"wikitext2": "WikiText-2 test", "c4": "C4 validation subset"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    h = 0.36
    for ax, d in zip(axes, datasets):
        for i, (key, _) in enumerate(_PPL_LABELS):
            for grid, offset, color in (("sym", -h / 2 - 0.01, SERIES_1), ("asym", h / 2 + 0.01, SERIES_2)):
                v = runs.get((*key, grid), {}).get(d)
                if not v:
                    continue
                delta = np.mean(v) - fp16[d]
                err = np.std(v, ddof=1) if len(v) > 1 else None
                ax.barh(i + offset, delta, height=h, color=color, xerr=err,
                        error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
                ax.text(delta + (err or 0) + 0.05, i + offset, f"+{delta:.2f}", va="center",
                        fontsize=8.5, color=INK)
        ax.set_title(f"{titles[d]}  (fp16 = {fp16[d]:.2f})")
        ax.set_xlabel("perplexity increase over fp16 (lower is better)")
        ax.set_yticks(range(len(_PPL_LABELS)), [label for _, label in _PPL_LABELS])
        ax.set_xlim(left=0)
        ax.margins(x=0.16)
    axes[0].invert_yaxis()               # y is shared: invert once
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_1), plt.Rectangle((0, 0), 1, 1, color=SERIES_2)]
    fig.legend(handles, ["symmetric grid", "asymmetric grid"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("int4 perplexity cost by method — Qwen2.5-0.5B, group size 128, context 2048",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "ppl.png")


# ---------------------------------------------------------------------------
# 2. Decode throughput, this repo's configurations and the mlx-lm reference
# ---------------------------------------------------------------------------

def plot_decode(bench_dir: Path, out_dir: Path) -> None:
    files = sorted(p for p in bench_dir.glob("decode_*.json") if "mlx_lm_reference" not in p.name)
    if not files:
        print("  decode: no result, skipped")
        return
    r = _load(files[0])
    ref = _load(bench_dir / files[0].name.replace("decode_", "decode_mlx_lm_reference_"))
    prompt = sorted(next(iter(r["configs"].values()))["prompts"], key=int)[0]

    labels = {"fp16": "fp16 (MLX, GPU)", "int4-neon": "int4, NEON kernel (CPU)",
              "int4-mlx": "int4, MLX quantized matmul (GPU)",
              "mlx_lm-bf16": "mlx-lm, bf16", "mlx_lm-4bit": "mlx-lm, 4-bit"}
    rows = [(labels.get(n, n), e["prompts"][prompt]["decode_tok_s"], SERIES_1)
            for n, e in r["configs"].items()]
    if ref:
        rows += [(labels.get(n, n), e["prompts"][prompt]["decode_tok_s"], NEUTRAL)
                 for n, e in ref["configs"].items()]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.6))
    for i, (_, d, color) in enumerate(rows):
        ax.barh(i, d["p50"], height=0.5, color=color,
                xerr=[[d["p50"] - d["p10"]], [d["p90"] - d["p50"]]],
                error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
        ax.text(d["p90"] + 2, i, f"{d['p50']:.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.margins(x=0.1)
    ax.set_xlabel("decode tokens per second, p50 with p10–p90 (higher is better)")
    ax.set_title(f"Decode speed — {r['model_id'].split('/')[-1]}, {prompt}-token prompt, greedy")
    if ref:
        handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_1), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
        fig.legend(handles, ["this repo", "external reference (mlx-lm)"], frameon=False,
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    _save(fig, out_dir / "decode.png")


# ---------------------------------------------------------------------------
# 3. GEMV micro-benchmark, one panel per matrix shape
# ---------------------------------------------------------------------------

def plot_gemv(bench_dir: Path, out_dir: Path) -> None:
    r = _load(bench_dir / "gemv.json")
    if not r:
        print("  gemv: no result, skipped")
        return
    labels = {"neon_q4_1t": "NEON int4, 1 thread", "neon_q4_mt": "NEON int4, threaded",
              "accelerate_fp32": "Accelerate fp32", "numpy_fp16": "NumPy fp16 (old baseline)",
              "mlx_fp16": "MLX fp16 (GPU)", "mlx_fp32": "MLX fp32 (GPU)", "mlx_q4": "MLX 4-bit (GPU)"}
    shapes = r["shapes"]
    cols = 3
    rows_n = -(-len(shapes) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.6 * cols, 2.9 * rows_n), sharey=True)
    axes = np.atleast_2d(axes)
    names = list(shapes[0]["contenders"])
    for ax, s in zip(axes.ravel(), shapes):
        for i, n in enumerate(names):
            ms = s["contenders"][n]["cold"]["p50_ms"]
            ax.barh(i, ms, height=0.55, color=SERIES_1 if n.startswith("neon") else NEUTRAL)
            ax.text(ms, i, f" {ms:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(f"{s['out_f']} × {s['in_f']}")
        ax.set_yticks(range(len(names)), [labels.get(n, n) for n in names])
        ax.set_xlim(left=0)
        ax.margins(x=0.22)
    axes[0, 0].invert_yaxis()            # y is shared: invert once
    for ax in axes.ravel()[len(shapes):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("ms per product, cold cache, p50 (lower is better)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_1), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
    fig.legend(handles, ["this repo's kernel", "baselines"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("One matrix-vector product (out × in), Apple M4", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "gemv.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ppl_dir = root / "ppl_full"
    if not any(ppl_dir.glob("fp16_*.json")):
        ppl_dir = root / "ppl"
    print(f"Figures from {root}/ (perplexity: {ppl_dir})")
    plot_ppl(ppl_dir, out_dir)
    plot_decode(root / "bench", out_dir)
    plot_gemv(root / "bench", out_dir)


if __name__ == "__main__":
    main()
