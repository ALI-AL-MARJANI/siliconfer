"""Generate Phase 7 benchmark plots from a results JSON file.

Usage:
    python eval/plots.py results.json            # saves PNGs next to the JSON
    python eval/plots.py results.json --show     # also opens interactive windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless default; --show switches to interactive
import matplotlib.pyplot as plt
import numpy as np

# Consistent colours across all figures
COLORS = {
    "fp16":  "#4C72B0",
    "rtn":   "#DD8452",
    "gptq":  "#55A868",
    "awq":   "#C44E52",
}
METHOD_LABELS = {
    "fp16":  "fp16 (MLX)",
    "rtn":   "RTN-int4",
    "gptq":  "GPTQ-int4",
    "awq":   "AWQ-int4",
}


def _savefig(fig, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. PPL bar chart
# ---------------------------------------------------------------------------

def plot_ppl(results: dict, out_dir: Path, show: bool) -> None:
    ppl_data = {k: v["ppl"] for k, v in results.items() if "ppl" in v}
    if not ppl_data:
        print("  [skip] no PPL data in results")
        return

    methods = list(ppl_data.keys())
    ppls    = [ppl_data[m] for m in methods]
    ref_ppl = ppl_data.get("fp16", ppls[0])
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    colors  = [COLORS.get(m, "#999") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, ppls, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    # Annotate bars with PPL value + Δ
    for bar, m, ppl in zip(bars, methods, ppls):
        delta = ppl - ref_ppl
        sign  = "+" if delta >= 0 else ""
        tag   = f"{ppl:.2f}" if m == "fp16" else f"{ppl:.2f}\n({sign}{delta:.2f})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                tag, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("WikiText-2 Perplexity (↓ lower is better)")
    ax.set_title("Quantization PPL: Qwen2.5-0.5B")
    ax.set_ylim(0, max(ppls) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, out_dir / "ppl.png", show)


# ---------------------------------------------------------------------------
# 2. Decode throughput
# ---------------------------------------------------------------------------

def plot_throughput(results: dict, out_dir: Path, show: bool) -> None:
    tput_data = {k: v["decode_tok_s"] for k, v in results.items() if "decode_tok_s" in v}
    if not tput_data:
        print("  [skip] no throughput data in results")
        return

    methods = list(tput_data.keys())
    tputs   = [tput_data[m] for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    colors  = [COLORS.get(m, "#999") for m in methods]
    ref_tput = tput_data.get("fp16", tputs[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, tputs, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    for bar, m, t in zip(bars, methods, tputs):
        speedup = t / ref_tput if ref_tput > 0 else 1.0
        tag = f"{t:.1f}" if m == "fp16" else f"{t:.1f}\n({speedup:.2f}×)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                tag, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Decode throughput (tok/s ↑)")
    ax.set_title("Decode Speed: Qwen2.5-0.5B, batch=1")
    ax.set_ylim(0, max(tputs) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, out_dir / "throughput.png", show)


# ---------------------------------------------------------------------------
# 3. Memory footprint
# ---------------------------------------------------------------------------

def plot_memory(results: dict, out_dir: Path, show: bool) -> None:
    mem_data = {k: v["total_mb"] for k, v in results.items() if "total_mb" in v}
    if not mem_data:
        print("  [skip] no memory data in results")
        return

    methods = list(mem_data.keys())
    mbs     = [mem_data[m] for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    colors  = [COLORS.get(m, "#999") for m in methods]
    ref_mb  = mem_data.get("fp16", mbs[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [m / 1000 for m in mbs], color=colors,
                  width=0.5, edgecolor="white", linewidth=0.8)

    for bar, m, mb in zip(bars, methods, mbs):
        ratio = ref_mb / mb if mb > 0 else 1.0
        tag = f"{mb/1000:.2f} GB" if m == "fp16" else f"{mb/1000:.2f} GB\n({ratio:.1f}× less)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                tag, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Weight memory (GB)")
    ax.set_title("Weight Footprint: Qwen2.5-0.5B projection layers")
    ax.set_ylim(0, max(m / 1000 for m in mbs) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, out_dir / "memory.png", show)


# ---------------------------------------------------------------------------
# 4. Roofline diagram
# ---------------------------------------------------------------------------

def plot_roofline(results: dict, out_dir: Path, show: bool, m4_bw_gbs: float = 120.0) -> None:
    """Arithmetic-intensity roofline for the decode GEMV kernel."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Roofline ceiling: bandwidth bound (straight line)
    ai_range = np.logspace(-3, 2, 400)        # FLOP/byte
    roof_bw  = m4_bw_gbs * ai_range           # GFLOP/s = BW(GB/s) × AI(FLOP/B)
    ax.loglog(ai_range, roof_bw, "k--", lw=1.5, label=f"M4 bandwidth ceiling ({m4_bw_gbs:.0f} GB/s)")

    # Our kernel operating points from results (if present)
    if "neon_bw_gbs" in results.get("kernel", {}):
        k = results["kernel"]
        bw    = k["neon_bw_gbs"]        # GB/s achieved
        ai_q4 = k.get("ai_q4", 0.25)   # typical: 0.5 FLOP / 0.5 byte = 1, but
                                         # with cache effects often lower
        perf  = bw * ai_q4
        ax.scatter([ai_q4], [perf], s=120, color=COLORS["rtn"], zorder=5,
                   label=f"NEON q4 kernel ({bw:.1f} GB/s achieved)")

    # Annotate typical regions
    ax.axvline(1.0, color="#aaa", lw=0.8, linestyle=":")
    ax.text(1.1, 0.5, "compute\nbound →", fontsize=8, color="#666", va="center")
    ax.text(0.9, 0.5, "← memory\nbound", fontsize=8, color="#666", va="center", ha="right")

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.set_title("Roofline: M4 Base, decode GEMV")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, out_dir / "roofline.png", show)


# ---------------------------------------------------------------------------
# 5. Summary table (text, always printed)
# ---------------------------------------------------------------------------

def print_table(results: dict) -> None:
    methods = list(results.keys())
    if "kernel" in methods:
        methods.remove("kernel")

    ref = results.get("fp16", {})
    ref_ppl  = ref.get("ppl", None)
    ref_tput = ref.get("decode_tok_s", None)
    ref_mem  = ref.get("total_mb", None)

    print("\n" + "=" * 82)
    print(f"{'Method':<18} {'bits':>4} {'PPL':>8} {'ΔPPL':>7} "
          f"{'decode tok/s':>13} {'speedup':>9} {'mem MB':>8} {'compression':>12}")
    print("-" * 82)

    for m in methods:
        v = results[m]
        bits     = 4 if m != "fp16" else 16
        ppl      = v.get("ppl")
        tput     = v.get("decode_tok_s")
        mem      = v.get("total_mb")

        ppl_str  = f"{ppl:.2f}"  if ppl  is not None else "—"
        dppl_str = (f"+{ppl-ref_ppl:.2f}" if ref_ppl and ppl and m != "fp16"
                    else ("—" if m == "fp16" else "?"))
        tput_str = f"{tput:.1f}"  if tput is not None else "—"
        spd_str  = (f"{tput/ref_tput:.2f}×" if ref_tput and tput else "—")
        mem_str  = f"{mem:.0f}"  if mem  is not None else "—"
        comp_str = (f"{ref_mem/mem:.1f}×" if ref_mem and mem and m != "fp16"
                    else ("—" if m == "fp16" else "?"))

        print(f"{METHOD_LABELS.get(m, m):<18} {bits:>4} {ppl_str:>8} {dppl_str:>7} "
              f"{tput_str:>13} {spd_str:>9} {mem_str:>8} {comp_str:>12}")

    print("=" * 82 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="Path to results JSON file.")
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for PNGs (default: same as results_json).")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    with open(results_path) as f:
        results = json.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.show:
        matplotlib.use("TkAgg")

    print_table(results)
    print(f"Generating plots in {out_dir} ...")
    plot_ppl(results, out_dir, args.show)
    plot_throughput(results, out_dir, args.show)
    plot_memory(results, out_dir, args.show)
    plot_roofline(results, out_dir, args.show)
    print("Done.")


if __name__ == "__main__":
    main()
