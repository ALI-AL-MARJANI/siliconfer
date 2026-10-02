# AWQ: this implementation vs the official one

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration" (arXiv:2306.00978), and the official code
`mit-han-lab/llm-awq` (`awq/quantize/auto_scale.py`, `auto_clip.py`,
`pre_quant.py`), read on 2026-10-02.

`siliconfer/quant/awq.py` is written from scratch. This file lists where it
matches the reference and where it does not.

## Same as the reference

| Aspect | Both |
|---|---|
| Channel statistic | `mean(|x|)` per input channel over all calibration tokens |
| Scale | `s = stat^α`, one α shared by projections that share an input |
| Search | 20-point grid over α, lowest output MSE wins |
| Groups | {q, k, v}, {gate, up}, {down} |
| Clipping (`--awq_clip`) | per (row, group): shrink the range in 5% steps up to 50%, keep the lowest partial-output MSE on 512 sampled tokens; q and k are never clipped |
| Block loss (`--awq_block_loss`) | α for {q, k, v} is scored on the attention module's output, α for {gate, up} on the MLP's output, with the whole group quantized |

## Different from the reference

| Aspect | llm-awq | Here | Why / effect |
|---|---|---|---|
| Default α loss | block output | one projection's own output (q_proj for {q,k,v}, gate_proj for {gate,up}) unless `--awq_block_loss` | The original implementation here. Kept as the default so earlier result files stay reproducible; the block-loss variant is reported as its own row. |
| Default clipping | on | off unless `--awq_clip` | Same reason. |
| α grid | {0, 0.05, …, 0.95} | {0, 0.05, …, 1.0} | α = 0 is the RTN starting point in both. One extra candidate here. |
| Scale normalisation | `s / sqrt(max(s)·min(s))` | none | A constant factor on `s` cancels exactly under per-(row, group) scales: `Q(c·W)/c = Q(W)`. It only matters for fp16 range safety when the scale is folded into a stored fp16 weight, which this code does not do (see next row). |
| Where `1/s` goes | folded into the previous operator (RMSNorm, `v_proj`, `up_proj`) | kept as a per-layer `input_scale` vector applied to the activation | Folding `v → o` needs `v_proj` and `o_proj` to have the same shape, which fails under grouped-query attention (Qwen2.5-0.5B: 128×896 vs 896×896). Cost here: one elementwise multiply per linear call. See `docs/packing.md`. |
| `o_proj` | not scaled when `v_proj`/`o_proj` shapes differ (the GQA case) | scaled, with its own α | Possible because of `input_scale`. An extension, not part of the reference behaviour for this model. |
| Quantizer inside the search | asymmetric (zero-point) by default | symmetric by default (`--asym` for asymmetric) | The symmetric NEON kernel is the main serving path here. |
| Calibration flow | each layer's inputs come from the *unquantized* previous layers | inputs come through the *already quantized* previous layers (cascaded) | Cascading lets later layers see the error they will see at inference. |
| Calibration data | 128 × 512 tokens from Pile validation | 128 × 512 tokens from WikiText-2 train | Kept in-distribution-agnostic check via the C4 evaluation set. |
| Block-loss sample | all calibration sequences | first 32 calibration sequences | Runtime on a 16 GB laptop. |

## Not compared

No layer-by-layer numerical comparison against `llm-awq` or AutoAWQ output has
been run: those need PyTorch/CUDA-oriented tooling, and the differences above
(symmetric quantizer, cascaded calibration, `o_proj` scaling) mean the weights
would not be expected to match even with a correct implementation. The claim
supported by this repository is therefore "follows the AWQ method, with the
listed deviations", not "reproduces llm-awq bit-for-bit".

## Results

See the README results table; rows `AWQ` (defaults) and
`AWQ + block loss + clip`, with the single-option ablations.
