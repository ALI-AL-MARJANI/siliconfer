# Quantization methods

What each method in `siliconfer/quant/` computes, and where it departs from
the paper or reference implementation it is named after. All methods quantize
the seven projections of every decoder block (`q/k/v/o_proj`,
`gate/up/down_proj`). The token embedding (tied to the output head in
Qwen2.5-0.5B), the normalisation weights and the attention biases stay in
fp16.

## Storage format: group-wise int4

A weight row is split into groups of `group_size` consecutive input columns
(128 by default). Each (row, group) stores one float32 scale, plus one
float32 zero-point in the asymmetric case; each weight stores a 4-bit code.

| | code range | value | scale |
|---|---|---|---|
| symmetric | −8 … 7 | `code · scale` | `max|w| / 7` |
| asymmetric | 0 … 15 | `(code − zero) · scale` | `(max − min) / 15`, `zero = round(−min/scale)` |

Two codes are packed per byte (low nibble = even column). Bits per weight:
`4 + 32/group_size` symmetric, `4 + 64/group_size` asymmetric, i.e. 4.25 and
4.5 at group size 128.

Symmetric rounding of the original weight never produces −8 (the largest
magnitude maps to ±7 by construction). Methods that move the weights after
fixing the grid can produce it; see `docs/packing.md`, case 3.

## RTN

Round every weight to the nearest code on the grid above. The baseline every
other method is compared with.

## GPTQ

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", arXiv:2210.17323; code
`IST-DASLab/gptq`.

Per layer, minimise the output error on calibration inputs `X`:

```
argmin_Ŵ ‖W·X − Ŵ·X‖²_F          H = 2·X·Xᵀ   (shared by all rows)
```

Quantize columns left to right; after rounding column `q`, push its error
onto the columns not yet quantized so that the layer output changes as little
as possible:

```
H ← H + λ·mean(diag H)·I                    λ = 0.01
U = upper Cholesky factor of H⁻¹            H⁻¹ = Uᵀ·U
for q = 0 … in−1 (in blocks of 128):
    ŵ_q = round_to_grid(w_q)
    e   = (w_q − ŵ_q) / U[q, q]
    W[:, q+1:] −= e · U[q, q+1:]
```

`U[q, q]²` is the inverse-Hessian diagonal *conditioned on the columns
already quantized* (a Schur complement), which is what the update needs;
`H⁻¹[q, q]` is the unconditional value and gives a wrong update.

Layers are processed in order and each layer's Hessian is collected from the
output of the already-quantized layers before it.

Differences from the reference:

| Aspect | IST-DASLab/gptq | Here |
|---|---|---|
| Group grid | re-computed from the error-corrected weights when a group is reached (`static_groups=False`, the default) | fixed from the original weights before the loop (the reference's `static_groups=True` behaviour) |
| Column order | optional activation ordering (`act_order`) | natural order only |
| Calibration | 128 × 2048 tokens of C4 | 128 × 512 tokens of WikiText-2 train |
| Dampening, block size | 0.01, 128 | same |

No layer-by-layer numerical comparison against GPTQModel/AutoGPTQ has been
run.

## AWQ

Reference: Lin et al., arXiv:2306.00978; code `mit-han-lab/llm-awq`.

```
W·X = (W·diag s) · (diag(s)⁻¹·X)        s_j = mean|x_j|^α
```

Scaling a column up before rounding gives it finer effective resolution;
`α ∈ [0, 1]` is grid-searched to minimise output error. The serving path
stores `Q(W·diag s)` and applies `1/s` to the layer input
(`docs/packing.md`, case 2).

Two options follow the reference more closely (block-output loss and weight
clipping). The full list of matches and differences is in
`docs/awq-vs-reference.md`.

## HQQ-style clip search (calibration-free)

Named after Badri & Shaji, "Half-Quadratic Quantization" (Mobius Labs, 2023;
code `mobiusml/hqq`). **This is not the HQQ algorithm.** HQQ solves for the
quantization parameters by a half-quadratic (proximal) iteration on an `ℓ_p`
loss; the implementation here keeps HQQ's objective and replaces its solver
with a small candidate search:

```
for each (row, group):
    m, d = median(w), 1.4826 · MAD(w)
    for k in (150, 100, 80, ∞):                 # ∞ = plain min/max
        clip w to [m − k·d, m + k·d]; quantize asymmetrically; dequantize
        loss_k = Σ |w − ŵ|^p                    p = 0.7
    keep the candidate with the smallest loss
```

It uses only the weights. It is always asymmetric.

Why the thresholds are so large: a clip search that trims ordinary tail
values (the first version trimmed up to 30% by percentile) passed its
synthetic tests and then destroyed the real model. One weight in an early
`down_proj`, far outside its group's distribution, turned out to be
structurally important — the effect described by Yu et al., "The Super Weight
in Large Language Models" (arXiv:2411.07191). A reconstruction loss on
weights alone cannot tell such a weight from a harmless outlier; that needs
sensitivity information, which a calibration-free method does not have. The
default grid is therefore conservative enough to leave isolated extreme
values alone, and behaves like asymmetric RTN on most groups. (The incident was observed during development and has not been re-measured
under the current protocol, so no number is quoted for it.)

## SINQ-style column rescaling (calibration-free)

Named after "SINQ: Sinkhorn-Normalized Quantization" (2025), which fits a row
scale and a column scale. Only the column half is implemented:

```
s = 1
repeat 10 times:
    Ŵ = Q(W·diag s) · diag(s)⁻¹
    r_j = RMS(W[:, j] − Ŵ[:, j]) / RMS(W[:, j])        # relative error per column
    s  ← clip(s · (r / mean r)^β, 0.1, 10)             β = 0.5
```

- **The row scale is a no-op here.** The quantizer already has an independent
  scale per (row, group). Multiplying a row by `t` multiplies its scales by
  `t` and leaves every code unchanged: `round(t·w / (t·scale)) = round(w/scale)`.
- **The update uses relative error.** Within a group the rounding step is
  shared, so absolute error is about the same for every column and carries no
  signal; relative error singles out small-magnitude columns that the shared
  step has crushed. `tests/test_sinq.py::test_sinq_relative_error_signal_matters`
  guards this.

Like AWQ, the serving path stores `Q(W·diag s)` and applies `1/s` to the
input.

## Mixed precision (research experiment, not served)

`quant/mixed_precision.py` ranks decoder blocks by a permutation-sampling
Shapley estimate of how much demoting each block to a lower bit-width raises
calibration loss, then demotes the least sensitive blocks under a memory
budget. Blocks all have the same size in this architecture, so the selection
is an exact top-k. The estimator is checked against closed-form Shapley
values on additive and pairwise-interaction games (`tests/test_mixed_precision.py`).

This is evaluated as fake-quant only. There is no 2-bit or 3-bit packed
kernel, so it gives no memory or speed benefit in the serving path, and the
loader refuses `method="mixed"` rather than storing those layers at 4 bits.
Results: `results/claims/mixed_precision_*.json`.

## Calibration data

GPTQ and AWQ use 128 sequences of 512 tokens sampled from the WikiText-2
*train* split with a seed (`--calib_seed`); results are reported over three
seeds. Evaluation uses the *test* split. Because calibration and the main
evaluation set come from the same corpus, the calibrated methods have an
in-domain advantage on WikiText-2 that the calibration-free methods do not;
the C4 evaluation is the check on that.
