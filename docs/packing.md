# Packing quantized weights without changing them

Every quantizer in `siliconfer/quant/` produces a *fake-quantized* float
weight (quantize, then dequantize), which is what algorithm-level evaluation
uses. The serving path packs it into 4-bit codes plus per-group scales
(`engine/q4_loader.py::_pack_and_replace_linears`) and runs it through a
kernel (`model/q4_linear.py::Q4Linear`).

Packing is itself a quantization step. It leaves a weight unchanged only if
the weight already lies on the grid the packer uses. This note lists the
cases where that needs care, and how each is handled. All of them are
covered by tests that compare the packed layer against the quantizer's own
output.

## The invariant

For one output row and one group of `group_size` input columns, the
symmetric packer stores `scale = max|w| / 7` and codes `q = round(w / scale)`.

A symmetric round-to-nearest group is `q₀ · scale₀` with integer
`q₀ ∈ [−7, 7]`, and its largest element has `|q₀| = 7` because `scale₀` was
defined from it. So `max|w| = 7·scale₀`, the packer recovers `scale₀` and the
same codes, and re-packing is a no-op.

## Case 1 — asymmetric grids

An asymmetric group is `(q − zero) · scale` with `q ∈ [0, 15]`. Its values
are not integer multiples of `max|w| / 7`, so it must be packed with the
asymmetric packer and multiplied by the asymmetric kernel.

`pack_weights_asym`, `q4_gemv_asym` / `q4_gemm_asym`, and the optional
`zeros` array of `Q4Linear` handle this. The HQQ-style quantizer is always
asymmetric; the others follow the `sym` flag.

Tests: `tests/test_kernel.py::test_gemv_asym_matches_reference`,
`test_gemm_asym_matches_reference_matmul`;
`tests/test_integration.py::test_q4linear_asym_matches_fake_quant`,
`test_pack_and_replace_linears_asym_mode`.

## Case 2 — per-column scales (AWQ, SINQ-style)

Both methods rescale input columns before quantizing:

```
W·X = (W·diag(s)) · (diag(s)⁻¹·X)        s ∈ ℝ^in, one entry per input column
```

and quantize `W·diag(s)`. The algorithm-level weight is

```
W_eff = Q(W·diag(s)) · diag(s)⁻¹
```

`Q(W·diag(s))` lies on the group grid. `W_eff` does not: inside one group,
column `j` is multiplied by its own `1/s[j]`, so the step differs from column
to column and no single per-group scale represents the group. Packing `W_eff`
would quantize it a second time and discard the per-column resolution the
method had just bought.

The serving path therefore keeps the two factors separate:

- it packs `W_grid = Q(W·diag(s))`, which is lossless;
- `Q4Linear` stores `input_scale = 1/s` and computes
  `y = dequant(W_packed) · (x ⊙ input_scale)`.

Cost: one elementwise multiply over `in_features` floats per call.

The reference AWQ implementation instead folds `1/s` into the preceding
operator (the RMSNorm for q/k/v and gate/up; `v_proj → o_proj`,
`up_proj → down_proj`). That costs nothing at run time, but `v → o` folding
needs the two weights to have the same shape, which fails under
grouped-query attention (Qwen2.5-0.5B: `v_proj` is 128×896, `o_proj` is
896×896). An explicit input scale handles every projection the same way.

Tests: `tests/test_awq.py::test_awq_packed_roundtrip_matches_weff`,
`test_awq_packing_weff_directly_is_wrong` (and the same pair in
`tests/test_sinq.py`);
`tests/test_integration.py::test_pack_and_replace_linears_awq_uses_grid_aligned_weight`.

## Case 3 — a grid fixed before the weights move (GPTQ)

GPTQ fixes each group's scale (and zero-point) from the original weight, then
quantizes column by column while error feedback shifts the columns not yet
quantized. The shifted weights are rounded and clamped on the original grid,
so a group can end up

- using code −8 (the symmetric range is [−8, 7]), which makes its largest
  magnitude `8·scale`, or
- never reaching ±7, which makes it `6·scale` or less.

Either way `max|w| / 7` is no longer the scale GPTQ used, and re-deriving the
grid from the weight would re-round the whole group. In the first three
layers of Qwen2.5-0.5B this concerns 1.7% of the groups.

`gptq_quantize_weight(..., return_grid=True)` returns the scales and
zero-points it used, `apply_gptq` attaches them to the projection, and
`pack_weights_on_grid` recovers the codes as `round(w / scale) (+ zero)` on
that grid, which is exact.

Tests: `tests/test_kernel.py::test_pack_on_grid_is_exact_where_rederiving_the_grid_is_not`,
`test_pack_on_grid_asymmetric_partial_range`;
`tests/test_integration.py::test_pack_and_replace_linears_gptq_packs_on_its_own_grid`.

## Mixed precision

There is no 2-bit or 3-bit packed format. `load_q4_model(method="mixed")`
raises instead of storing those layers at 4 bits on a different grid; mixed
precision is evaluated as fake-quant only.

## Dtypes

Quantized layers compute in float32 and return float32. Two places take care
not to let that promote fp16 tensors:

- RoPE computes its angles in float32 and casts the cos/sin tables to the
  input's dtype. Left in float32 they would promote the queries and keys, and
  through them every later activation, so that each matmul re-casts its fp16
  weight on every call.
- The output head casts its input to the head's own dtype, so a float32
  activation does not force a per-token cast of the vocabulary matrix.
