# Evaluation protocol

The perplexity protocol used for every number reported in the README.

## Protocol

- **Dataset:** WikiText-2 raw test split, loaded via
  `Salesforce/wikitext`/`wikitext-2-raw-v1`. A second, independent dataset (a
  fixed subset of the C4 `en` validation split, selected by `--c4_seed`, which
  is independent of the calibration seed) is used to check that a method's
  ranking is not an artifact of one dataset.
- **Two tiers.** `scripts/run_ppl_matrix.sh` runs the same matrix at two
  budgets:
  - *quick tier* (`results/ppl/`): the first 32,768 tokens of WikiText-2 test
    (16 windows) and 16 C4 windows. About one to five minutes per run.
  - *full tier* (`results/ppl_full/`): the whole WikiText-2 test split
    (145 non-overlapping windows = 296,960 scored tokens of its 298,938) and
    64 C4 windows. The GPTQ/AWQ papers
    evaluate C4 on 256 windows; 64 is a runtime compromise for a laptop.

  The two tiers are not comparable to each other (the fp16 perplexity itself
  differs between a 32,768-token prefix and the full split). Compare methods
  only within a tier. Every result file records the token budget and, for
  C4, a SHA-256 of the scored token ids, so equal test sets can be checked.
- **Context length:** 2048 tokens, non-overlapping windows. This matches the
  GPTQ/AWQ papers' own evaluation protocol.
- **Precision:** cross-entropy computed in float32 with a numerically stable
  log-sum-exp, regardless of the model's storage dtype.
- **Calibration (GPTQ/AWQ only):** 128 sequences of 512 tokens from the
  WikiText-2 *train* split (disjoint from the *test* split used for
  evaluation), sampled with an explicit, varyable seed
  (`--calib_seed`). Three different seeds should be run and the result
  reported as mean ± std — a single-seed number does not distinguish a real
  effect from calibration-sample noise.
- **Implementation:** `scripts/eval_ppl.py`, calling
  `siliconfer.eval.perplexity.compute_perplexity`. Every run writes a JSON to
  `results/ppl/` with the full config, the measured PPL, wall-clock time, and
  environment metadata (git SHA, hardware, library versions) from
  `siliconfer.eval.env_info`.

## Why a single fixed protocol

A short context and a few thousand tokens are too noisy to separate methods
that differ by one perplexity point, and numbers measured at different
context lengths are not comparable. Every reported perplexity therefore comes
from `scripts/eval_ppl.py` under the settings above. `scripts/quantize.py`
remains for algorithm-level (fake-quant) experiments.

## fp16 reference validation

`scripts/validate_fp16_reference.py` computes PPL twice on the identical
tokenized, chunked WikiText-2 test set: once with this project's from-scratch
MLX decoder, once with `transformers.AutoModelForCausalLM` (float32, MPS
backend). It reports the absolute difference rather than asserting a
threshold, so the number in the README is always the one actually measured.
See `results/ppl/fp16_reference_validation.json` for the full-test-set run.

## Known limitation of this protocol

Non-overlapping 2048-token windows discard cross-window context (each window
is scored independently, with no KV cache carried from the previous window).
This matches common practice in the GPTQ/AWQ papers but is not the only valid
protocol (sliding-window evaluation with overlap is another common choice,
and gives slightly lower/more favorable PPL since every token except the
first `stride` tokens of the corpus gets full context). This project reports
the non-overlapping-window number only; it should not be compared directly to
a sliding-window number from another paper or library without checking which
protocol that other number used.
