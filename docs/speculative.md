# Speculative decoding

`siliconfer/engine/speculative.py`. References: Leviathan et al., "Fast
Inference from Transformers via Speculative Decoding" (arXiv:2211.17192);
Chen et al., "Accelerating Large Language Model Decoding with Speculative
Sampling" (arXiv:2302.01318).

## Algorithm

A cheap *draft* model proposes `K` tokens autoregressively; the *target*
model scores all of them in one forward pass. With `p` the target
distribution and `q` the draft distribution at a position, a drafted token
`x ~ q` is

- accepted with probability `min(1, p(x)/q(x))`;
- otherwise rejected, and a replacement is drawn from the residual
  `r(v) ∝ max(0, p(v) − q(v))`. Everything after the rejected position is
  discarded.

If all `K` are accepted, one extra token is sampled from the target's
distribution at position `K+1`, which the same forward pass already produced.

**Why the output distribution is exactly the target's.** The probability of
emitting `v` at a position is

```
P(v) = q(v)·min(1, p(v)/q(v))  +  (1 − Z)·r(v)        Z = Σ_u min(p(u), q(u))
     = min(p(v), q(v))         +  max(0, p(v) − q(v))
     = p(v)
```

since `Σ_v max(0, p(v) − q(v)) = 1 − Z` is exactly the normaliser of `r`.

With temperature 0 both distributions are one-hot, acceptance is "do the
argmaxes agree", and the output equals greedy decoding of the target token
for token. `tests/test_speculative.py` asserts that exact match.

**Cache bookkeeping.** After a round that accepts `j` of `K` drafted tokens,
both KV caches are trimmed to the committed length. When all `K` are
accepted the draft has not yet processed its own last token, so one extra
draft step re-synchronises it. Both plain and int8-quantized caches are
supported.

## Dynamic depth

`dynamic_K=True` raises `K` by one after a fully accepted round and lowers it
by one after an early rejection, within `[K_min, K_max]`. `K` only decides
how many tokens are drafted before verifying; it does not appear in the
accept/reject rule, so the argument above is unchanged.

## A scheme that does not work: independent retries

A natural shortcut for getting more out of each target call is to retry: on
a rejection, draw a fresh draft token at the same position and test it
again, up to `M` times, before falling back to the residual. This is **not**
lossless for `M ≥ 2`.

Let `a(v) = min(p(v), q(v))`, so one attempt emits `v` with probability
`a(v)` and fails with probability `1 − Z`. With `M` independent attempts the
accept path emits `v` with probability

```
a(v)·(1 + (1−Z) + … + (1−Z)^(M−1)) = a(v)·(1 − (1−Z)^M)/Z
```

For the total to equal `p(v)`, the fallback (reached with probability
`(1−Z)^M`) would have to contribute `p(v) − a(v)·(1 − (1−Z)^M)/Z`. Take
`M = 2` and any token the draft over-weights, `q(v) ≥ p(v)`, so `a(v) = p(v)`:

```
p(v) − p(v)·(2 − Z) = p(v)·(Z − 1) < 0      whenever Z < 1
```

A negative probability: the accept path has already emitted such tokens more
often than the target would, and no fallback distribution can take that back.
Clamping at zero gives a valid sampler that is only approximately correct.
Multi-candidate methods in the literature (SpecInfer, EAGLE-2) verify a tree
of *correlated* candidates for this reason.

`tests/test_speculative.py::test_naive_multicandidate_retry_requires_negative_fallback_probability`
checks the algebra. The scheme is not implemented.

## Measurements

`scripts/eval_speculative.py` → `results/claims/speculative.json`.

1. **Distribution check.** Two small random models with different weights,
   temperature 1. The first three generated tokens are sampled 20,000 times
   and compared by chi-square with the target's exact distribution, obtained
   by enumerating every prefix. The same test is run on ordinary sampling as
   a control. See the README for the p-values.
2. **Real model.** Target: the fp16 model. Draft: the same model with int4
   weights. This is the only draft available without a second model, and it
   costs nearly as much per token as the target, so it cannot produce a
   speedup; what it measures is the acceptance rate of an int4 model against
   its own fp16 original and the overhead of the mechanism.

## Limitations

- No smaller draft model has been evaluated, so there is no measured speedup.
- Each draft token and each acceptance test forces an evaluation and a
  Python-side scalar read; the loop is written for clarity, not throughput.
- The trained draft head (`model/draft_head.py`) is not connected to this
  loop.
