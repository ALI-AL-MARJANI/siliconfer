"""Speculative decoding (Leviathan et al. 2022 / Chen et al. 2022).

A small draft model proposes K tokens; the large target model verifies all K+1
(including the anchor) in a single parallel forward pass. Accepted tokens are
committed; the first rejection triggers resampling from the adjusted target
distribution. The algorithm is lossless: the output distribution matches sampling
from the target model alone.

Expected speedup: (mean_accepted + 1) / (K * t_draft + t_target) vs 1 / t_target.
Requires t_draft << t_target and mean_accepted ≈ K for a net win.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import mlx.core as mx

from siliconfer.engine.generate import SamplingParams
from siliconfer.model.kv_cache import QuantizedKVCache, make_quantized_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_cache(cache, n: int):
    """Return cache sliced to first n KV positions (cheap MLX view).

    Handles both cache representations Attention.__call__ accepts (Phase 9b):
    plain (k, v) tuples are sliced into new arrays; QuantizedKVCache objects
    are trimmed in place (their own .trim() mutates packed codes + scales
    directly) and the same objects are returned.
    """
    if cache and isinstance(cache[0], QuantizedKVCache):
        for c in cache:
            c.trim(n)
        return cache
    return [(k[:, :, :n, :], v[:, :, :n, :]) for k, v in cache]


def _tok_prob(logits_1d: mx.array, tok_id: int, temperature: float) -> float:
    """Probability of tok_id under the distribution defined by logits_1d."""
    if temperature <= 0.0:
        best = int(mx.argmax(logits_1d).item())
        return 1.0 if tok_id == best else 0.0
    probs = mx.softmax(logits_1d / temperature)
    mx.eval(probs)
    return float(probs[tok_id].item())


def _sample_logits(logits_1d: mx.array, temperature: float) -> int:
    """Sample a token id from logits_1d with the given temperature."""
    if temperature <= 0.0:
        return int(mx.argmax(logits_1d).item())
    tok = mx.random.categorical(logits_1d / temperature)
    mx.eval(tok)
    return int(tok.item())


def _sample_adjusted(
    target_logits_1d: mx.array,
    draft_logits_1d: mx.array,
    temperature: float,
) -> int:
    """Sample from max(0, p_target - p_draft) / Z (rejection complement)."""
    if temperature <= 0.0:
        t_tok = int(mx.argmax(target_logits_1d).item())
        d_tok = int(mx.argmax(draft_logits_1d).item())
        return t_tok if t_tok != d_tok else t_tok

    p_t = mx.softmax(target_logits_1d / temperature)
    p_d = mx.softmax(draft_logits_1d / temperature)
    adj = mx.maximum(0.0, p_t - p_d)
    total = adj.sum()
    mx.eval(adj, total)
    total_val = float(total.item())
    if total_val < 1e-10:
        return _sample_logits(target_logits_1d, temperature)
    adj = adj / total_val
    tok = mx.random.categorical(mx.log(adj + 1e-30))
    mx.eval(tok)
    return int(tok.item())



# ---------------------------------------------------------------------------
# Phase 9d research note: a multi-candidate "retry with a fresh draft sample"
# scheme was attempted here and rejected after rigorous testing found it is
# NOT lossless. The intuition ("relative winning probabilities among i.i.d.
# retries don't depend on the number of trials") is true and easy to prove,
# but the conclusion drawn from it was wrong: as retry count M grows, values
# favored more by the draft model than the target model get accepted via the
# accept-path *more* than their fair p_target share, and there is no way to
# correct for this via an independent fallback draw — the exact fallback
# formula that would be required works out to a NEGATIVE probability for some
# tokens (confirmed algebraically, not just empirically: for M=2 candidates,
# `p_target(v) - min(p_draft(v),p_target(v))*(2-Z)` goes negative whenever
# p_draft(v) >= p_target(v) and Z is small enough). Clamping negatives to
# zero would "fix" this into a valid distribution but makes the scheme only
# *approximately* lossless — inconsistent with this project's standard of
# exact, provable correctness for speculative decoding (Phase 8's algorithm
# is verified via exact greedy token-for-token match, not just "close").
# This is exactly why the real tree-attention literature (SpecInfer, EAGLE-2)
# needs careful *correlated* multi-candidate verification, not naive
# independent retries — a genuinely harder problem than this shortcut
# assumed. See `dynamic_K` below for what Phase 9d actually ships instead.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeResult:
    token_ids: list[int]
    prefill_time: float
    decode_time: float
    num_prefill_tokens: int
    num_decode_tokens: int
    total_rounds: int
    total_draft_tokens: int
    total_accepted: int

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted / self.total_draft_tokens

    @property
    def effective_tok_s(self) -> float:
        if self.decode_time == 0:
            return 0.0
        return self.num_decode_tokens / self.decode_time

    @property
    def prefill_tok_s(self) -> float:
        if self.prefill_time == 0:
            return 0.0
        return self.num_prefill_tokens / self.prefill_time


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def speculative_generate(
    draft,
    target,
    prompt_ids: mx.array,
    params: SamplingParams | None = None,
    K: int = 4,
    eos_token_id: int | None = None,
    on_token: "Callable[[int], None] | None" = None,
    seed: int | None = None,
    dynamic_K: bool = False,
    K_min: int = 1,
    K_max: int = 8,
    quantize_kv_cache: bool = False,
) -> SpeculativeResult:
    """Generate tokens using speculative decoding.

    Args:
        draft:          Small model used to propose K candidate tokens per round.
        target:         Large model used to verify and correct.
        prompt_ids:     [1, T] or [T] integer array (prompt token ids).
        params:         Sampling parameters (temperature, top_p, etc.).
        K:              Speculation depth — number of draft tokens per round.
                        With dynamic_K=True this is only the *starting* depth.
        eos_token_id:   Stop generation when this token is produced.
        on_token:       Callback invoked with each accepted/sampled token id.
        seed:           Random seed for reproducible sampling.
        dynamic_K:      (Phase 9d) adapt K round-to-round based on the running
                        acceptance rate — speculate deeper after rounds that
                        accepted everything (the draft model is "in sync" with
                        the target right now), pull back after a round rejects
                        early (wasted draft compute). This changes only which
                        K value each round's *already-proven-lossless*
                        rejection-sampling algorithm uses — K never appears in
                        that correctness proof, so this is losslessness-neutral
                        by construction (unlike multi-candidate schemes, see
                        the research note above `dynamic_K` docstring in the
                        module — that approach was tried and found NOT lossless).
        K_min, K_max:   Bounds for dynamic_K's adaptation.
        quantize_kv_cache: (Phase 9b+9d) store both draft's and target's KV
                        cache as group-wise int8 instead of fp16. `_trim_cache`
                        and `Attention.__call__` already handle both cache
                        representations transparently (Phase 9b), so this is
                        just correctly initializing both caches with
                        `make_quantized_cache()` instead of `None`. Note this
                        makes speculative decoding exactly reproduce
                        *non-speculative generation from this same
                        quantized-cache model* (verified — see
                        `test_speculative.py`), not the fp16-cache model's
                        output; quantized-cache decoding is itself only
                        approximately equal to fp16 (Phase 9b: ΔPPL ≈ +1.2).

    Returns:
        SpeculativeResult with token_ids and statistics.

    Algorithm (per round):
        1. Draft generates K tokens autoregressively from the last accepted token
           (K itself may change round-to-round if dynamic_K=True).
        2. Target verifies [last_accepted, d_0, ..., d_{K-1}] in one forward pass
           (K+1 tokens), producing K+1 logit vectors.
        3. For q = 0 .. K-1:
               p_t = P_target(d_q)  at position q
               p_d = P_draft(d_q)   at position q
               Accept d_q with prob min(1, p_t / p_d).
               If rejected: replace d_q with sample from max(0, P_target - P_draft).
        4. If all K accepted: sample one bonus token from target_logits[:, K, :].
        5. Output accepted tokens + the bonus/replacement token.
        6. Trim KV caches to match the committed context length.
        7. If dynamic_K: adjust K for the next round based on whether this
           round accepted everything (K += 1, capped at K_max) or rejected
           early (K -= 1, floored at K_min).
    """
    if params is None:
        params = SamplingParams()
    if seed is not None:
        mx.random.seed(seed)

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]
    T = prompt_ids.shape[1]
    temp = params.temperature

    # ------------------------------------------------------------------ prefill
    target_cache = make_quantized_cache(len(target.layers)) if quantize_kv_cache else None
    draft_cache = make_quantized_cache(len(draft.layers)) if quantize_kv_cache else None

    t0 = time.perf_counter()
    t_logits, target_cache = target(prompt_ids, target_cache)
    d_logits, draft_cache  = draft(prompt_ids, draft_cache)
    mx.eval(t_logits, d_logits)
    t_prefill = time.perf_counter()

    # Sample first token from target
    cur_tok = _sample_logits(t_logits[0, -1], temp)
    if on_token is not None:
        on_token(cur_tok)
    generated: list[int] = [cur_tok]

    # cache size N = number of KV entries = T (prompt tokens processed so far)
    N = T

    total_rounds = 0
    total_draft_tokens = 0
    total_accepted = 0

    cur_K = K

    t_decode_start = time.perf_counter()

    while len(generated) < params.max_tokens:
        round_K = cur_K   # this round's speculation depth (cur_K may change below)

        # --------------------------------------------------------- draft phase
        draft_tokens: list[int] = []
        draft_logits_list: list[mx.array] = []  # each [1, 1, vocab]

        d_input = mx.array([[cur_tok]])
        for _ in range(round_K):
            d_log, draft_cache = draft(d_input, draft_cache)
            mx.eval(d_log)
            d_id = _sample_logits(d_log[0, 0], temp)
            draft_tokens.append(d_id)
            draft_logits_list.append(d_log)
            d_input = mx.array([[d_id]])

        total_draft_tokens += round_K

        # -------------------------------------------------------- target verify
        # Batch [cur_tok, d_0, ..., d_{round_K-1}] — round_K+1 tokens at positions N..N+round_K
        verify_ids = mx.array([[cur_tok] + draft_tokens])  # [1, round_K+1]
        t_log, target_cache = target(verify_ids, target_cache)
        mx.eval(t_log)
        # t_log[0, q] = target logits for predicting draft_tokens[q]
        # t_log[0, round_K] = target logits for the bonus token

        # ------------------------------------------------------ rejection sample
        j = round_K  # number of accepted draft tokens (default: all)
        replacement = -1

        for q in range(round_K):
            p_t = _tok_prob(t_log[0, q], draft_tokens[q], temp)
            p_d = _tok_prob(draft_logits_list[q][0, 0], draft_tokens[q], temp)

            accept_prob = min(1.0, p_t / (p_d + 1e-12))
            if random.random() < accept_prob:
                continue  # accepted

            # Rejected at position q
            j = q
            replacement = _sample_adjusted(t_log[0, q], draft_logits_list[q][0, 0], temp)
            break
        else:
            # All round_K accepted — sample bonus from target_logits[:, round_K, :]
            replacement = _sample_logits(t_log[0, round_K], temp)

        total_accepted += j

        # ------------------------------------------- dynamic K adaptation
        # K never appears in the accept/reject correctness proof above — this
        # only changes how many tokens the *next* round speculates, which is
        # losslessness-neutral by construction.
        if dynamic_K:
            if j == round_K:
                cur_K = min(round_K + 1, K_max)   # fully accepted — speculate deeper
            else:
                cur_K = max(round_K - 1, K_min)   # rejected early — pull back

        # ------------------------------------------- commit accepted tokens
        # Clip batch to remaining budget before appending (on_token must not
        # fire for tokens that will be discarded by max_tokens).
        new_batch = draft_tokens[:j] + [replacement]
        budget = params.max_tokens - len(generated)
        new_batch = new_batch[:budget]

        for tok in new_batch:
            generated.append(tok)
            if on_token is not None:
                on_token(tok)

        cur_tok = generated[-1]

        # ------------------------------------------- sync KV caches
        new_N = N + j + 1  # cache covers positions 0 .. N+j (N+j+1 entries)

        target_cache = _trim_cache(target_cache, new_N)

        if j == round_K:
            # draft_cache at N+round_K; d_{round_K-1} accepted but never INPUT to draft.
            # Run draft on d_{round_K-1} to extend its cache to N+round_K+1 = new_N.
            _, draft_cache = draft(mx.array([[draft_tokens[round_K - 1]]]), draft_cache)
            mx.eval(draft_cache)
        else:
            # draft_cache at N+round_K from the speculation run; trim to new_N.
            draft_cache = _trim_cache(draft_cache, new_N)

        N = new_N
        total_rounds += 1

        # Stop if any committed token was eos, or max_tokens reached
        if len(generated) >= params.max_tokens:
            break
        if eos_token_id is not None and any(t == eos_token_id for t in new_batch):
            break

    t_done = time.perf_counter()

    return SpeculativeResult(
        token_ids=generated,
        prefill_time=t_prefill - t0,
        decode_time=t_done - t_decode_start,
        num_prefill_tokens=T,
        num_decode_tokens=len(generated) - 1,
        total_rounds=total_rounds,
        total_draft_tokens=total_draft_tokens,
        total_accepted=total_accepted,
    )
