"""Speculative decoding (Leviathan et al., arXiv:2211.17192; Chen et al., arXiv:2302.01318).

A draft model proposes K tokens; the target model scores all of them in one
forward pass. Each drafted token is accepted with probability
min(1, p_target / p_draft); the first rejection is replaced by a sample from
the residual max(0, p_target − p_draft), and everything after it is dropped.
The output distribution is exactly the target's. See docs/speculative.md.

A speedup needs a draft much cheaper than the target and a high acceptance
rate: (accepted + 1) tokens cost K draft steps plus one target pass.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from siliconfer.engine.generate import SamplingParams
from siliconfer.model.kv_cache import QuantizedKVCache, make_quantized_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_cache(cache, n: int):
    """Trim a KV cache to its first n positions.

    Plain (k, v) tuples are sliced into new arrays; QuantizedKVCache objects are
    trimmed in place and returned.
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



# Retrying a rejected position with fresh independent draft samples is not
# lossless: the fallback distribution it would need has negative entries.
# See docs/speculative.md and tests/test_speculative.py.


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
    on_token: Callable[[int], None] | None = None,
    seed: int | None = None,
    dynamic_K: bool = False,
    K_min: int = 1,
    K_max: int = 8,
    quantize_kv_cache: bool = False,
) -> SpeculativeResult:
    """Generate tokens with speculative decoding.

    Args:
        draft:          model that proposes K tokens per round.
        target:         model that verifies them.
        prompt_ids:     [1, T] or [T] prompt token ids.
        params:         sampling parameters.
        K:              tokens drafted per round (the starting value when
                        dynamic_K is set).
        eos_token_id:   stop when this token is produced.
        on_token:       callback invoked with each committed token id.
        seed:           random seed.
        dynamic_K:      raise K by one after a fully accepted round and lower it
                        by one after a rejection. K does not appear in the
                        accept/reject rule, so the output distribution is
                        unchanged.
        K_min, K_max:   bounds for dynamic_K.
        quantize_kv_cache: keep both models' KV caches as int8. The output then
                        matches non-speculative generation with the same
                        quantized cache, not generation with an fp16 cache.

    Returns:
        SpeculativeResult with token ids and acceptance statistics.

    Per round:
        1. The draft generates K tokens from the last committed token.
        2. The target scores [last, d_0, ..., d_{K-1}] in one pass (K+1 logits).
        3. For q = 0..K-1: accept d_q with probability
           min(1, p_target(d_q) / p_draft(d_q)); on rejection, sample a
           replacement from max(0, p_target − p_draft) and stop.
        4. If all K were accepted, sample one more token from the target's
           logits at position K.
        5. Commit the accepted tokens plus the replacement or extra token, and
           trim both KV caches to the committed length.
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
