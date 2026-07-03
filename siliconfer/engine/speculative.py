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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_cache(cache, n: int):
    """Return cache sliced to first n KV positions (cheap MLX view)."""
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
) -> SpeculativeResult:
    """Generate tokens using speculative decoding.

    Args:
        draft:          Small model used to propose K candidate tokens per round.
        target:         Large model used to verify and correct.
        prompt_ids:     [1, T] or [T] integer array (prompt token ids).
        params:         Sampling parameters (temperature, top_p, etc.).
        K:              Speculation depth — number of draft tokens per round.
        eos_token_id:   Stop generation when this token is produced.
        on_token:       Callback invoked with each accepted/sampled token id.
        seed:           Random seed for reproducible sampling.

    Returns:
        SpeculativeResult with token_ids and statistics.

    Algorithm (per round):
        1. Draft generates K tokens autoregressively from the last accepted token.
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
    t0 = time.perf_counter()
    t_logits, target_cache = target(prompt_ids)
    d_logits, draft_cache  = draft(prompt_ids)
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

    t_decode_start = time.perf_counter()

    while len(generated) < params.max_tokens:
        # --------------------------------------------------------- draft phase
        draft_tokens: list[int] = []
        draft_logits_list: list[mx.array] = []  # each [1, 1, vocab]

        d_input = mx.array([[cur_tok]])
        for _ in range(K):
            d_log, draft_cache = draft(d_input, draft_cache)
            mx.eval(d_log)
            d_id = _sample_logits(d_log[0, 0], temp)
            draft_tokens.append(d_id)
            draft_logits_list.append(d_log)
            d_input = mx.array([[d_id]])

        total_draft_tokens += K

        # -------------------------------------------------------- target verify
        # Batch [cur_tok, d_0, ..., d_{K-1}] — K+1 tokens at positions N..N+K
        verify_ids = mx.array([[cur_tok] + draft_tokens])  # [1, K+1]
        t_log, target_cache = target(verify_ids, target_cache)
        mx.eval(t_log)
        # t_log[0, q] = target logits for predicting draft_tokens[q]
        # t_log[0, K] = target logits for the bonus token

        # ------------------------------------------------------ rejection sample
        j = K  # number of accepted draft tokens (default: all)
        replacement = -1

        for q in range(K):
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
            # All K accepted — sample bonus from target_logits[:, K, :]
            replacement = _sample_logits(t_log[0, K], temp)

        total_accepted += j

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

        if j == K:
            # draft_cache at N+K; d_{K-1} accepted but never INPUT to draft.
            # Run draft on d_{K-1} to extend its cache to N+K+1 = new_N.
            _, draft_cache = draft(mx.array([[draft_tokens[K - 1]]]), draft_cache)
            mx.eval(draft_cache)
        else:
            # draft_cache at N+K from the speculation run; trim to new_N.
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
