"""Generation loop with sampling strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.model.kv_cache import make_quantized_cache


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_tokens: int = 128


def sample_token(
    logits: mx.array,
    params: SamplingParams,
    generated_ids: list[int] | None = None,
) -> mx.array:
    logits = logits[:, -1, :]

    if params.repetition_penalty != 1.0 and generated_ids:
        penalty = params.repetition_penalty
        token_ids = mx.array(generated_ids)
        penalty_logits = logits[:, token_ids]
        penalty_logits = mx.where(
            penalty_logits > 0,
            penalty_logits / penalty,
            penalty_logits * penalty,
        )
        logits[:, token_ids] = penalty_logits

    if params.temperature <= 0 or params.temperature == 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / params.temperature

    if params.top_k > 0:
        top_k_vals = mx.topk(logits, k=min(params.top_k, logits.shape[-1]))
        threshold = top_k_vals[:, -1:]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    if params.top_p < 1.0:
        sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        cutoff_mask = cumulative_probs - sorted_probs > params.top_p
        sorted_logits = mx.where(cutoff_mask, mx.array(float("-inf")), sorted_logits)

        restore_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, restore_indices, axis=-1)

    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))


@dataclass
class GenerationResult:
    token_ids: list[int]
    prefill_time: float
    decode_time: float
    num_prefill_tokens: int
    num_decode_tokens: int

    @property
    def prefill_tok_s(self) -> float:
        if self.prefill_time == 0:
            return 0.0
        return self.num_prefill_tokens / self.prefill_time

    @property
    def decode_tok_s(self) -> float:
        if self.decode_time == 0:
            return 0.0
        return self.num_decode_tokens / self.decode_time


def warmup(model: LlamaModel, quantize_kv_cache: bool = False) -> None:
    """Run throwaway forward passes to trigger MLX's lazy-graph compilation
    before any real request is timed or served.

    MLX (like JAX) builds and compiles its computation graph lazily on first
    use for each distinct shape/op combination — the first prefill call and
    the first decode call are different traced shapes, so both are warmed
    here, not just one. This is a real, documented phenomenon (confirmed via
    MLX's own compile docs and GitHub issues, not assumed), and "run a dummy
    forward pass before serving" is the standard, real mitigation for it —
    see CLAUDE.md's Roadmap C research note for the fact-check that ruled out
    the fabricated-sounding "out-of-band memory allocator" framing floated
    for this same problem.

    Call this once right after loading/quantizing a model and before the
    first `generate()` call whose latency actually matters (e.g. before
    starting a server loop, or before benchmarking TTFT).
    """
    cache = make_quantized_cache(len(model.layers)) if quantize_kv_cache else None
    prefill_dummy = mx.array([[1, 2]])
    logits, cache = model(prefill_dummy, cache)
    mx.eval(logits)

    decode_dummy = mx.array([[1]])
    logits, cache = model(decode_dummy, cache)
    mx.eval(logits)


def generate(
    model: LlamaModel,
    prompt_ids: mx.array,
    params: SamplingParams | None = None,
    eos_token_id: int | None = None,
    stream: bool = False,
    on_token: Callable[[int], None] | None = None,
    quantize_kv_cache: bool = False,
) -> GenerationResult:
    if params is None:
        params = SamplingParams()

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]

    B, T = prompt_ids.shape
    generated_ids: list[int] = []

    cache = make_quantized_cache(len(model.layers)) if quantize_kv_cache else None

    t0 = time.perf_counter()
    logits, cache = model(prompt_ids, cache)
    mx.eval(logits)
    t_prefill = time.perf_counter()

    token = sample_token(logits, params, generated_ids)
    mx.eval(token)
    tok_id = token.item()
    generated_ids.append(tok_id)

    if on_token is not None:
        on_token(tok_id)
    elif stream:
        print(tok_id, end=" ", flush=True)

    t_decode_start = time.perf_counter()
    for _ in range(params.max_tokens - 1):
        if eos_token_id is not None and tok_id == eos_token_id:
            break

        token_input = token.reshape(1, 1)
        logits, cache = model(token_input, cache)
        token = sample_token(logits, params, generated_ids)
        mx.eval(token)
        tok_id = token.item()
        generated_ids.append(tok_id)

        if on_token is not None:
            on_token(tok_id)
        elif stream:
            print(tok_id, end=" ", flush=True)

    t_done = time.perf_counter()

    if stream:
        print()

    return GenerationResult(
        token_ids=generated_ids,
        prefill_time=t_prefill - t0,
        decode_time=t_done - t_decode_start,
        num_prefill_tokens=T,
        num_decode_tokens=len(generated_ids) - 1,
    )
