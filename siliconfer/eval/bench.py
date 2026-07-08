"""Throughput and memory benchmarks for siliconfer models.

Measures:
  - Prefill tok/s and decode tok/s for a given model
  - Weight memory footprint (MLX parameters + Q4Linear packed arrays)

Used by scripts/run_benchmarks.py to produce the Phase 7 results table.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.engine.generate import generate, SamplingParams


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def measure_throughput(
    model: LlamaModel,
    prompt_ids: mx.array,
    n_decode_tokens: int = 100,
    n_runs: int = 3,
) -> dict[str, float]:
    """Measure prefill and decode throughput (tok/s).

    Runs generate() n_runs times and returns the median decode tok/s to reduce
    JIT warm-up noise. Prefill is measured from the first run (one-shot).

    Args:
        model:            The model to benchmark (fp16 or Q4Linear).
        prompt_ids:       Token IDs, shape [1, T].
        n_decode_tokens:  Number of decode tokens to generate per run.
        n_runs:           Number of independent runs; median is returned.

    Returns:
        dict with keys: prefill_tok_s, decode_tok_s, prefill_time_s, decode_time_s.
    """
    params = SamplingParams(temperature=0.0, max_tokens=n_decode_tokens)

    # Warmup: one run to trigger JIT / Metal compilation
    _ = generate(model, prompt_ids, params=params)

    prefill_times: list[float] = []
    decode_tok_s_list: list[float] = []

    for _ in range(n_runs):
        result = generate(model, prompt_ids, params=params)
        prefill_times.append(result.prefill_time)
        decode_tok_s_list.append(result.decode_tok_s)

    median_decode_tok_s  = float(np.median(decode_tok_s_list))
    median_prefill_time  = float(np.median(prefill_times))
    n_prompt = prompt_ids.shape[1]

    return {
        "prefill_tok_s":   n_prompt / median_prefill_time if median_prefill_time > 0 else 0.0,
        "decode_tok_s":    median_decode_tok_s,
        "prefill_time_s":  median_prefill_time,
        "decode_time_s":   n_decode_tokens / median_decode_tok_s if median_decode_tok_s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------

def measure_memory(model: LlamaModel) -> dict[str, float]:
    """Measure model weight memory in MB.

    Returns:
        dict with keys: mlx_param_mb, q4_packed_mb, total_mb.
    """
    from siliconfer.model.q4_linear import Q4Linear

    # MLX-tracked parameters (embed, norms, fp16 attn/mlp in non-quantized models)
    mlx_bytes = 0
    def _visit(obj):
        if isinstance(obj, mx.array):
            n = 1
            for d in obj.shape:
                n *= d
            return n * obj.dtype.size
        if isinstance(obj, dict):
            return sum(_visit(v) for v in obj.values())
        if isinstance(obj, list):
            return sum(_visit(v) for v in obj)
        return 0
    mlx_bytes = _visit(model.parameters())

    # Q4Linear packed arrays (numpy, not tracked by MLX)
    q4_bytes = 0
    for layer in model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"):
                lin = getattr(parent, name, None)
                if isinstance(lin, Q4Linear):
                    q4_bytes += lin._packed.nbytes + lin._scales.nbytes
                    if lin._zeros is not None:
                        q4_bytes += lin._zeros.nbytes

    return {
        "mlx_param_mb": mlx_bytes / 1e6,
        "q4_packed_mb": q4_bytes / 1e6,
        "total_mb":     (mlx_bytes + q4_bytes) / 1e6,
    }


# ---------------------------------------------------------------------------
# KV-cache memory (Phase 9b)
# ---------------------------------------------------------------------------

def measure_kv_cache_memory(config, seq_len: int, batch: int = 1) -> dict[str, float]:
    """Analytical KV-cache memory footprint at a given context length.

    No generation is run — this is a closed-form size calculation (mirrors
    how weight-quantization group overhead is accounted for in NOTES.md),
    since the cache's byte count is a deterministic function of the config
    and sequence length, not something that needs to be measured empirically.

    Args:
        config: ModelConfig (needs num_hidden_layers, num_key_value_heads, head_dim).
        seq_len: number of cached KV positions.
        batch: batch size.

    Returns:
        dict with keys: fp16_mb, int8_mb, compression.
    """
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    n_values = 2 * n_layers * batch * n_kv_heads * seq_len * head_dim   # k + v

    fp16_bytes = n_values * 2
    # int8: 1 byte/value + one float32 scale per (batch, head, token) vector
    n_vectors = 2 * n_layers * batch * n_kv_heads * seq_len
    int8_bytes = n_values * 1 + n_vectors * 4

    return {
        "fp16_mb": fp16_bytes / 1e6,
        "int8_mb": int8_bytes / 1e6,
        "compression": fp16_bytes / int8_bytes,
    }
