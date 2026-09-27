"""Mixed-precision quantization: per-block bit-widths chosen by Shapley sensitivity.

Inspired by CoopQ (Zhao et al., arXiv:2509.15455), which frames bit-width
assignment as a cooperative game among layers. No reference code is public;
the value function and the assignment rule here are this project's own.

1. `shapley_layer_sensitivity`: permutation-sampling Monte Carlo estimate of
   each block's Shapley value under an arbitrary `value_fn(coalition)`
   (Castro et al. 2009). Checked against closed-form values in
   tests/test_mixed_precision.py.

2. `assign_bitwidths`: demote the least sensitive blocks until a memory
   budget is met. Every block has the same parameter count in this
   architecture, so the knapsack reduces to a top-k selection and the greedy
   rule is exact. That stops holding at per-projection granularity, where
   sizes differ.

Evaluated as fake-quant only: there is no 2-bit or 3-bit packed kernel.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# 1. Generic permutation-sampling Shapley value estimator
# ---------------------------------------------------------------------------

def shapley_layer_sensitivity(
    value_fn: Callable[[frozenset[int]], float],
    n_layers: int,
    n_permutations: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Estimate each layer's Shapley value under `value_fn` by permutation sampling.

    `value_fn(S)` returns a quality score when the layers in S are in their
    better state (e.g. 4-bit) and all others in their worse state (e.g. 2-bit).
    For a random order, the marginal contribution of layer i is
    `value_fn(S + i) - value_fn(S)`; its average over orders converges to the
    Shapley value.

    Args:
        value_fn: coalition -> quality score. Should be deterministic: results
            are cached per coalition.
        n_layers: number of players (e.g. transformer blocks).
        n_permutations: number of random orders. Value functions with strong
            interactions need more than additive ones.
        seed: RNG seed for the permutation sampling.

    Returns:
        sensitivity: float64 array [n_layers]. Higher means the layer should be
            kept at the higher bit-width.
    """
    if n_layers <= 0:
        return np.zeros(0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    shapley = np.zeros(n_layers, dtype=np.float64)
    cache: dict[frozenset[int], float] = {}

    def cached_value(S: frozenset[int]) -> float:
        if S not in cache:
            cache[S] = value_fn(S)
        return cache[S]

    empty_value = cached_value(frozenset())

    for _ in range(n_permutations):
        perm = rng.permutation(n_layers)
        S: frozenset[int] = frozenset()
        prev_value = empty_value
        for i in perm:
            i = int(i)
            S = S | {i}
            v = cached_value(S)
            shapley[i] += v - prev_value
            prev_value = v

    shapley /= n_permutations
    return shapley


# ---------------------------------------------------------------------------
# 2. Bit-width assignment under a memory budget
# ---------------------------------------------------------------------------

def assign_bitwidths(
    sensitivity: np.ndarray,
    bytes_per_layer_at_high_bits: np.ndarray,
    memory_budget_bytes: float,
    high_bits: int = 4,
    low_bits: int = 2,
) -> np.ndarray:
    """Assign each layer `high_bits` or `low_bits` to fit a memory budget.

    Layers are demoted in ascending order of `sensitivity` until the total
    packed footprint fits `memory_budget_bytes`. Exact for uniform layer sizes
    (see the module docstring).

    Args:
        sensitivity: float array [n_layers]; higher = keep at `high_bits`.
        bytes_per_layer_at_high_bits: float array [n_layers], footprint of each
            layer at `high_bits`.
        memory_budget_bytes: total footprint ceiling.
        high_bits: bit-width for sensitive layers (default 4).
        low_bits: bit-width for demoted layers (default 2).

    Returns:
        bits: int array [n_layers], each entry `high_bits` or `low_bits`.
    """
    n = len(sensitivity)
    bytes_at_high = np.asarray(bytes_per_layer_at_high_bits, dtype=np.float64)
    bytes_at_low = bytes_at_high * (low_bits / high_bits)
    savings = bytes_at_high - bytes_at_low

    bits = np.full(n, high_bits, dtype=np.int32)
    bytes_used = float(bytes_at_high.sum())
    if bytes_used <= memory_budget_bytes:
        return bits

    order = np.argsort(sensitivity)  # ascending: least sensitive demoted first
    for i in order:
        if bytes_used <= memory_budget_bytes:
            break
        bits[i] = low_bits
        bytes_used -= savings[i]

    return bits


# ---------------------------------------------------------------------------
# 3. LLM-specific value function + end-to-end driver
# ---------------------------------------------------------------------------

def make_block_nll_value_fn(
    model,
    calib_input_ids,
    group_size: int = 128,
    high_bits: int = 4,
    low_bits: int = 2,
) -> Callable[[frozenset[int]], float]:
    """Build a Shapley `value_fn`: negative mean cross-entropy of `model` on
    `calib_input_ids` when the blocks in the coalition are quantized to
    `high_bits` and every other block to `low_bits`.

    Both versions of every block's weights are quantized once up front; a
    coalition evaluation only selects which of the two each block uses.

    Args:
        model: a loaded fp16 LlamaModel. Its weights are overwritten on every
            call and left at the last coalition evaluated.
        calib_input_ids: mx.array [n_seqs, seq_len]. Keep it small: one forward
            pass runs per distinct coalition.
        group_size: quantization group size.
        high_bits: bit-width for coalition members.
        low_bits: bit-width for everyone else.

    Returns:
        value_fn suitable for `shapley_layer_sensitivity`.
    """
    import mlx.core as mx

    from siliconfer.quant.hqq import _DEFAULT_K_GRID, _hqq_weight

    proj_names = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]

    high_weights: list[list] = []
    low_weights: list[list] = []
    for layer in model.layers:
        row_high, row_low = [], []
        for parent_name, proj_name in proj_names:
            parent = getattr(layer, parent_name)
            w_orig = getattr(parent, proj_name).weight
            row_high.append(_hqq_weight(w_orig, group_size, 0.7, _DEFAULT_K_GRID, bits=high_bits))
            row_low.append(_hqq_weight(w_orig, group_size, 0.7, _DEFAULT_K_GRID, bits=low_bits))
        high_weights.append(row_high)
        low_weights.append(row_low)

    def value_fn(coalition: frozenset[int]) -> float:
        for layer_idx, layer in enumerate(model.layers):
            weights = high_weights[layer_idx] if layer_idx in coalition else low_weights[layer_idx]
            for (parent_name, proj_name), w in zip(proj_names, weights):
                parent = getattr(layer, parent_name)
                getattr(parent, proj_name).weight = w
        mx.eval(model.parameters())

        logits, _ = model(calib_input_ids)
        log_probs = logits[:, :-1, :].astype(mx.float32)
        targets = calib_input_ids[:, 1:]
        logsumexp = mx.logsumexp(log_probs, axis=-1)
        target_logits = mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        nll = (logsumexp - target_logits)
        mean_nll = float(mx.mean(nll))

        return -mean_nll  # higher (less negative) = lower loss = better

    return value_fn


def apply_mixed_precision(
    model,
    bits_per_block: list[int] | np.ndarray,
    group_size: int = 128,
    high_bits: int = 4,
    low_bits: int = 2,
    verbose: bool = True,
):
    """Quantize each transformer block to the bit-width in `bits_per_block`.

    Both tiers use `hqq_quantize_weight(..., bits=...)`.

    Args:
        model: a loaded LlamaModel, modified in place.
        bits_per_block: sequence of length len(model.layers), each entry
            `high_bits` or `low_bits`.
        group_size: quantization group size.
        high_bits: bit-width for high-precision blocks (default 4).
        low_bits: bit-width for demoted blocks (default 2).
        verbose: print per-layer progress.

    Returns:
        The same model with mixed-precision weights.
    """
    import mlx.core as mx

    from siliconfer.quant.hqq import _DEFAULT_K_GRID, _hqq_weight

    if len(bits_per_block) != len(model.layers):
        raise ValueError(
            f"bits_per_block has {len(bits_per_block)} entries, "
            f"model has {len(model.layers)} layers"
        )

    proj_names = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]

    n_layers = len(model.layers)
    for i, (layer, bits) in enumerate(zip(model.layers, bits_per_block)):
        bits = int(bits)
        if bits not in (high_bits, low_bits):
            raise ValueError(f"bits_per_block[{i}]={bits} must be {high_bits} or {low_bits}")
        if verbose:
            print(f"  mixed-precision layer {i+1}/{n_layers} (bits={bits}) ...", end=" ", flush=True)

        for parent_name, proj_name in proj_names:
            parent = getattr(layer, parent_name)
            lin = getattr(parent, proj_name)
            lin.weight = _hqq_weight(lin.weight, group_size, 0.7, _DEFAULT_K_GRID, bits=bits)

        mx.eval(layer.parameters())
        if verbose:
            print("done")

    return model
