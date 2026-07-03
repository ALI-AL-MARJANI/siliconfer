"""RTN (Round-To-Nearest) quantization: apply fake-quant to all linear projection weights."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.quant.primitives import fake_quantize


def _rtn_weight(w: mx.array, group_size: int, sym: bool) -> mx.array:
    """Apply RTN fake-quant to a single MLX weight tensor.

    Skips the weight if in_features < group_size (e.g. tiny synthetic models).
    """
    in_features = w.shape[-1]
    if in_features < group_size:
        return w

    orig_dtype = w.dtype
    w_np = np.array(w.astype(mx.float32))

    # Pad last dim to a multiple of group_size if needed
    pad = (-in_features) % group_size
    if pad > 0:
        pad_shape = list(w_np.shape)
        pad_shape[-1] = pad
        w_np = np.concatenate([w_np, np.zeros(pad_shape, dtype=np.float32)], axis=-1)

    w_q = fake_quantize(w_np, group_size=group_size, sym=sym)

    if pad > 0:
        w_q = w_q[..., :in_features]

    return mx.array(w_q).astype(orig_dtype)


def apply_rtn(
    model: LlamaModel,
    group_size: int = 128,
    sym: bool = True,
) -> LlamaModel:
    """Apply RTN int4 fake-quant to all attention + MLP linear weights in the model.

    Modifies the model in place and returns it. Embedding weights are intentionally
    skipped (not standard practice to quantize embeddings, and Qwen ties them to lm_head).

    Args:
        model: a loaded LlamaModel.
        group_size: 64 or 128.
        sym: if True use symmetric quant; if False use asymmetric.

    Returns:
        The same model with quantized weights.
    """
    for layer in model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            proj.weight = _rtn_weight(proj.weight, group_size, sym)

        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            proj.weight = _rtn_weight(proj.weight, group_size, sym)

        # Force evaluation layer-by-layer to bound peak memory
        mx.eval(layer.parameters())

    return model
