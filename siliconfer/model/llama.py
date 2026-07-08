"""Full Llama/Qwen-style decoder model."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.config import ModelConfig
from siliconfer.model.layers import (
    RMSNorm,
    TransformerBlock,
    _compute_rope_freqs,
)
from siliconfer.model.kv_cache import QuantizedKVCache
from siliconfer.engine.loader import load_model, weight_summary


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rope_freqs = _compute_rope_freqs(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, self.rope_freqs)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[tuple[mx.array, mx.array] | QuantizedKVCache] | None = None,
        feature_layers: list[int] | None = None,
    ) -> (
        tuple[mx.array, list[tuple[mx.array, mx.array] | QuantizedKVCache]]
        | tuple[mx.array, list[tuple[mx.array, mx.array] | QuantizedKVCache], list[mx.array]]
    ):
        """Forward pass. Returns (logits, cache) unless `feature_layers` is
        given, in which case it returns (logits, cache, hidden_states) where
        hidden_states[i] is this block's output for feature_layers[i] — added
        for the EAGLE-3-style draft head (model/draft_head.py), which fuses
        hidden states from a few chosen depths rather than only the final
        layer. Backward compatible: omitting feature_layers keeps the
        original 2-tuple return exactly as before.
        """
        x = self.embed_tokens(input_ids)

        new_cache = []
        by_layer: dict[int, mx.array] = {}
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, kv = layer(x, layer_cache)
            new_cache.append(kv)
            if feature_layers is not None and i in feature_layers:
                by_layer[i] = x

        x = self.norm(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = self.embed_tokens.as_linear(x)

        if feature_layers is not None:
            missing = [l for l in feature_layers if l not in by_layer]
            if missing:
                raise ValueError(
                    f"feature_layers={missing} out of range for a "
                    f"{len(self.layers)}-layer model."
                )
            hidden_states = [by_layer[l] for l in feature_layers]
            return logits, new_cache, hidden_states

        return logits, new_cache

    @staticmethod
    def from_pretrained(
        model_dir: str | Path,
        dtype: mx.Dtype = mx.float32,
    ) -> tuple[LlamaModel, ModelConfig]:
        config, weights = load_model(model_dir)
        model = LlamaModel(config)
        _load_weights_into_model(model, weights, config, dtype)
        return model, config


def _load_weights_into_model(
    model: LlamaModel,
    weights: dict[str, mx.array],
    config: ModelConfig,
    dtype: mx.Dtype = mx.float32,
) -> None:
    def cast(w: mx.array) -> mx.array:
        return w.astype(dtype)

    model.embed_tokens.weight = cast(weights["model.embed_tokens.weight"])

    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        layer = model.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp

        attn.q_proj.weight = cast(weights[f"{prefix}.self_attn.q_proj.weight"])
        attn.k_proj.weight = cast(weights[f"{prefix}.self_attn.k_proj.weight"])
        attn.v_proj.weight = cast(weights[f"{prefix}.self_attn.v_proj.weight"])
        attn.o_proj.weight = cast(weights[f"{prefix}.self_attn.o_proj.weight"])

        if config.attention_bias:
            attn.q_proj.bias = cast(weights[f"{prefix}.self_attn.q_proj.bias"])
            attn.k_proj.bias = cast(weights[f"{prefix}.self_attn.k_proj.bias"])
            attn.v_proj.bias = cast(weights[f"{prefix}.self_attn.v_proj.bias"])

        mlp.gate_proj.weight = cast(weights[f"{prefix}.mlp.gate_proj.weight"])
        mlp.up_proj.weight = cast(weights[f"{prefix}.mlp.up_proj.weight"])
        mlp.down_proj.weight = cast(weights[f"{prefix}.mlp.down_proj.weight"])

        layer.input_layernorm.weight = cast(weights[f"{prefix}.input_layernorm.weight"])
        layer.post_attention_layernorm.weight = cast(weights[f"{prefix}.post_attention_layernorm.weight"])

    model.norm.weight = cast(weights["model.norm.weight"])

    if model.lm_head is not None and "lm_head.weight" in weights:
        model.lm_head.weight = cast(weights["lm_head.weight"])
