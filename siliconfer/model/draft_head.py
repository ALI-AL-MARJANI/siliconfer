"""Draft head for speculative decoding, after EAGLE-3 (arXiv:2503.01840).

One transformer block at the target's hidden width, fed with the target's
hidden states from a few depths (`feature_layers`) fused by a linear
projection. The target's embedding table and output head are reused and
kept frozen, so only the fusion projection and the block are trained.

This is a small-scale version: EAGLE-3 trains on 500K+ examples with a
multi-step rollout objective; here training is teacher-forced on a few
hundred sequences (engine/draft_training.py).

Two forward modes:

- `forward_train(input_ids, target_hidden_states)`: full sequence,
  teacher-forced, used for training.
- `__call__(input_ids, cache, fused_context)`: incremental. `fused_context`
  carries the target's features for the first token of a round; later
  tokens rely on the block's own KV cache.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.config import ModelConfig
from siliconfer.model.kv_cache import QuantizedKVCache
from siliconfer.model.layers import RMSNorm, TransformerBlock, _compute_rope_freqs


class FeatureFusionDraftHead(nn.Module):
    def __init__(self, target_config: ModelConfig, feature_layers: list[int]):
        super().__init__()
        # Leading underscore: MLX walks any list attribute as parameters,
        # but skips names starting with an underscore.
        self._feature_layers = list(feature_layers)
        self.hidden_size = target_config.hidden_size

        self.fuse = nn.Linear(len(feature_layers) * target_config.hidden_size, target_config.hidden_size)
        rope_freqs = _compute_rope_freqs(target_config)
        self.block = TransformerBlock(target_config, rope_freqs)
        self.norm = RMSNorm(target_config.hidden_size, target_config.rms_norm_eps)

        # Set by attach_target_embeddings() — frozen, shared with the target
        # model, never trained (matches real EAGLE's design: only the fusion
        # layer + one transformer block are new parameters).
        self._embed_tokens = None
        self._lm_head_fn = None

    def attach_target_embeddings(self, target) -> None:
        """Share the target's embedding table and output head.

        Must be called before any forward pass. They are stored under
        leading-underscore attributes, which MLX leaves out of the parameter tree,
        so the optimizer never updates the target's weights.
        """
        self._embed_tokens = target.embed_tokens
        if target.lm_head is not None:
            self._lm_head_fn = target.lm_head
        else:
            self._lm_head_fn = target.embed_tokens.as_linear

    def fuse_features(self, hidden_states: list[mx.array]) -> mx.array:
        """hidden_states: list of [B, T, hidden] (one per feature_layers entry,
        same order) -> fused [B, T, hidden]."""
        concatenated = mx.concatenate(hidden_states, axis=-1)
        return self.fuse(concatenated)

    def forward_train(
        self,
        input_ids: mx.array,
        target_hidden_states: list[mx.array],
    ) -> mx.array:
        """Full-sequence teacher-forced forward pass.

        Args:
            input_ids: [B, T] token ids.
            target_hidden_states: list of [B, T, hidden] arrays, the target's
                hidden states at `feature_layers` for the same input.

        Returns:
            logits [B, T, vocab]; logits[:, t] predicts input_ids[:, t+1].
        """
        assert self._embed_tokens is not None, "call attach_target_embeddings() first"
        tok_embeds = self._embed_tokens(input_ids)
        fused = self.fuse_features(target_hidden_states)
        x = tok_embeds + fused
        x, _ = self.block(x, cache=None)
        x = self.norm(x)
        return self._lm_head_fn(x)

    def __call__(
        self,
        input_ids: mx.array,
        cache: tuple[mx.array, mx.array] | QuantizedKVCache | None = None,
        fused_context: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | QuantizedKVCache]:
        """Incremental forward pass for drafting.

        Args:
            input_ids: [B, T] token ids for this step (T=1 when drafting).
            cache: the block's KV cache, reset at the start of each round.
            fused_context: [B, T, hidden] target features, given only for the
                first token of a round.

        Returns:
            (logits [B, T, vocab], new_cache)
        """
        assert self._embed_tokens is not None, "call attach_target_embeddings() first"
        x = self._embed_tokens(input_ids)
        if fused_context is not None:
            x = x + fused_context
        x, new_cache = self.block(x, cache)
        x = self.norm(x)
        logits = self._lm_head_fn(x)
        return logits, new_cache
