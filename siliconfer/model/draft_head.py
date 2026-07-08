"""EAGLE-3-inspired scaled-down draft head for speculative decoding.

Real EAGLE-3 (arXiv:2503.01840, fact-checked live before building this — see
CLAUDE.md/NOTES.md) abandons EAGLE/EAGLE-2's feature-*prediction* objective
for direct token prediction, using multi-layer feature fusion (hidden states
from a few different target depths, not just the last layer) as context for
a small trained decoder. It is NOT training-free — the paper trains on
500K+ distilled examples. This is a scaled-down but structurally faithful
version, not a training-free trick:

- Fuses hidden states from `feature_layers` (a few target depths — early,
  middle, late) via one linear projection back to hidden_size.
- Runs the fused features through ONE `TransformerBlock` at the target's own
  hidden width — this mirrors the real EAGLE draft model's own architecture
  (embedding + one decoder layer + LM head), not a simplification invented
  for this project.
- Reuses the TARGET's embedding table and LM head (tied, frozen, untrained)
  instead of training a new vocabulary projection — again matching EAGLE's
  actual design, and it sharply cuts down how much needs to be learned: only
  the fusion projection + one transformer block are new parameters.

Two forward modes:

- `forward_train(input_ids, target_hidden_states)`: full-sequence, teacher-
  forced (used only for training — see engine/draft_training.py). Predicts
  next-token logits at every position from real target features, no
  autoregressive KV cache involved.
- `__call__(input_ids, fused_context, cache)`: single/few-token incremental
  mode for actual drafting — `fused_context` (from the target's most recent
  verification pass, refreshed once per speculative round) seeds the first
  step; the block's own KV cache carries state across subsequent draft steps
  within the round, same as EAGLE's real design (only the round's anchor
  token gets genuine target features; later draft-round tokens condition on
  the draft head's own evolving hidden state, since the target hasn't seen
  them yet).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from siliconfer.model.config import ModelConfig
from siliconfer.model.layers import RMSNorm, TransformerBlock, _compute_rope_freqs
from siliconfer.model.kv_cache import QuantizedKVCache


class FeatureFusionDraftHead(nn.Module):
    def __init__(self, target_config: ModelConfig, feature_layers: list[int]):
        super().__init__()
        # Leading underscore: keeps this plain-int-list attribute out of
        # MLX's parameter tree walk (verified empirically — see
        # test_embed_tokens_and_lm_head_excluded_from_parameter_tree, which
        # first caught MLX including *any* list attribute, not just
        # mx.array/nn.Module ones, before this was renamed).
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
        """Share the target model's embedding table and LM head. Must be
        called before any forward pass. Not stored as MLX submodules (so
        `mlx.nn.value_and_grad` never computes gradients for them, which
        would silently try to fine-tune the target's embeddings)."""
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
        """Full-sequence teacher-forced forward for training.

        Args:
            input_ids: [B, T] token ids (the real sequence — teacher forcing,
                not the draft head's own past predictions).
            target_hidden_states: list of [B, T, hidden] arrays, the target
                model's real hidden states at self.feature_layers for this
                same input_ids (computed once by the target, reused here).

        Returns:
            logits: [B, T, vocab] — logits[:, t, :] predicts input_ids[:, t+1].
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
        """Incremental forward for actual drafting.

        Args:
            input_ids: [B, T] token ids for this step (T=1 for normal
                autoregressive drafting).
            cache: the draft head's own single-block KV cache (reset at the
                start of each speculative round — see engine/speculative.py's
                self-drafting mode).
            fused_context: [B, T, hidden] — required only on the first call of
                a round (the anchor token), where it carries the target's
                real multi-layer features; omitted (None) for subsequent
                draft steps within the same round, since the target hasn't
                seen those tokens yet and the draft head's own hidden state
                (carried via `cache`) is all there is to condition on.

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
