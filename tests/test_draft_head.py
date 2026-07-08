"""Tests for the EAGLE-3-inspired FeatureFusionDraftHead (model/draft_head.py)
and its supervised distillation training (engine/draft_training.py).

Uses a tiny synthetic LlamaModel as the "target" — no model download needed.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel
from siliconfer.model.draft_head import FeatureFusionDraftHead
from siliconfer.engine.draft_training import (
    collect_distillation_example,
    train_draft_head,
    evaluate_top1_accuracy,
)


_TINY_CFG = dict(
    architectures=["LlamaForCausalLM"],
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=32,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    hidden_act="silu",
)
_FEATURE_LAYERS = [2, 4, 7]


def _make_target(seed: int = 0) -> LlamaModel:
    config = ModelConfig(**_TINY_CFG)
    model = LlamaModel(config)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# LlamaModel.feature_layers backward compatibility + correctness
# ---------------------------------------------------------------------------

def test_forward_without_feature_layers_unchanged():
    """Omitting feature_layers must return the original 2-tuple exactly."""
    model = _make_target()
    input_ids = mx.array([[1, 2, 3, 4]])
    result = model(input_ids)
    assert len(result) == 2
    logits, cache = result
    assert logits.shape == (1, 4, _TINY_CFG["vocab_size"])


def test_forward_with_feature_layers_returns_3_tuple():
    model = _make_target()
    input_ids = mx.array([[1, 2, 3, 4]])
    logits, cache, hidden_states = model(input_ids, feature_layers=_FEATURE_LAYERS)
    assert len(hidden_states) == len(_FEATURE_LAYERS)
    for h in hidden_states:
        assert h.shape == (1, 4, _TINY_CFG["hidden_size"])


def test_feature_layers_out_of_range_raises():
    import pytest
    model = _make_target()
    input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="out of range"):
        model(input_ids, feature_layers=[2, 100])


def test_feature_layers_order_matches_request_order():
    """hidden_states must come back in the order feature_layers was given,
    not ascending layer-index order — regression guard since the
    implementation collects them in a dict keyed by layer index."""
    model = _make_target()
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    _, _, hs_reversed = model(input_ids, feature_layers=[7, 4, 2])
    _, _, hs_forward = model(input_ids, feature_layers=[2, 4, 7])
    # Same layers, opposite request order -> arrays should be pairwise equal
    # when reversed, not identical position-by-position.
    np.testing.assert_array_equal(np.array(hs_reversed[0]), np.array(hs_forward[2]))
    np.testing.assert_array_equal(np.array(hs_reversed[2]), np.array(hs_forward[0]))


# ---------------------------------------------------------------------------
# FeatureFusionDraftHead shapes and embedding-sharing
# ---------------------------------------------------------------------------

def test_draft_head_forward_train_shape():
    target = _make_target()
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)
    head.attach_target_embeddings(target)
    mx.eval(head.parameters())

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    hidden_states, labels = collect_distillation_example(target, input_ids, _FEATURE_LAYERS)
    logits = head.forward_train(input_ids, hidden_states)
    assert logits.shape == (1, 5, _TINY_CFG["vocab_size"])
    assert labels.shape == (1, 4)


def test_draft_head_incremental_call_shape():
    target = _make_target()
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)
    head.attach_target_embeddings(target)
    mx.eval(head.parameters())

    input_ids = mx.array([[1]])
    fused = mx.zeros((1, 1, _TINY_CFG["hidden_size"]))
    logits, cache = head(input_ids, cache=None, fused_context=fused)
    assert logits.shape == (1, 1, _TINY_CFG["vocab_size"])

    # Second step: no fused_context (as in real drafting past the anchor),
    # cache carries state forward.
    input_ids_2 = mx.array([[2]])
    logits_2, cache_2 = head(input_ids_2, cache=cache, fused_context=None)
    assert logits_2.shape == (1, 1, _TINY_CFG["vocab_size"])


def test_embed_tokens_and_lm_head_excluded_from_parameter_tree():
    """The shared target embedding/LM head must NOT appear in the draft
    head's own trainable parameter tree (leading-underscore attribute
    convention) — otherwise the optimizer would silently update the
    target's real embedding weights while "training the draft head"."""
    target = _make_target()
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)
    head.attach_target_embeddings(target)

    params = head.parameters()
    assert "_embed_tokens" not in params
    assert "_lm_head_fn" not in params
    assert set(params.keys()) == {"fuse", "block", "norm"}


# ---------------------------------------------------------------------------
# Training: loss decreases, target model is untouched
# ---------------------------------------------------------------------------

def _make_learnable_sequences(vocab_size: int, seq_len: int, n_seqs: int, seed: int) -> list[mx.array]:
    """Deterministic repeating-cycle sequences — fully learnable in principle
    (next token is an exact function of the current one), so a training loop
    that's actually learning anything should drive the loss down measurably."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_seqs):
        start = int(rng.integers(0, vocab_size))
        step = int(rng.integers(1, 5))
        ids = [(start + i * step) % vocab_size for i in range(seq_len)]
        seqs.append(mx.array([ids], dtype=mx.int32))
    return seqs


def test_training_reduces_loss():
    target = _make_target(seed=1)
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)

    train_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=20, seed=2)
    val_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=5, seed=3)

    history = train_draft_head(
        target, head, train_seqs, val_seqs, _FEATURE_LAYERS,
        lr=2e-3, n_epochs=8, verbose=False,
    )
    assert history["train_loss"][-1] < history["train_loss"][0] * 0.9, (
        f"Expected meaningful loss decrease, got {history['train_loss']}"
    )


def test_training_does_not_modify_target_embeddings():
    target = _make_target(seed=4)
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)

    embed_before = np.array(target.embed_tokens.weight)

    train_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=10, seed=5)
    val_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=3, seed=6)
    train_draft_head(target, head, train_seqs, val_seqs, _FEATURE_LAYERS,
                      lr=2e-3, n_epochs=3, verbose=False)

    embed_after = np.array(target.embed_tokens.weight)
    np.testing.assert_array_equal(embed_before, embed_after)


def test_train_draft_head_reports_best_epoch_and_val_loss():
    target = _make_target(seed=9)
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)

    train_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=20, seed=10)
    val_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=5, seed=11)

    history = train_draft_head(target, head, train_seqs, val_seqs, _FEATURE_LAYERS,
                                lr=2e-3, n_epochs=8, verbose=False)
    assert 1 <= history["best_epoch"] <= 8
    assert history["best_val_loss"] == min(history["val_loss"])


def test_train_draft_head_restores_best_checkpoint_not_final_epoch():
    """If val_loss gets worse in later epochs, the parameters left on
    draft_head after training must match the BEST epoch's snapshot, not
    whatever the optimizer left after the final epoch — the real bug found
    during the first real-model run (best val_loss was at epoch 28 of 40,
    but the reported result used epoch 40's overfit weights)."""
    target = _make_target(seed=20)
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)

    # A tiny, noisy val set on a head trained fast enough to plausibly overfit
    # within a handful of epochs, so val_loss is not guaranteed monotonic.
    train_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=6, seed=21)
    val_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=4, seed=22)

    history = train_draft_head(target, head, train_seqs, val_seqs, _FEATURE_LAYERS,
                                lr=5e-3, n_epochs=25, verbose=False)

    # Re-evaluate val_loss on the head as returned (should equal best_val_loss,
    # not whatever the final epoch's val_loss happened to be, if they differ).
    from siliconfer.engine.draft_training import _nll_loss
    val_losses_after = []
    for seq in val_seqs:
        hidden_states, labels = collect_distillation_example(target, seq, _FEATURE_LAYERS)
        val_losses_after.append(float(_nll_loss(head, seq, hidden_states, labels).item()))
    final_val_loss = float(np.mean(val_losses_after))

    assert abs(final_val_loss - history["best_val_loss"]) < 1e-4, (
        f"Restored checkpoint's actual val_loss ({final_val_loss}) should match "
        f"best_val_loss ({history['best_val_loss']}), not the last epoch's "
        f"({history['val_loss'][-1]})"
    )


def test_train_draft_head_early_stopping_triggers():
    """With patience=2 and enough epochs for val_loss to plateau/worsen, the
    run should stop before n_epochs and report a best_epoch well before the
    end — not silently run the full budget every time regardless of whether
    it's still helping."""
    target = _make_target(seed=30)
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)

    train_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=6, seed=31)
    val_seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=4, seed=32)

    history = train_draft_head(target, head, train_seqs, val_seqs, _FEATURE_LAYERS,
                                lr=5e-3, n_epochs=100, patience=2, verbose=False)
    assert len(history["train_loss"]) < 100, "Expected early stopping to cut the run short"
    assert len(history["train_loss"]) >= history["best_epoch"]


def test_evaluate_top1_accuracy_returns_valid_fractions():
    target = _make_target(seed=7)
    config = ModelConfig(**_TINY_CFG)
    head = FeatureFusionDraftHead(config, _FEATURE_LAYERS)
    head.attach_target_embeddings(target)

    seqs = _make_learnable_sequences(_TINY_CFG["vocab_size"], seq_len=16, n_seqs=5, seed=8)
    draft_acc, target_acc = evaluate_top1_accuracy(target, head, seqs, _FEATURE_LAYERS)
    assert 0.0 <= draft_acc <= 1.0
    assert 0.0 <= target_acc <= 1.0
