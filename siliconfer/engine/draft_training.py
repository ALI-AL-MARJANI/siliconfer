"""Supervised distillation training for FeatureFusionDraftHead (EAGLE-3-style).

Real EAGLE-3 trains its draft head on 500K+ target-model rollouts (ShareGPT +
UltraChat-200K) — confirmed via live research before building this, not
assumed (see CLAUDE.md/NOTES.md). This module is an honestly small-scale
version: enough real sequences from WikiText-2 to check whether the
mechanism (multi-layer feature fusion + one small transformer block, reusing
the target's own frozen embedding/LM head) can learn anything useful at all,
not a claim of matching the paper's scale, data mixture, or results.

Training is full-sequence teacher forcing (no autoregressive rollout needed
at train time — the target's real hidden states and real next tokens are
available for every position in one batched forward pass), which is both
simpler and cheaper than the paper's "training-time test" multi-step rollout
simulation; that's a real, named simplification, not something obscured.
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx.utils import tree_map

from siliconfer.model.llama import LlamaModel
from siliconfer.model.draft_head import FeatureFusionDraftHead


def collect_distillation_example(
    target: LlamaModel,
    input_ids: mx.array,
    feature_layers: list[int],
) -> tuple[list[mx.array], mx.array]:
    """Run target once (teacher forcing) and return (hidden_states, labels).

    hidden_states are stop-gradiented — they are frozen distillation targets;
    nothing should ever backprop into the target model through them.

    Explicitly evaluated (`mx.eval`) before returning: MLX builds its
    computation graph lazily, so without this, "precomputing" examples ahead
    of the training loop (see train_draft_head) would not actually compute
    anything — it would just accumulate every example's full 24-layer target
    forward pass as one large unevaluated graph, deferred until the training
    loop happens to touch each one. Confirmed this was a real, not
    theoretical, problem: a version without this eval() ran a 150-example,
    40-epoch training job for many minutes with almost no CPU progress
    (adding maybe 5-10s of CPU time per 5 minutes of wall clock — effectively
    stalled) before being killed; forcing eager evaluation here fixed it.
    """
    _, _, hidden_states = target(input_ids, feature_layers=feature_layers)
    hidden_states = [mx.stop_gradient(h) for h in hidden_states]
    labels = input_ids[:, 1:]
    mx.eval(hidden_states, labels)
    return hidden_states, labels


def _nll_loss(model: FeatureFusionDraftHead, input_ids, hidden_states, labels) -> mx.array:
    logits = model.forward_train(input_ids, hidden_states)
    logits = logits[:, :-1, :].astype(mx.float32)
    logsumexp = mx.logsumexp(logits, axis=-1)
    target_logits = mx.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)
    nll = logsumexp - target_logits
    return mx.mean(nll)


def train_draft_head(
    target: LlamaModel,
    draft_head: FeatureFusionDraftHead,
    train_sequences: list[mx.array],
    val_sequences: list[mx.array],
    feature_layers: list[int],
    lr: float = 1e-3,
    n_epochs: int = 3,
    patience: int | None = None,
    verbose: bool = True,
) -> dict:
    """Train draft_head's `fuse` + `block` + `norm` parameters via supervised
    distillation. The target's embed_tokens/lm_head are shared but excluded
    from the parameter tree (leading-underscore attributes — verified this is
    how MLX's nn.Module actually behaves, not assumed) so they're never
    updated by the optimizer, regardless of what forward_train computes
    through them.

    The target is frozen throughout training, so its hidden states for each
    sequence are identical every epoch — precomputed once here rather than
    re-run through the (much more expensive) full target model on every
    epoch. This is the same class of fix as mixed_precision.py's
    precompute-once optimization earlier this session (there: re-quantizing
    all blocks per Shapley coalition; here: re-running target's forward pass
    per training epoch) — confirmed worth doing before, not assumed
    universally necessary, so worth re-applying deliberately rather than by
    reflex.

    Early stopping / best-checkpoint selection: a first run at this scale
    (150 sequences, 40 fixed epochs, no checkpointing) overfit past epoch 28
    (val_loss rose from ~5.79 there to 6.39 by epoch 40, even as train_loss
    kept dropping) — the reported result used whatever the last epoch left,
    understating what the run's own best point could do. Fixed properly here,
    not by picking a smaller fixed epoch count: after every epoch where
    val_loss improves, snapshot draft_head's parameters (a real deep copy via
    `tree_map(mx.array, ...)`, not a reference to the same live arrays the
    optimizer keeps mutating); at the end, restore the best snapshot via
    `draft_head.update(...)`. If `patience` is set, training also stops early
    once `patience` consecutive epochs pass without a new best val_loss,
    rather than always running the full `n_epochs` regardless of whether
    it's still helping.

    Returns a dict with per-epoch train/val loss history plus
    `best_epoch`/`best_val_loss` (1-indexed; the checkpoint actually left on
    `draft_head` after this function returns).
    """
    draft_head.attach_target_embeddings(target)
    optimizer = optim.AdamW(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(draft_head, _nll_loss)

    precompute_t0 = time.perf_counter()
    train_examples = [collect_distillation_example(target, seq, feature_layers) for seq in train_sequences]
    val_examples = [collect_distillation_example(target, seq, feature_layers) for seq in val_sequences]
    if verbose:
        print(f"  precomputed {len(train_examples)+len(val_examples)} distillation examples "
              f"in {time.perf_counter()-precompute_t0:.1f}s")

    history: dict[str, object] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = None
    epochs_since_best = 0

    for epoch in range(n_epochs):
        epoch_t0 = time.perf_counter()
        epoch_losses = []
        for seq, (hidden_states, labels) in zip(train_sequences, train_examples):
            loss, grads = loss_and_grad_fn(draft_head, seq, hidden_states, labels)
            optimizer.update(draft_head, grads)
            mx.eval(draft_head.parameters(), optimizer.state)
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses))

        val_losses = []
        for seq, (hidden_states, labels) in zip(val_sequences, val_examples):
            loss = _nll_loss(draft_head, seq, hidden_states, labels)
            val_losses.append(float(loss.item()))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_params = tree_map(lambda x: mx.array(x), draft_head.parameters())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        epoch_dt = time.perf_counter() - epoch_t0
        if verbose:
            marker = " *" if improved else ""
            print(f"  epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}{marker}  ({epoch_dt:.1f}s)")

        if patience is not None and epochs_since_best >= patience:
            if verbose:
                print(f"  early stopping: no val_loss improvement for {patience} epochs "
                      f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    if best_params is not None:
        draft_head.update(best_params)
        mx.eval(draft_head.parameters())

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return history


def evaluate_top1_accuracy(
    target: LlamaModel,
    draft_head: FeatureFusionDraftHead,
    sequences: list[mx.array],
    feature_layers: list[int],
) -> tuple[float, float]:
    """Returns (draft_head_top1_acc, target_top1_acc).

    target_top1_acc (the target's own top-1-vs-actual-next-token accuracy) is
    reported alongside as a reference ceiling, not a bar the draft head is
    expected to clear — it's a ~24-layer model conditioned on the full
    sequence, the draft head is one layer conditioned on 3 borrowed feature
    vectors. What matters is whether the draft head does meaningfully better
    than chance / an untrained baseline, which is checked separately in
    tests.
    """
    correct_draft = 0
    correct_target = 0
    total = 0
    for seq in sequences:
        hidden_states, labels = collect_distillation_example(target, seq, feature_layers)

        draft_logits = draft_head.forward_train(seq, hidden_states)[:, :-1, :]
        target_logits_full, _ = target(seq)
        target_logits = target_logits_full[:, :-1, :]

        draft_pred = mx.argmax(draft_logits, axis=-1)
        target_pred = mx.argmax(target_logits, axis=-1)
        mx.eval(draft_pred, target_pred)

        draft_pred_np = np.array(draft_pred)
        target_pred_np = np.array(target_pred)
        labels_np = np.array(labels)

        correct_draft += int((draft_pred_np == labels_np).sum())
        correct_target += int((target_pred_np == labels_np).sum())
        total += labels_np.size

    return correct_draft / total, correct_target / total
