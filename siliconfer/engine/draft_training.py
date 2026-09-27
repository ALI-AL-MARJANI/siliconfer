"""Supervised distillation for FeatureFusionDraftHead.

Teacher-forced on WikiText-2 sequences: the target's hidden states and the
true next tokens are available for every position from one forward pass.
EAGLE-3 trains on 500K+ rollouts with a multi-step objective; this is a
small-scale check of whether the architecture learns at all.
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from siliconfer.model.draft_head import FeatureFusionDraftHead
from siliconfer.model.llama import LlamaModel


def collect_distillation_example(
    target: LlamaModel,
    input_ids: mx.array,
    feature_layers: list[int],
) -> tuple[list[mx.array], mx.array]:
    """Run the target once and return (hidden_states, labels).

    The hidden states are stop-gradient and evaluated before returning: MLX is
    lazy, so without the explicit `mx.eval` every example would stay an
    unevaluated graph of the full target forward pass.
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
    """Train the head's fusion projection and block by supervised distillation.

    The target is frozen, so its hidden states are computed once before the
    first epoch. After every epoch that improves the validation loss the
    parameters are copied; the best copy is restored at the end. With
    `patience` set, training stops after that many epochs without improvement.

    Returns a dict with the per-epoch train/val losses, `best_epoch` (1-indexed)
    and `best_val_loss`.
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
    """Return (draft_head_top1, target_top1) next-token accuracy on `sequences`.

    The target's own accuracy is given as a reference point.
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
