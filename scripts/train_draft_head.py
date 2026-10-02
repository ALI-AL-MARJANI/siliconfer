"""Train the feature-fusion draft head and report its next-token accuracy.

The draft head (model/draft_head.py) is one transformer block fed with the
target model's hidden states from three depths, reusing the target's frozen
embedding and output head. It is trained by supervised distillation on
WikiText-2 *train* sequences; early stopping uses a second sample of train
sequences; the reported top-1 accuracy is measured on WikiText-2 *test*
sequences the training never sees.

One run per --seeds entry (seed controls the head's initialisation and the
training sample). Reports the draft head's top-1 next-token accuracy per seed
and mean ± std, next to the target model's own top-1 accuracy on the same
sequences.

    python scripts/train_draft_head.py

Writes results/claims/draft_head.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.engine.draft_training import evaluate_top1_accuracy, train_draft_head
from siliconfer.eval.env_info import collect_env_info
from siliconfer.eval.perplexity import load_wikitext2_test_tokens
from siliconfer.model.draft_head import FeatureFusionDraftHead
from siliconfer.model.llama import LlamaModel
from siliconfer.quant.calibration import load_calibration_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--feature_layers", default="6,12,23")
    parser.add_argument("--n_train", type=int, default=250)
    parser.add_argument("--n_val", type=int, default=30)
    parser.add_argument("--n_test", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default="results/claims/draft_head.json")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    )
    target, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float32)
    mx.eval(target.parameters())
    feature_layers = [int(v) for v in args.feature_layers.split(",")]

    test_tokens = load_wikitext2_test_tokens(args.model_id)
    test_seqs = [mx.array(test_tokens[i * args.seq_len:(i + 1) * args.seq_len][None, :])
                 for i in range(args.n_test)]

    runs = []
    for seed in (int(v) for v in args.seeds.split(",")):
        t0 = time.time()
        train_seqs = load_calibration_sequences(args.model_id, args.n_train, args.seq_len, seed=seed)
        val_seqs = load_calibration_sequences(args.model_id, args.n_val, args.seq_len,
                                              seed=10_000 + seed)
        mx.random.seed(seed)
        head = FeatureFusionDraftHead(config, feature_layers)
        head.attach_target_embeddings(target)
        mx.eval(head.parameters())

        untrained_acc, target_acc = evaluate_top1_accuracy(target, head, test_seqs, feature_layers)
        history = train_draft_head(target, head, train_seqs, val_seqs, feature_layers,
                                   lr=args.lr, n_epochs=args.max_epochs, patience=args.patience,
                                   verbose=True)
        acc, _ = evaluate_top1_accuracy(target, head, test_seqs, feature_layers)
        runs.append({
            "seed": seed,
            "draft_top1_test": acc,
            "draft_top1_test_untrained": untrained_acc,
            "target_top1_test": target_acc,
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "epochs_run": len(history["train_loss"]),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[train_draft_head] seed {seed}: draft top-1 {acc:.4f} "
              f"(untrained {untrained_acc:.4f}, target {target_acc:.4f}), "
              f"best epoch {history['best_epoch']}")

    accs = np.array([r["draft_top1_test"] for r in runs])
    out = {
        "model_id": args.model_id,
        "config": vars(args),
        "n_test_tokens": args.n_test * (args.seq_len - 1),
        "draft_top1_test_mean": float(accs.mean()),
        "draft_top1_test_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "target_top1_test": runs[0]["target_top1_test"],
        "runs": runs,
        "env": collect_env_info(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[train_draft_head] top-1 {accs.mean():.4f} ± {out['draft_top1_test_std']:.4f} "
          f"(target {out['target_top1_test']:.4f}); wrote {out_path}")


if __name__ == "__main__":
    main()
