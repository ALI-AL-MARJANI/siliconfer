"""Perplexity on WikiText-2 or C4.

Non-overlapping windows of seq_len tokens, one forward pass per window (no KV
cache), cross-entropy through a stable logsumexp.

  "wikitext2": the WikiText-2 raw test split.
  "c4":        a seeded sample of the C4 "en" validation split, streamed until
               c4_n_seqs * seq_len tokens are collected.

See docs/eval-protocol.md.
"""

from __future__ import annotations

import hashlib
import math

import mlx.core as mx
import numpy as np

from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel


def load_wikitext2_test_tokens(tokenizer_id: str) -> np.ndarray:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    token_ids = tokenizer.encode(text)
    return np.array(token_ids, dtype=np.int32)


def _load_c4_val_tokens(tokenizer_id: str, n_seqs: int, seq_len: int, seed: int = 0) -> np.ndarray:
    """Stream n_seqs * seq_len + 1 tokens from the C4 "en" validation split.

    `seed` shuffles the stream and therefore selects the documents.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    need = n_seqs * seq_len + 1
    ids: list[int] = []
    for ex in ds:
        ids.extend(tokenizer.encode(ex["text"]))
        if len(ids) >= need:
            break
    return np.array(ids[:need], dtype=np.int32)


def compute_perplexity(
    model: LlamaModel,
    config: ModelConfig,
    tokenizer_id: str,
    seq_len: int = 2048,
    max_tokens: int | None = None,
    dataset: str = "wikitext2",
    c4_n_seqs: int = 256,
    seed: int = 0,
    verbose: bool = True,
    return_info: bool = False,
) -> float | tuple[float, dict]:
    """Compute perplexity on WikiText-2 (default) or a C4 sample.

    Args:
        model: a loaded LlamaModel, quantized or not.
        config: its ModelConfig.
        tokenizer_id: Hugging Face model id for the tokenizer.
        seq_len: window length in tokens.
        max_tokens: cap on WikiText-2 tokens; ignored for C4.
        dataset: "wikitext2" or "c4".
        c4_n_seqs: number of windows drawn from C4.
        seed: shuffle seed that selects the C4 documents. It defines the test
            set, so it must be the same for every run being compared.
        verbose: print progress per window.
        return_info: also return a dict with the window and token counts and a
            SHA-256 of the scored token ids.

    Returns:
        The perplexity, or (perplexity, info) with return_info.
    """
    if dataset == "wikitext2":
        token_ids = load_wikitext2_test_tokens(tokenizer_id)
        if max_tokens is not None:
            token_ids = token_ids[: max_tokens + 1]
    elif dataset == "c4":
        token_ids = _load_c4_val_tokens(tokenizer_id, c4_n_seqs, seq_len, seed=seed)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose 'wikitext2' or 'c4'.")

    total_tokens = len(token_ids)
    n_chunks = (total_tokens - 1) // seq_len
    if n_chunks == 0:
        raise ValueError(
            f"Not enough tokens ({total_tokens}) for even one chunk of seq_len={seq_len}. "
            "Lower seq_len or raise max_tokens."
        )

    all_nlls: list[float] = []

    for i in range(n_chunks):
        start = i * seq_len
        end = start + seq_len + 1            # +1 so we have seq_len input → seq_len targets
        chunk = token_ids[start:end]         # (seq_len + 1,)
        if len(chunk) < seq_len + 1:
            break

        chunk_input = mx.array(chunk[:-1][None, :])   # (1, seq_len)
        logits, _ = model(chunk_input)
        mx.eval(logits)

        logits_np = np.array(logits[0], dtype=np.float32)   # (seq_len, vocab)
        targets = chunk[1:].astype(np.int64)                # (seq_len,)

        # Numerically stable cross-entropy: NLL = log_sum_exp(logits) - logit[target]
        max_l = logits_np.max(axis=-1, keepdims=True)        # (T, 1)
        log_sum_exp = (
            np.log(np.exp(logits_np - max_l).sum(axis=-1)) + max_l.squeeze(-1)
        )                                                     # (T,)
        target_logit = logits_np[np.arange(len(targets)), targets]  # (T,)
        nll = float((log_sum_exp - target_logit).mean())

        all_nlls.append(nll)

        if verbose:
            running_ppl = math.exp(np.mean(all_nlls))
            print(
                f"  chunk {i+1}/{n_chunks}  nll={nll:.4f}  "
                f"running PPL={running_ppl:.2f}",
                flush=True,
            )

    mean_nll = float(np.mean(all_nlls))
    ppl = math.exp(mean_nll)

    if verbose:
        label = {"wikitext2": "WikiText-2", "c4": "C4"}[dataset]
        print(f"\n{label} PPL = {ppl:.2f}  (mean NLL={mean_nll:.4f}, {n_chunks} chunks)")

    if return_info:
        evaluated = np.asarray(token_ids[: n_chunks * seq_len + 1], dtype=np.int64)
        info = {
            "n_chunks": n_chunks,
            "n_tokens_scored": n_chunks * seq_len,
            "tokens_sha256": hashlib.sha256(evaluated.tobytes()).hexdigest(),
        }
        return ppl, info
    return ppl
