"""WikiText-2 perplexity evaluation.

Uses non-overlapping chunks of seq_len tokens. Each chunk is a single forward
pass (no KV cache); cross-entropy is computed with a numerically stable logsumexp.
"""

from __future__ import annotations

import math

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.model.config import ModelConfig


def compute_perplexity(
    model: LlamaModel,
    config: ModelConfig,
    tokenizer_id: str,
    seq_len: int = 2048,
    max_tokens: int | None = None,
    verbose: bool = True,
) -> float:
    """Compute WikiText-2 test perplexity.

    Args:
        model: a loaded (and optionally quantized) LlamaModel.
        config: the model's ModelConfig.
        tokenizer_id: HuggingFace model ID for the tokenizer (e.g. "Qwen/Qwen2.5-0.5B").
        seq_len: chunk length in tokens. Standard is 2048.
        max_tokens: if set, stop after this many tokens (useful for quick estimates).
        verbose: print progress per chunk.

    Returns:
        Perplexity (float).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load WikiText-2 test split and join all text
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    token_ids = tokenizer.encode(text)
    token_ids = np.array(token_ids, dtype=np.int32)

    if max_tokens is not None:
        token_ids = token_ids[: max_tokens + 1]

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
        print(f"\nWikiText-2 PPL = {ppl:.2f}  (mean NLL={mean_nll:.4f}, {n_chunks} chunks)")

    return ppl
