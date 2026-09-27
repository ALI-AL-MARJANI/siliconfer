"""Tests for eval/perplexity.py's test-set selection and fingerprinting.

The C4 test sample must depend only on its own seed, never on the
calibration seed, so that every method is scored on the same tokens. The C4
loader is monkeypatched: no download needed.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from siliconfer.eval import perplexity
from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel

_TINY_CFG = dict(
    architectures=["LlamaForCausalLM"],
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=32,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    hidden_act="silu",
)


def _tiny_model():
    config = ModelConfig(**_TINY_CFG)
    model = LlamaModel(config)
    mx.eval(model.parameters())
    return model, config


def _fake_c4(seen_seeds):
    def load(tokenizer_id, n_seqs, seq_len, seed=0):
        seen_seeds.append(seed)
        rng = np.random.default_rng(seed)
        return rng.integers(0, 32, size=n_seqs * seq_len + 1)
    return load


def test_c4_seed_selects_test_set_and_is_fingerprinted(monkeypatch):
    seen = []
    monkeypatch.setattr(perplexity, "_load_c4_val_tokens", _fake_c4(seen))
    model, config = _tiny_model()

    kw = dict(seq_len=16, dataset="c4", c4_n_seqs=2, verbose=False, return_info=True)
    ppl_a, info_a = perplexity.compute_perplexity(model, config, "tiny", seed=0, **kw)
    ppl_b, info_b = perplexity.compute_perplexity(model, config, "tiny", seed=0, **kw)
    ppl_c, info_c = perplexity.compute_perplexity(model, config, "tiny", seed=1, **kw)

    assert seen == [0, 0, 1]
    assert info_a["tokens_sha256"] == info_b["tokens_sha256"]
    assert info_a["tokens_sha256"] != info_c["tokens_sha256"]
    assert info_a["n_chunks"] == 2 and info_a["n_tokens_scored"] == 32
    assert ppl_a == ppl_b


def test_return_info_off_returns_plain_float(monkeypatch):
    monkeypatch.setattr(perplexity, "_load_c4_val_tokens", _fake_c4([]))
    model, config = _tiny_model()
    ppl = perplexity.compute_perplexity(
        model, config, "tiny", seq_len=16, dataset="c4", c4_n_seqs=1, verbose=False
    )
    assert isinstance(ppl, float)


def test_eval_ppl_does_not_tie_c4_test_set_to_calib_seed():
    """eval_ppl.py must pass --c4_seed, never --calib_seed, as the C4 shuffle seed."""
    from pathlib import Path

    src = (Path(__file__).parent.parent / "scripts" / "eval_ppl.py").read_text()
    call = src.split("= compute_perplexity(", 1)[1].split(")", 1)[0]
    assert "seed=args.c4_seed" in call
    assert "calib_seed" not in call
