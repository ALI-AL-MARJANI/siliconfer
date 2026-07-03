"""Phase 1 logit parity test: our engine vs HuggingFace reference."""

import pytest
pytestmark = pytest.mark.integration

import numpy as np
import mlx.core as mx
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from siliconfer.model.llama import LlamaModel


MODEL_ID = "Qwen/Qwen2.5-0.5B"
PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent",
    "Once upon a time there was",
]


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def our_model():
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        MODEL_ID, allow_patterns=["*.safetensors", "config.json", "tokenizer*"]
    )
    model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float32)
    return model, config


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logit_parity(hf_model_and_tokenizer, our_model, prompt):
    """Compare logits between our engine and HuggingFace for the same input."""
    hf_model, tokenizer = hf_model_and_tokenizer
    model, config = our_model

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_pt = inputs["input_ids"]
    input_ids_mx = mx.array(input_ids_pt.numpy())

    with torch.no_grad():
        hf_out = hf_model(input_ids_pt)
        hf_logits = hf_out.logits.float().numpy()

    our_logits, _ = model(input_ids_mx)
    mx.eval(our_logits)
    our_logits_np = np.array(our_logits.astype(mx.float32))

    max_abs_diff = np.max(np.abs(hf_logits - our_logits_np))
    mean_abs_diff = np.mean(np.abs(hf_logits - our_logits_np))

    hf_top1 = np.argmax(hf_logits[:, -1, :], axis=-1)
    our_top1 = np.argmax(our_logits_np[:, -1, :], axis=-1)
    top1_match = (hf_top1 == our_top1).all()

    hf_top5 = set(np.argsort(hf_logits[0, -1, :])[-5:])
    our_top5 = set(np.argsort(our_logits_np[0, -1, :])[-5:])
    top5_overlap = len(hf_top5 & our_top5) / 5

    hf_probs = torch.softmax(torch.tensor(hf_logits[:, -1, :]), dim=-1).numpy()
    our_probs_t = torch.softmax(torch.tensor(our_logits_np[:, -1, :]), dim=-1).numpy()
    kl = np.sum(hf_probs * np.log((hf_probs + 1e-10) / (our_probs_t + 1e-10)), axis=-1).mean()

    print(f"\n  Prompt: '{prompt}'")
    print(f"  Max abs diff: {max_abs_diff:.6f}")
    print(f"  Mean abs diff: {mean_abs_diff:.6f}")
    print(f"  Top-1 match: {top1_match} (HF={hf_top1[0]}, ours={our_top1[0]})")
    print(f"  Top-5 overlap: {top5_overlap:.0%}")
    print(f"  KL divergence: {kl:.6f}")

    assert max_abs_diff < 1.0, f"Max abs diff too large: {max_abs_diff}"
    assert top1_match, f"Top-1 mismatch: HF={hf_top1[0]} vs ours={our_top1[0]}"
    assert top5_overlap >= 0.6, f"Top-5 overlap too low: {top5_overlap}"
    assert kl < 0.01, f"KL divergence too large: {kl}"


def test_generation_coherence(our_model, hf_model_and_tokenizer):
    """Verify generated text is coherent (not garbage)."""
    from siliconfer.engine.generate import generate, SamplingParams

    model, config = our_model
    _, tokenizer = hf_model_and_tokenizer

    prompt = "The theory of relativity states that"
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    input_ids_mx = mx.array(input_ids)

    params = SamplingParams(temperature=0.0, max_tokens=32)
    result = generate(model, input_ids_mx, params, eos_token_id=config.eos_token_id)

    full_ids = input_ids[0].tolist() + result.token_ids
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    print(f"\n  Generated: {text}")
    print(f"  Prefill: {result.prefill_tok_s:.1f} tok/s, Decode: {result.decode_tok_s:.1f} tok/s")

    assert len(result.token_ids) > 10, "Generation too short"
    assert result.decode_tok_s > 0, "Decode speed should be positive"
