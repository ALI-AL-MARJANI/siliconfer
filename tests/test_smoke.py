"""Phase 0 smoke tests: import, config parsing, loader basics."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import siliconfer
from siliconfer.model.config import ModelConfig
from siliconfer.engine.loader import load_model, weight_summary


def test_version():
    assert siliconfer.__version__ == "0.1.0"


def test_config_from_json_qwen():
    """Parse a Qwen2.5-0.5B-style config."""
    config_data = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "bos_token_id": 151643,
        "eos_token_id": 151643,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        cfg = ModelConfig.from_json(f.name)

    assert cfg.hidden_size == 896
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 64
    assert cfg.gqa_groups == 7
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.rope_theta == 1000000.0
    assert cfg.tie_word_embeddings is True
    assert cfg.vocab_size == 151936


def test_config_from_json_llama():
    """Parse a Llama-3.2-1B-style config with RoPE scaling."""
    config_data = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        cfg = ModelConfig.from_json(f.name)

    assert cfg.hidden_size == 2048
    assert cfg.head_dim == 64
    assert cfg.gqa_groups == 4
    assert cfg.rope_scaling is not None
    assert cfg.rope_scaling.type == "llama3"
    assert cfg.rope_scaling.factor == 32.0
    assert cfg.tie_word_embeddings is True


def test_loader_with_synthetic_weights():
    """Create a tiny synthetic model dir with safetensors + config, load it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        config_data = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "hidden_act": "silu",
            "torch_dtype": "float16",
        }
        with open(tmpdir / "config.json", "w") as f:
            json.dump(config_data, f)

        weights = {
            "model.embed_tokens.weight": mx.random.normal((128, 32)).astype(mx.float16),
            "model.layers.0.input_layernorm.weight": mx.ones((32,), dtype=mx.float16),
            "model.layers.0.self_attn.q_proj.weight": mx.random.normal((32, 32)).astype(mx.float16),
            "model.layers.0.self_attn.k_proj.weight": mx.random.normal((16, 32)).astype(mx.float16),
            "model.layers.0.self_attn.v_proj.weight": mx.random.normal((16, 32)).astype(mx.float16),
            "model.layers.0.self_attn.o_proj.weight": mx.random.normal((32, 32)).astype(mx.float16),
            "model.layers.0.post_attention_layernorm.weight": mx.ones((32,), dtype=mx.float16),
            "model.layers.0.mlp.gate_proj.weight": mx.random.normal((64, 32)).astype(mx.float16),
            "model.layers.0.mlp.up_proj.weight": mx.random.normal((64, 32)).astype(mx.float16),
            "model.layers.0.mlp.down_proj.weight": mx.random.normal((32, 64)).astype(mx.float16),
            "model.norm.weight": mx.ones((32,), dtype=mx.float16),
        }
        mx.save_safetensors(str(tmpdir / "model.safetensors"), weights)

        cfg, loaded = load_model(tmpdir)

        assert cfg.hidden_size == 32
        assert cfg.tie_word_embeddings is True
        assert "lm_head.weight" in loaded
        assert "model.embed_tokens.weight" in loaded
        np.testing.assert_array_equal(
            np.array(loaded["lm_head.weight"]),
            np.array(loaded["model.embed_tokens.weight"]),
        )

        summary = weight_summary(loaded)
        assert "tensors" in summary
