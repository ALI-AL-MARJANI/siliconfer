"""Model configuration parsed from HuggingFace config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoPEScalingConfig:
    type: str = ""
    factor: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class ModelConfig:
    architectures: list[str] = field(default_factory=list)
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: RoPEScalingConfig | None = None
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    torch_dtype: str = "float16"
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    attention_bias: bool = False
    model_type: str = ""

    def __post_init__(self) -> None:
        if self.head_dim == 0 and self.num_attention_heads > 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def gqa_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        path = Path(path)
        with open(path) as f:
            raw: dict[str, Any] = json.load(f)

        rope_scaling = None
        if raw_rope := raw.get("rope_scaling"):
            rope_scaling = RoPEScalingConfig(
                type=raw_rope.get("type", ""),
                factor=raw_rope.get("factor", 1.0),
                low_freq_factor=raw_rope.get("low_freq_factor", 1.0),
                high_freq_factor=raw_rope.get("high_freq_factor", 4.0),
                original_max_position_embeddings=raw_rope.get(
                    "original_max_position_embeddings", 8192
                ),
            )

        return cls(
            architectures=raw.get("architectures", []),
            hidden_size=raw.get("hidden_size", 0),
            intermediate_size=raw.get("intermediate_size", 0),
            num_hidden_layers=raw.get("num_hidden_layers", 0),
            num_attention_heads=raw.get("num_attention_heads", 0),
            num_key_value_heads=raw.get("num_key_value_heads", 0),
            head_dim=raw.get("head_dim", 0),
            vocab_size=raw.get("vocab_size", 0),
            max_position_embeddings=raw.get("max_position_embeddings", 0),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-5),
            rope_theta=raw.get("rope_theta", 500000.0),
            rope_scaling=rope_scaling,
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            hidden_act=raw.get("hidden_act", "silu"),
            torch_dtype=raw.get("torch_dtype", "float16"),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=raw.get("eos_token_id"),
            attention_bias=raw.get("attention_bias", False),
            model_type=raw.get("model_type", ""),
        )

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_type} ({self.architectures})",
            f"  hidden_size={self.hidden_size}, layers={self.num_hidden_layers}",
            f"  Q heads={self.num_attention_heads}, KV heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, GQA={self.gqa_groups}:1",
            f"  intermediate={self.intermediate_size}, vocab={self.vocab_size}",
            f"  rope_theta={self.rope_theta}, rms_norm_eps={self.rms_norm_eps}",
            f"  tie_embeddings={self.tie_word_embeddings}, attn_bias={self.attention_bias}",
            f"  dtype={self.torch_dtype}",
        ]
        if self.rope_scaling:
            lines.append(
                f"  rope_scaling: type={self.rope_scaling.type}, "
                f"factor={self.rope_scaling.factor}"
            )
        return "\n".join(lines)
