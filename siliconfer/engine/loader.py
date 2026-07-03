"""Load HuggingFace safetensors models into our own parameter dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from siliconfer.model.config import ModelConfig


def _find_safetensor_files(model_dir: Path) -> list[Path]:
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index: dict[str, Any] = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        return [model_dir / name for name in shard_names]

    raise FileNotFoundError(
        f"No safetensors found in {model_dir}. "
        "Expected model.safetensors or model.safetensors.index.json"
    )


def load_weights(model_dir: str | Path) -> dict[str, mx.array]:
    model_dir = Path(model_dir)
    shard_files = _find_safetensor_files(model_dir)

    weights: dict[str, mx.array] = {}
    for shard_path in shard_files:
        shard_weights = mx.load(str(shard_path))
        weights.update(shard_weights)

    return weights


def load_model(model_dir: str | Path) -> tuple[ModelConfig, dict[str, mx.array]]:
    model_dir = Path(model_dir)

    config = ModelConfig.from_json(model_dir / "config.json")
    weights = load_weights(model_dir)

    if config.tie_word_embeddings and "lm_head.weight" not in weights:
        if "model.embed_tokens.weight" in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    if "model.layers.0.self_attn.q_proj.bias" in weights:
        config.attention_bias = True

    return config, weights


def weight_summary(weights: dict[str, mx.array]) -> str:
    lines = []
    total_params = 0
    total_bytes = 0
    for name in sorted(weights.keys()):
        t = weights[name]
        n_params = 1
        for d in t.shape:
            n_params *= d
        n_bytes = n_params * t.dtype.size
        total_params += n_params
        total_bytes += n_bytes
        lines.append(f"  {name}: {list(t.shape)} {t.dtype}")

    lines.insert(0, f"Weights: {len(weights)} tensors, {total_params:,} params, "
                    f"{total_bytes / 1e9:.2f} GB")
    return "\n".join(lines)
