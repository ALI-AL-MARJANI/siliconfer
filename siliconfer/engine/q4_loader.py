"""Load a pretrained model, quantize it, and replace its projections with Q4Linear.

    from siliconfer.engine.q4_loader import load_q4_model
    model, config = load_q4_model(model_dir, method="gptq", backend="mlx")

Methods: "rtn", "gptq", "awq", "hqq", "sinq". Mixed precision is not served
(there is no 2-bit or 3-bit packed kernel); use scripts/quantize.py for it.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from siliconfer.kernels.neon import pack_weights_asym, pack_weights_on_grid, pack_weights_sym
from siliconfer.model.config import ModelConfig
from siliconfer.model.llama import LlamaModel
from siliconfer.model.q4_linear import Q4Linear


def _pack_and_replace_linears(
    model: LlamaModel,
    group_size: int = 128,
    skip_layers: set[int] | None = None,
    pack_sym: bool = True,
    backend: str = "neon",
) -> None:
    """Replace every attention and MLP nn.Linear with a Q4Linear, in place.

    Packing must use the grid the weight was quantized on (docs/packing.md).
    Three cases:

    - RTN, HQQ: the grid is re-derived from the fake-quantized weight, which is
      exact as long as `pack_sym` matches how it was quantized.
    - GPTQ: the grid is fixed before error feedback moves the weights, so it
      cannot be re-derived. `apply_gptq` attaches `_q4_grid` / `_q4_w_q`, which
      are packed with `pack_weights_on_grid`.
    - AWQ, SINQ: the effective weight Q(W·diag(s))·diag(1/s) is not on a group
      grid at all. They attach the grid-aligned weight and the input scale; the
      former is packed and the latter is passed to Q4Linear.

    Args:
        skip_layers: layer indices to leave in fp16.
        pack_sym: whether the weights are on a symmetric (True) or asymmetric
            (False) grid.
        backend: "neon" (CPU kernel) or "mlx" (GPU, MLX's quantized matmul on
            the same codes).
    """
    for i, layer in enumerate(model.layers):
        if skip_layers and i in skip_layers:
            continue
        attn = layer.self_attn
        mlp = layer.mlp

        proj_pairs = [
            (attn, "q_proj"),
            (attn, "k_proj"),
            (attn, "v_proj"),
            (attn, "o_proj"),
            (mlp,  "gate_proj"),
            (mlp,  "up_proj"),
            (mlp,  "down_proj"),
        ]

        for parent, name in proj_pairs:
            lin = getattr(parent, name)
            in_f = lin.weight.shape[1]

            if in_f < group_size:
                # Too small to quantize (synthetic / tiny test models)
                continue

            # AWQ/SINQ: pack the grid-aligned weight; Q4Linear applies input_scale.
            w_grid = getattr(lin, "_awq_w_grid", None)
            input_scale = getattr(lin, "_awq_input_scale", None)
            if w_grid is None:
                w_grid = getattr(lin, "_sinq_w_grid", None)
                input_scale = getattr(lin, "_sinq_input_scale", None)

            W_np = w_grid if w_grid is not None else np.array(lin.weight.astype(mx.float32))
            bias = getattr(lin, "bias", None)

            grid = getattr(lin, "_q4_grid", None)
            if grid is not None:
                # The quantizer fixed its grid before moving the weights (GPTQ):
                # pack on that grid rather than re-deriving one from the weight.
                scales, zeros = grid
                packed = pack_weights_on_grid(lin._q4_w_q, scales, zeros, group_size=group_size)
            elif pack_sym:
                packed, scales = pack_weights_sym(W_np, group_size=group_size)
                zeros = None
            else:
                packed, scales, zeros = pack_weights_asym(W_np, group_size=group_size)
            setattr(parent, name, Q4Linear(packed, scales, zeros=zeros, bias=bias,
                                           group_size=group_size, input_scale=input_scale,
                                           backend=backend))


def load_q4_model(
    model_dir: str | Path,
    method: str = "rtn",
    group_size: int = 128,
    sym: bool = True,
    calib_model_id: str | None = None,
    n_calib_seqs: int = 128,
    calib_len: int = 512,
    calib_seed: int = 42,
    awq_block_loss: bool = False,
    awq_clip: bool = False,
    skip_layers: set[int] | None = None,
    backend: str = "neon",
    verbose: bool = True,
) -> tuple[LlamaModel, ModelConfig]:
    """Load a model from disk, quantize it to int4 and return it with packed layers.

    Args:
        model_dir:      Hugging Face model directory (safetensors + config.json).
        method:         "rtn" | "gptq" | "awq" | "hqq" | "sinq".
        sym:            symmetric (True) or asymmetric (False) grid.
        calib_model_id: model id for the calibration tokenizer (GPTQ/AWQ).
                        Defaults to the basename of model_dir.
        n_calib_seqs:   number of calibration sequences (GPTQ/AWQ).
        calib_len:      tokens per calibration sequence.
        calib_seed:     seed for sampling the calibration sequences.
        awq_block_loss: AWQ: score α on the enclosing block's output.
        awq_clip:       AWQ: per-group weight clipping after scaling.
        skip_layers:    layer indices to keep in fp16.
        backend:        "neon" (CPU kernel) or "mlx" (GPU).
        verbose:        print progress.

    Returns:
        (model, config)
    """
    model_dir = Path(model_dir)

    if verbose:
        print(f"[q4_loader] Loading fp16 model from {model_dir} ...")
    model, config = LlamaModel.from_pretrained(model_dir, dtype=mx.float16)
    mx.eval(model.parameters())

    if method == "rtn":
        if verbose:
            print(f"[q4_loader] Applying RTN-int4 (group_size={group_size}, sym={sym}) ...")
        from siliconfer.quant.rtn import apply_rtn
        apply_rtn(model, group_size=group_size, sym=sym)
        mx.eval(model.parameters())

    elif method == "hqq":
        if verbose:
            print(f"[q4_loader] Applying HQQ-int4 (group_size={group_size}) ...")
        from siliconfer.quant.hqq import apply_hqq
        apply_hqq(model, group_size=group_size, verbose=verbose)
        mx.eval(model.parameters())

    elif method == "sinq":
        if verbose:
            print(f"[q4_loader] Applying SINQ-int4 (group_size={group_size}, sym={sym}) ...")
        from siliconfer.quant.sinq import apply_sinq
        apply_sinq(model, group_size=group_size, sym=sym, verbose=verbose)
        mx.eval(model.parameters())

    elif method == "mixed":
        raise ValueError(
            "method='mixed' cannot be served: only a 4-bit packed format exists, so a "
            "2-bit or 3-bit layer would be re-quantized and stored at 4 bits. "
            "Use scripts/quantize.py --method mixed for fake-quant evaluation."
        )

    elif method in ("gptq", "awq"):
        mid = calib_model_id or model_dir.name
        if verbose:
            print(f"[q4_loader] Loading {n_calib_seqs} calibration sequences "
                  f"(model_id={mid}) ...")
        from siliconfer.quant.calibration import load_calibration_sequences
        calib_seqs = load_calibration_sequences(mid, n_seqs=n_calib_seqs, seq_len=calib_len, seed=calib_seed)

        if method == "gptq":
            if verbose:
                print("[q4_loader] Applying GPTQ-int4 ...")
            from siliconfer.quant.gptq import apply_gptq
            apply_gptq(model, calib_seqs, group_size=group_size, sym=sym, verbose=verbose)
        else:
            if verbose:
                print("[q4_loader] Applying AWQ-int4 ...")
            from siliconfer.quant.awq import apply_awq
            apply_awq(model, calib_seqs, group_size=group_size, sym=sym,
                      fold_scales=False, block_loss=awq_block_loss, auto_clip=awq_clip,
                      verbose=verbose)
        mx.eval(model.parameters())

    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'rtn', 'gptq', 'awq', 'hqq', or 'sinq'.")

    # HQQ is always asymmetric; the other methods follow `sym`.
    pack_sym = False if method == "hqq" else sym

    if verbose:
        print(f"[q4_loader] Packing int4 weights and replacing linear layers "
              f"({'symmetric' if pack_sym else 'asymmetric'}) ...")
    _pack_and_replace_linears(model, group_size=group_size, skip_layers=skip_layers,
                              pack_sym=pack_sym, backend=backend)

    if verbose:
        _report_memory(model)

    return model, config


def _report_memory(model: LlamaModel) -> None:
    """Print approximate packed weight memory for the model."""
    total_bytes = 0
    n_q4 = 0
    for layer in model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"):
                lin = getattr(parent, name, None)
                if isinstance(lin, Q4Linear):
                    # packed weights + scales (+ zeros if asymmetric)
                    total_bytes += lin.nbytes
                    n_q4 += 1
    print(f"[q4_loader] Replaced {n_q4} linear layers with Q4Linear. "
          f"Packed weight footprint: {total_bytes / 1e6:.1f} MB")
