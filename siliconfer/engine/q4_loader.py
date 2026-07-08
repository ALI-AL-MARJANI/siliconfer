"""Phase 6: Load a pretrained model and replace all attn/MLP linears with Q4Linear.

Usage:
    from siliconfer.engine.q4_loader import load_q4_model
    model, config = load_q4_model(model_dir, method="rtn")

Supported methods: "rtn", "gptq", "awq", "hqq", "sinq". "mixed" (2/4-bit mixed
precision, quant/mixed_precision.py) is deliberately NOT served here yet — see
the ValueError in load_q4_model for why; use scripts/quantize.py --method
mixed for algorithm-level PPL validation instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

from siliconfer.model.llama import LlamaModel
from siliconfer.model.config import ModelConfig
from siliconfer.model.q4_linear import Q4Linear
from siliconfer.kernels.neon import pack_weights_sym, pack_weights_asym


def _pack_and_replace_linears(
    model: LlamaModel,
    group_size: int = 128,
    skip_layers: set[int] | None = None,
    pack_sym: bool = True,
) -> None:
    """Replace every attn+MLP nn.Linear in the model with Q4Linear, in-place.

    Reads the current .weight (which may be fake-quant after RTN/GPTQ/AWQ/HQQ) and
    re-packs it into uint8 + float32 scales (+ zero-points if asymmetric). The
    round-trip is lossless *only if `pack_sym` matches the grid the fake-quant
    values are actually sitting on* — e.g. symmetric fake-quant (-8 is never
    reached, see NOTES.md) re-packed symmetrically recovers the same values
    exactly, but re-packing an asymmetric grid with the symmetric packer
    silently corrupts it (this was a real bug: HQQ is always asymmetric and
    was always re-packed symmetric before this parameter existed — see
    CLAUDE.md §9 for the diagnosis).

    Args:
        skip_layers: set of layer indices to leave in fp16 (mixed precision). E.g.,
            {0, n_layers-1} keeps the first and last layers in fp16 for lower PPL.
        pack_sym: whether the current weights are on a symmetric (True) or
            asymmetric (False) int4 grid — must match how they were quantized.
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

            W_np = np.array(lin.weight.astype(mx.float32))
            bias = getattr(lin, "bias", None)

            if pack_sym:
                packed, scales = pack_weights_sym(W_np, group_size=group_size)
                setattr(parent, name, Q4Linear(packed, scales, bias=bias, group_size=group_size))
            else:
                packed, scales, zeros = pack_weights_asym(W_np, group_size=group_size)
                setattr(parent, name, Q4Linear(packed, scales, zeros=zeros, bias=bias, group_size=group_size))


def load_q4_model(
    model_dir: str | Path,
    method: str = "rtn",
    group_size: int = 128,
    sym: bool = True,
    calib_model_id: str | None = None,
    n_calib_seqs: int = 128,
    calib_len: int = 512,
    skip_layers: set[int] | None = None,
    verbose: bool = True,
) -> tuple[LlamaModel, ModelConfig]:
    """Load a model from disk, quantize to int4, and return a kernel-backed model.

    Args:
        model_dir:      Path to HF model directory (safetensors + config.json).
        method:         "rtn" | "gptq" | "awq" | "hqq" | "sinq" — quantization algorithm.
        group_size:     int4 group size (64 or 128).
        sym:            Symmetric quantization (True) or asymmetric (False).
        calib_model_id: HF model ID for calibration tokenizer (GPTQ/AWQ only).
                        Defaults to the basename of model_dir.
        n_calib_seqs:   Number of WikiText-2 calibration sequences (GPTQ/AWQ).
        calib_len:      Sequence length per calibration sequence.
        skip_layers:    Optional set of layer indices to keep in fp16 (mixed precision).
                        E.g., skip_layers={0, 23} keeps first and last layers in fp16.
        verbose:        Print progress messages.

    Returns:
        (model, config) — LlamaModel with all linear layers replaced by Q4Linear.
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
            "method='mixed' is not supported here yet: _pack_and_replace_linears/Q4Linear "
            "only know how to pack a 4-bit (nibble) grid. Packing a 2-bit-quantized layer "
            "through the 4-bit packer would silently re-fit new scale/zero values onto a "
            "4-bit grid and store 4 bits per weight anyway — the exact same class of "
            "silent-mismatch bug documented in CLAUDE.md §9 for asymmetric packing, just "
            "for bit-width instead of symmetry. A real int2 packed kernel (2-bit "
            "nibble-of-4 packing + NEON GEMV/GEMM) is required before 'mixed' can be "
            "served through this path; use scripts/quantize.py --method mixed for the "
            "algorithm-level PPL comparison in the meantime (matches how every other "
            "method here was validated before its packed-kernel integration existed)."
        )

    elif method in ("gptq", "awq"):
        mid = calib_model_id or model_dir.name
        if verbose:
            print(f"[q4_loader] Loading {n_calib_seqs} calibration sequences "
                  f"(model_id={mid}) ...")
        from siliconfer.quant.calibration import load_calibration_sequences
        calib_seqs = load_calibration_sequences(mid, n_seqs=n_calib_seqs, seq_len=calib_len)

        if method == "gptq":
            if verbose:
                print(f"[q4_loader] Applying GPTQ-int4 ...")
            from siliconfer.quant.gptq import apply_gptq
            apply_gptq(model, calib_seqs, group_size=group_size, sym=sym, verbose=verbose)
        else:
            if verbose:
                print(f"[q4_loader] Applying AWQ-int4 ...")
            from siliconfer.quant.awq import apply_awq
            apply_awq(model, calib_seqs, group_size=group_size, sym=sym,
                      fold_scales=False, verbose=verbose)
        mx.eval(model.parameters())

    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'rtn', 'gptq', 'awq', 'hqq', or 'sinq'.")

    # HQQ has no `sym` concept at all — it's always asymmetric (see quant/hqq.py's
    # module docstring: the whole mechanism is fitting a zero-point). Every other
    # method's packing symmetry follows the `sym` flag actually used to fake-quantize.
    pack_sym = False if method == "hqq" else sym

    if verbose:
        print(f"[q4_loader] Packing int4 weights and replacing linear layers "
              f"({'symmetric' if pack_sym else 'asymmetric'}) ...")
    _pack_and_replace_linears(model, group_size=group_size, skip_layers=skip_layers, pack_sym=pack_sym)

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
                    total_bytes += lin._packed.nbytes + lin._scales.nbytes
                    if lin._zeros is not None:
                        total_bytes += lin._zeros.nbytes
                    n_q4 += 1
    print(f"[q4_loader] Replaced {n_q4} linear layers with Q4Linear. "
          f"Packed weight footprint: {total_bytes / 1e6:.1f} MB")
