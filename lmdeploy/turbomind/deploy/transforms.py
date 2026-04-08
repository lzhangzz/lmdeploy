# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch

from .linear import Linear, chunk_linears, interleave_linears


def _should_fuse_silu(w1_linear: Linear, act_type: str, is_moe: bool = False) -> bool:
    """Determine if fused SiLU (interleave) should be used for w1+w3 fusion.

    Gold standard condition (from GEMM kernel constraints — trust it):
        act_type == SiLU && (int4 || mxfp4 || fp8 || moe) && !(fp8 && SM90)
    """
    if act_type not in ('', 'silu', 'SiLU'):
        return False

    # Dense bf16/fp16 without MoE -> chunk, not interleave
    weight = w1_linear.tensors.get("weight")
    if weight is None:
        weight = w1_linear.tensors.get("qweight")
    is_quantized = weight is not None and weight.element_size() < 2
    if not is_quantized and not is_moe:
        return False

    # FP8 on SM90 -> chunk
    fmt = w1_linear.weight_format
    if fmt is not None and fmt.name == "fp8":
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap == (9, 0):
                return False

    return True


def _shard_linear_for_tp(linear: Linear, tp: int, rank: int) -> Linear:
    """Extract the TP shard for *rank*, handling block-scale alignment.

    For formats with ``block_out`` (e.g. FP8), scale/zero tensors have reduced
    dimensions ``[K/gs, N/gs]`` that may not be evenly divisible by *tp*.
    This function splits weight/bias normally along the output dim but extracts
    scale/zero entries based on block boundaries so that each rank gets exactly
    ``cdiv(N/tp, block_out)`` entries — matching the C++ allocation.
    """
    fmt = linear.weight_format
    if fmt is None or fmt.block_out is None:
        return linear.split_out_dim(tp)[rank]

    scales = linear.tensors.get("scales")
    if scales is None or scales.size(-1) % tp == 0:
        return linear.split_out_dim(tp)[rank]

    # Block-scale tensors can't be evenly split — extract by block boundary.
    weight = linear.tensors.get("weight")
    if weight is None:
        weight = linear.tensors.get("qweight")
    if weight is None:
        return linear.split_out_dim(tp)[rank]

    N = weight.size(-1)
    block_out = fmt.block_out
    W = N // tp  # per-rank output dim

    shard_tensors: dict[str, torch.Tensor] = {}
    for kind, t in linear.tensors.items():
        if kind in ("scales", "zeros") and t.dim() >= 2:
            # Extract scale entries covering this rank's weight columns.
            start_block = (rank * W) // block_out
            end_block = -(-(rank + 1) * W // block_out)  # ceil div
            shard_tensors[kind] = t[:, start_block:end_block].contiguous()
        elif t.dim() >= 2:
            # Weight / qweight: split output dim normally
            split_size = t.size(-1) // tp
            shard_tensors[kind] = t[..., rank * split_size:(rank + 1) * split_size].contiguous()
        else:
            # 1-D (bias): split normally
            split_size = t.size(0) // tp
            shard_tensors[kind] = t[rank * split_size:(rank + 1) * split_size].contiguous()

    return Linear(tensors=shard_tensors, weight_format=fmt,
                  data_format=linear.data_format)


def _can_fuse_w1w3(w1: Linear, tp: int) -> bool:
    """Check whether w1+w3 fusion is safe for the given TP.

    Fusion (interleave or chunk) concatenates w1 and w3 along the output dim.
    For block-quantized formats (e.g. FP8 with block_out=128), the fused
    scale count ``2 * cdiv(N/tp, block_out)`` must equal
    ``cdiv(2*N/tp, block_out)``.  This holds iff ``(N/tp) % block_out == 0``.
    When it doesn't, the fused module's C++ allocation won't match the
    concatenated scales and we must commit w1/w3 separately.
    """
    if tp <= 1:
        return True
    fmt = w1.weight_format
    if fmt is None or fmt.block_out is None:
        return True
    w = w1.tensors.get("weight")
    if w is None:
        w = w1.tensors.get("qweight")
    if w is None:
        return True
    return (w.size(-1) // tp) % fmt.block_out == 0


def fuse_ffn_linears(
    w1: Linear,
    w3: Linear,
    tp: int,
    act_type: str,
    is_moe: bool = False,
) -> tuple[Linear | None, bool]:
    """Optionally fuse w1/w3 on full (unsharded) tensors for FFN.

    Returns (fused_w1w3_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set.
    When block-scale boundaries prevent fusion, returns (None, fused_silu).

    TP sharding is NOT done here — the caller's commit path handles it
    via split_side=SplitSide.OUTPUT.  ``tp`` is only used for the
    block-scale alignment check in ``_can_fuse_w1w3``.
    """
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)

    if can_fuse:
        if fused_silu:
            w1w3 = interleave_linears(w1, w3)
        else:
            w1w3 = chunk_linears(w1, w3)
        return (w1w3, fused_silu)
    else:
        return (None, fused_silu)
