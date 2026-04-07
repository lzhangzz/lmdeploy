# Spec: Layered Split of module.py

**Date:** 2026-04-07
**Status:** Draft
**Scope:** Python only (C++ unchanged this round)

## Problem

`module.py` is 1072 lines mixing three distinct operational layers:
reading/assembling weights, transforming them, and committing them to C++.
This makes it hard to trace data flow, debug weight issues, or understand
what operations happen to a weight between checkpoint and GPU.

## Approach

Split `module.py` into three files organized by operational layer, plus a
backward-compatible facade. No C++ changes. All existing model specs and
`text_model_loader.py` continue working without import changes.

## Layered Architecture

```
spec.py (Read & Assemble)
    ↓ calls
transforms.py (Transform)
    ↓ called by
commit.py (Shard & Commit)
    ↓ writes to
C++ Module tree (unchanged)
```

Each layer depends only on the layer below. `commit.py` imports only
`SplitSide` (a simple enum) from `spec.py`. `transforms.py` operates on
`Linear` objects without touching C++ modules.

## File Assignments

### `spec.py` (~250 lines) — Read & Assemble

Responsible for reading weights from checkpoint, detecting format, building
`Linear` bundles, and assembling composite weights (QKV merge, GDN fusion).

**Contents:**
- `SplitSide` enum — semantic TP split direction (used by `TextModelSpec.raw_layer_tensors()` return type)
- `TextModelSpec` ABC — the declarative model weight interface
- `_dequant_linear()` — dequantize when needed for merge
- `_ensure_compatible_formats()` — unify formats before merge
- `_block_ops_need_dequant()` — check if block boundaries are crossed
- `merge_qkv_linear()` — merge Q/K/V into single interleaved Linear
- `fuse_gdn_in_proj()` — fuse GDN input projections with TP interleaving
- Raw tensor helpers: `permute_v2()`, `permute_v2_partial()`,
  `merge_qkv_v2()`, `merge_qkvg_v2()`
- Constants: `_GDN_IN_PROJ_KEYS`, `_tp_interleave_tensor()`

### `transforms.py` (~100 lines) — Transform

Responsible for weight transformations that happen before TP sharding and
commit. Operates on `Linear` objects only — no C++ interaction.

**Contents:**
- `_should_fuse_silu()` — decide interleave vs chunk for FFN
- `_can_fuse_w1w3()` — check block-scale alignment for w1/w3 fusion
- `_shard_linear_for_tp()` — extract TP shard from a Linear, handling
  block-scale alignment (operates on `Linear` only, no C++ alloc)
- `fuse_ffn_linears(w1, w3, tp, rank, act_type, is_moe)` — **new pure
  function** that handles TP sharding + w1/w3 interleave/chunk. Returns
  a tuple of `(fused_linear_or_none, w1_shard_or_none, w3_shard_or_none,
  fused_silu_flag)` for the commit layer to handle.

### `commit.py` (~350 lines) — Shard & Commit

Responsible for the final step: allocating C++ tensors and copying data to
GPU. Also owns TP split rules (configuration for the sharding step).

**Contents:**
- `SplitSide` is imported from `spec.py` (defined there because `TextModelSpec` uses it)
- `_SPLIT_SIDE_TO_DIM` mapping
- Dtype helpers: `_torch_dtype_to_cpp()`, `_cast_shard_for_tm()`
- Format inference: `_infer_cpp_linear_dtype()`, `_infer_compute_dtype()`
- Core commit: `_commit_tensors()` — the tensor commit loop
- `commit_linear()` — commit a Linear bundle to a C++ Module
- `commit_tensor()` — commit a raw tensor to a C++ Module
- `commit_ffn(ffn_mod, w1, w3, w2, tp, rank, act_type, is_moe,
  model_dtype)` — **new** orchestration function that calls
  `transforms.fuse_ffn_linears()` then `commit_linear()` for the result.
  Replaces the old `_fuse_and_commit_ffn()`.
- TP rules: `_ATTN_TP_RULES`, `_FFN_TP_RULES`, `_LINEAR_ATTN_TP_RULES`

### `module.py` (~30 lines) — Backward-compatible facade

Re-exports all public names so nothing breaks. Example:

```python
# module.py - backward-compatible facade
from .spec import (TextModelSpec, permute_v2, permute_v2_partial,
                   merge_qkv_v2, merge_qkvg_v2, merge_qkv_linear,
                   fuse_gdn_in_proj)
from .commit import (SplitSide, commit_linear, commit_tensor,
                     commit_ffn, _ATTN_TP_RULES, _FFN_TP_RULES,
                     _LINEAR_ATTN_TP_RULES, _fuse_and_commit_ffn)
from .transforms import (_should_fuse_silu, _shard_linear_for_tp,
                         _can_fuse_w1w3)

# Backward compat: _fuse_and_commit_ffn is now commit_ffn
_fuse_and_commit_ffn = commit_ffn
```

## Changes to Existing Files

### `text_model_loader.py`

Import `_fuse_and_commit_ffn` continues to work via the facade. A follow-up
PR can update imports to use `commit_ffn` directly.

No other files need changes.

## New Function: `fuse_ffn_linears`

The key new abstraction. Extracted from `_fuse_and_commit_ffn` to separate
transform logic from commit logic:

```python
def fuse_ffn_linears(
    w1: Linear, w3: Linear, tp: int, rank: int,
    act_type: str, is_moe: bool = False,
) -> tuple[Linear | None, Linear | None, Linear | None, bool]:
    """TP-shard and optionally fuse w1/w3 for FFN.

    Returns (fused_w1w3_or_none, w1_shard_or_none, w3_shard_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set and shards are None.
    When block-scale boundaries prevent fusion, shards are set individually.
    """
```

The commit layer's `commit_ffn()` calls this and handles the C++ allocation:

```python
def commit_ffn(ffn_mod, w1, w3, w2, tp, rank, act_type,
               is_moe=False, model_dtype=None):
    fused, w1_shard, w3_shard, fused_silu = fuse_ffn_linears(
        w1, w3, tp, rank, act_type, is_moe)
    if fused is not None:
        commit_linear(ffn_mod, fused, "w1w3", model_dtype=model_dtype)
        ffn_mod.set_fused_silu(fused_silu)
    else:
        for name, shard in (("w1", w1_shard), ("w3", w3_shard)):
            commit_linear(ffn_mod, shard, name, ...)
    if w2 is not None:
        commit_linear(ffn_mod, w2, "w2", ...)
```

## Verification

- All existing model specs (GptOssSpec, Qwen3Spec, Qwen3.5Spec, etc.)
  continue working without code changes
- `text_model_loader.py` continues working via facade re-exports
- Run existing model tests to verify correctness with TP=1 and TP=2
- Line counts: spec.py ~250, transforms.py ~100, commit.py ~350, module.py ~30

## Out of Scope

- C++ Module tree restructuring
- `turbomind.cc` refactoring
- Updating imports in model specs (can do in follow-up)
- `linear.py` or `kind_map.py` changes
