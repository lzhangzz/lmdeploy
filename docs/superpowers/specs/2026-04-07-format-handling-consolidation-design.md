# Format Handling Consolidation Design

Date: 2026-04-07

## Problem

The TurboMind model loading pipeline has format handling knowledge scattered
across multiple files. The codebase is in active transition from an old
parameter-based API to a new `WeightFormat`/`Linear`-based pipeline, and the
old code remains alongside the new. Specifically:

- `policy.py` contains old normalizer functions (`process_awq_gemm`,
  `process_gptq`, etc.) that duplicate what `kind_map.py` already provides
  via `WeightFormat.normalizer`.
- `parameter.py` contains old `Parameter` subclasses (`QuantWeightOnly`,
  `WeightScaleInv`, etc.) that have no callers, alongside the new
  `build_linear()` function.
- `load_context.py` duplicates commit logic from `module.py`'s
  `commit_linear()` in `_commit_linear_to_handle()`.
- Source model specs store `self.policy` from `input_policy` but never use it.

## Goal

Consolidate all format handling into `kind_map.py` as the single source of
truth, delete dead code, and deduplicate commit logic.

## Design

### 1. `kind_map.py` becomes the format package

`kind_map.py` already contains:
- `WeightFormat` dataclass (descriptors, normalizers, packers, accepts)
- Per-format singletons (`DENSE_FORMAT`, `AWQ_FORMAT`, etc.)
- Format priority list for auto-detection

Move into it:
- `build_linear()` from `parameter.py` — auto-detects format and builds
  `Linear` bundles from checkpoint tensors.
- `pack_u4_row()` from `parameter.py` — uint8 packing utility used by
  format packers defined in `kind_map.py`.
- Small dtype helpers (`identity`, `to_half`, `to_float`, `to_fp8`) from
  `parameter.py` that are used by `kind_map.py` packers.

### 2. Delete `policy.py`

`policy.py` exports `get_input_policy()` which returns format-specific
normalizer functions. These functions are stored as `self.policy` in source
model specs but never called. The `converter.py` line that calls
`get_input_policy()` and passes the result to constructors is dead plumbing.

Actions:
- Delete `policy.py`.
- Remove `get_input_policy` import from `converter.py`.
- Remove `input_policy = get_input_policy(...)` from `converter.py`.
- Remove `input_policy=input_policy` from input model constructor call.
- Remove `self.policy = kwargs.get('input_policy')` from all source model
  spec constructors (`qwen3_spec.py`, `qwen3_5_spec.py`, `gpt_oss_spec.py`,
  `glm4_moe_lite_spec.py`).

### 3. Delete `parameter.py`

After moving `build_linear()`, `pack_u4_row()`, and dtype helpers into
`kind_map.py`, `parameter.py` contains only the old `Parameter` subclasses
and `get_params()` which have no callers outside the file.

Actions:
- Move `build_linear()` to `kind_map.py`.
- Move `pack_u4_row()` to `kind_map.py`.
- Move dtype helpers (`identity`, `to_half`, `to_float`, `to_fp8`) to
  `kind_map.py`.
- Move `generate_zero_point()` to `kind_map.py` (used by GPTQ format).
- Update all imports of these symbols to point to `kind_map.py`.
- Delete `parameter.py`.

### 4. Deduplicate commit logic

`module.py`'s `commit_linear()` (lines 864-1008) and `load_context.py`'s
`_commit_linear_to_handle()` (lines 144-195) perform the same tensor
commit operations. The load_context version is a stripped-down copy.

Solution: Extract the tensor-commit loop (pack, split, allocate, cast, pad,
copy) from `commit_linear()` into a standalone function
`_commit_tensors(handle, linear, cpp_dtype, group_size, split_side, split_num, rank)`
that operates on a pre-created LinearWeight handle.

`commit_linear()` calls `_commit_tensors()` after child creation.
`LoadContext.load_linear()` calls `_commit_tensors()` after creating the
child via `create_child()`. Remove `_commit_linear_to_handle()` from
`load_context.py`.

### 5. File structure after refactoring

| File | Role |
|------|------|
| `kind_map.py` | All format knowledge: `WeightFormat`, `build_linear()`, packers, normalizers |
| `policy.py` | **Deleted** |
| `parameter.py` | **Deleted** |
| `module.py` | `TextModelSpec`, `commit_linear`, `commit_tensor`, QKV/GDN helpers, TP rules |
| `load_context.py` | `LoadContext` — thin composable wrapper, delegates to `module.commit_linear` |
| `text_model_loader.py` | `TextModelLoader` — driver, unchanged |
| `converter.py` | Orchestration, no `input_policy` plumbing |
| `source_model/*.py` | Specs, no `self.policy` dead state |

## Verification

Test with `turbomind-tester` agent across multiple model formats:
- Dense (BF16/FP16) — most common baseline
- AWQ quantized — verifies old policy removal doesn't break AWQ loading
- GPTQ quantized — same for GPTQ
- FP8 — verifies block-scale handling
- MXFP4 — verifies packer logic after move

Each test must produce meaningful human-readable output for at least 128
tokens.

## Scope

This refactoring touches only the Python loading pipeline
(`lmdeploy/turbomind/deploy/`). No C++ changes are needed. No new features
are added — this is purely cleanup of the existing transition.
