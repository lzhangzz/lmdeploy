# MoE Gate/Shared_Gate as Linear Weights

Date: 2026-04-12

## Summary

Change MoE gate and shared_gate weight handling from raw `commit_tensor` to `commit_linear` via `Linear` bundles, consistent with how attention and FFN linears are committed.

## Motivation

Currently `_process_moe` pre-creates gate/shared_gate children with explicit `LinearConfig` then commits weights via `commit_tensor` through a dot-navigation loop. This is inconsistent with how all other linear layers (attention, FFN, expert FFN) are handled — they go through `commit_linear` with `Linear` bundles.

## Changes

### 1. New `moe_gate()` method on `TextModelSpec`

Base class returns `{}`. Each MoE spec overrides to return gate (and shared_gate) as `dict[str, Linear]`:

| Spec | Keys |
|------|------|
| Qwen3.5 | `"gate": Linear({"weight": ...})`, `"shared_gate": Linear({"weight": ...})` |
| Qwen3 | `"gate": Linear({"weight": ...})` |
| GptOss | `"gate": Linear({"weight": ..., "bias": ...})` |
| GLM4 MoE Lite | `"gate": Linear({"weight": ..., "bias": ...})` |

The transposition (`gate.t()`) moves from `moe_params()` to `moe_gate()`. No change in behavior.

### 2. Simplify `moe_params()`

Remove all gate/shared_gate entries. Only `score_correction_bias` remains (GLM4 MoE Lite only). All other specs return `{}`.

### 3. Rewrite `_process_moe` non-expert section

Remove gate/shared_gate pre-creation (6 lines) and the dot-navigation loop (7 lines). Replace with:

```python
# --- gate linears ---
for name, linear in spec.moe_gate(layer).items():
    moe.commit_linear(name, linear, model_dtype=dtype)

# --- non-expert MoE parameters (score_correction_bias, etc.) ---
for name, (tensor, split_side) in spec.moe_params(layer).items():
    moe.commit_tensor(name, tensor, split_side=split_side)
```

`commit_linear` handles child creation automatically. `moe_params()` now only returns flat names, so no dot-navigation needed.

### 4. Remove unused variables

After removing gate pre-creation, `hidden` is no longer used in `_process_moe` outside of the removed code. Remove it.

## Files affected

- `lmdeploy/turbomind/deploy/spec.py` — add `moe_gate()` base method
- `lmdeploy/turbomind/deploy/text_model_loader.py` — rewrite `_process_moe` non-expert section
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — add `moe_gate()`, simplify `moe_params()`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — add `moe_gate()`, simplify `moe_params()`
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — add `moe_gate()`, simplify `moe_params()`
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — add `moe_gate()`, simplify `moe_params()`
