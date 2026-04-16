# Spec attn() Cleanup Design

## Goal

Remove dead and redundant parameters from spec `attn()` methods and config factory functions: `repeat_kv`, `tp_rank`, and `window_size`.

## Background

After the `add_qkv_proj` refactoring (which moved KV repetition into the builder's `pad_for_tp`), several parameters flowing through specs to config factories are now redundant or dead.

## Changes

### 1. Remove `repeat_kv` entirely

`repeat_kv` is dead. KV head repetition is now handled by `pad_for_tp` in the builder. The field serves no purpose in the C++ runtime.

**Files to modify:**

- `module_configs.py` — remove `repeat_kv` param from `make_attention_config` and `cfg.repeat_kv = repeat_kv` line. Remove `cfg.repeat_kv = 0` from `make_mla_config`.
- `spec.py` — remove `_repeat_kv: int = 0` class attribute.
- `text_model_loader.py` — remove `spec._repeat_kv = self.model.repeat_kv`.
- `target_model/base.py` — remove `repeat_kv` from `finalize_config` return value. Remove `repeat_kv` param from `BaseOutputModel.__init__`. Remove `self.repeat_kv = repeat_kv`.
- `converter.py` — stop capturing `repeat_kv` from `finalize_config`. Remove `repeat_kv` from `create_turbomind_model` return. Remove `repeat_kv=` from `BaseOutputModel(...)` constructor call.

**Verification:** grep for all remaining `repeat_kv` references after changes and delete them.

### 2. Default `tp_rank=0` in factory signatures

The builder's `_ensure_handles()` already sets the correct `tp_rank` via `cfg.tp_rank = self._ranks[i]` when `tp > 1`. Specs always pass `tp_rank=0`, which gets overridden. Make `tp_rank` a defaulted keyword arg so specs can omit it.

**Files to modify:**

- `module_configs.py` — change `make_attention_config`, `make_mla_config`, `make_ffn_config`, `make_moe_config`, `make_deltanet_config` signatures to `tp_rank=0` (keyword default, keep the assignment to `cfg.tp_rank`).
- All spec files — remove `tp_rank=0` from factory calls (it's the default).

### 3. Default `window_size=0` in factory signatures

Only `gpt_oss_spec.py` passes non-zero window_size. All other specs always pass 0.

**Files to modify:**

- `module_configs.py` — change `make_attention_config` signature to `window_size=0` (keyword default). Same for `make_mla_config`.
- `qwen3_spec.py` — remove the 4-line per-layer window_size lookup and the `window_size=window_size` arg.
- `qwen3_5_spec.py` — same removal.
- `gpt_oss_spec.py` — keep the per-layer lookup, it's still needed.
- `glm4_moe_lite_spec.py` — remove `window_size=0` (it's the default).

## Scope

This is a parameter cleanup only. No logic changes. No new functions. No behavioral changes.
