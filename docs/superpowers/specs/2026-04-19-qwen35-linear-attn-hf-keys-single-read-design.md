# Qwen3.5 spec: single read site for linear-attention HF keys

**Date:** 2026-04-19  
**Status:** Approved approach (per-spec locals)

## Problem

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, the block that builds `DeltaNetConfig` reads Hugging Face config keys `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, and `linear_value_head_dim` from `hf_cfg`, then the same keys are read again immediately below to compute `self._linear_qkv_split`. Duplicate lookups invite drift if one site is edited (e.g. defaults or `.get()` behavior) and the other is not.

## Decision

Use **per-spec deduplication only** (brainstorm option 1): inside `Qwen3_5Spec.__init__`, when `self._layer_types` is non-empty, bind those four values **once** (local names or a single unpack), assign `self._dn_cfg` fields from those bindings, and compute `self._linear_qkv_split` from the same bindings.

No new shared helper under `source_model/`, no changes to `TextModelSpec._parse_base`, and no new datatypes unless a future spec repeats the pattern and a helper becomes justified.

## Behavior and compatibility

- **Semantics:** Identical tensor layout and C++ config values after the change; only the Python data flow is simplified.
- **Errors:** Keep required-key semantics: if a key is missing today (`[]` access), it should remain missing after refactor (no silent introduction of `.get()` defaults for the four head/dim keys unless product explicitly requires it).
- **`linear_conv_kernel_dim`:** Unchanged; only the four duplicated head/dim reads are consolidated.

## Verification

- Run existing tests or deploy smoke paths that exercise `qwen3_5` / `qwen3_5-moe` loading if present in the repo.
- Quick sanity: `self._linear_qkv_split` tuple equals the prior formula for the same `hf_cfg`.

## Out of scope

- Cross-spec helpers for linear attention.
- Broader deduplication of `AttentionConfig` / `FfnConfig` construction across `qwen3_spec.py` and `qwen3_5_spec.py`.
