# Eliminate Python-side ModelConfig Design

## Goal

Remove the Python `ModelConfig` dataclass and its supporting infrastructure. The C++ side no longer reads `model_config` from YAML — model architecture is discovered from weights. The Python `ModelConfig` exists solely as dead serialization format and should be eliminated.

## Current State

The C++ side (`turbomind.cc`) stopped parsing `node["model_config"]` in a prior change. It reads only `attention_config` and `engine_config` from the YAML string. `ModelWeight::prepare()` derives all model architecture fields from loaded weights.

The Python side still:
1. Builds a `ModelConfig` with ~45 fields via `to_legacy_config()` and `_copy_*_fields()`
2. Serializes it into YAML via `TurbomindModelConfig.to_dict()`
3. Passes it to `_tm.TurboMind.create()` — where C++ ignores both `model_config` and `lora_config` sections

**Note on naming**: `ModelConfig` uses `data_type`, `TurbomindEngineConfig` uses `dtype`. C++ reads `engine["dtype"]`. After migration, consumers read `engine_config.dtype`.

`update_from_engine_config()` patches 7 fields from `TurbomindEngineConfig` into model/attention configs:
- → `model_config`: `session_len`, `model_format`, `attn_tp_size`, `attn_cp_size`, `mlp_tp_size`
- → `attention_config`: `cache_block_seq_len`, `use_logn_attn`
- Special: `rope_scaling_factor` → creates/updates `attention_config.rope_param`

The attention_config patching must be preserved — C++ reads `attention_config.cache_block_seq_len` from YAML.

Runtime reads from `ModelConfig` (only 2 actual runtime reads after construction):
- `turbomind.py:558` — `config.model_config.data_type` for embedding dtype (every inference with input embeddings)
- `turbomind.py:660` — `config.model_config.vocab_size` for grammar compilation (guided decoding)

Build-time reads from `ModelConfig`:
- `turbomind.py:175` — `config.session_len` (snapshot at init)
- `LoadContext:~399` — `model_config.data_type` for C++ dtype
- `BaseOutputModel` — `model_config.attn_tp_size`, `attn_cp_size`, `mlp_tp_size`

All of these values are available from `TurbomindEngineConfig`. `vocab_size` is not exposed via pybind — it can be saved from the spec during construction (e.g., `self._vocab_size = spec._vocab_size`).

## PR Sequence

### PR 1: Stop serializing `model_config` to YAML

**Files:** `lmdeploy/turbomind/turbomind.py`, `lmdeploy/turbomind/deploy/config.py`

In `_postprocess_config()`, change `TurbomindModelConfig.to_dict()` to exclude `model_config`. The YAML dict should contain only `attention_config` (C++ reads `cache_block_seq_len` from it) plus `engine_config` from the merge. `lora_config` is also dead on the C++ side but is left for a separate cleanup.

Changes:
- `TurbomindModelConfig.to_dict()`: drop `model_config` from the returned dict
- `_postprocess_config()`: still calls `update_from_engine_config()` — it patches `attention_config` with `cache_block_seq_len`, `use_logn_attn`, and `rope_param` from `engine_config`, all of which C++ reads from the `attention_config` YAML section

The `model_config` Python object is still constructed and stored on `TurbomindModelConfig` for Python-side reads (migrated in PR 2).

### PR 2: Migrate Python consumers to `TurbomindEngineConfig` and weights

**Files:** `lmdeploy/turbomind/turbomind.py`, `lmdeploy/turbomind/deploy/load_context.py`, `lmdeploy/turbomind/deploy/target_model/base.py`

| Consumer | Current read | New source |
|---|---|---|
| `turbomind.py` `self.session_len` | `config.session_len` | `engine_config.session_len` |
| `turbomind.py` embedding prep | `config.model_config.data_type` | `engine_config.dtype` |
| `turbomind.py` grammar compilation | `config.model_config.vocab_size` | Save `vocab_size` from spec during `_from_hf()` as `self._vocab_size = spec._vocab_size` (pybind does not expose vocab_size) |
| `LoadContext._cpp_dtype()` | `self.model_config.data_type` | Accept `dtype: str` parameter; callers pass `engine_config.dtype` |
| `BaseOutputModel` | `model_config.attn_tp_size` etc. | Store from `engine_config` directly |

`TurboMindInstance` currently receives `self.config` (TurbomindModelConfig) and `self.tm_model.config_dict` (plain dict). After migration, it reads `data_type` from `self.tm_model.engine_config.dtype` and `vocab_size` from `self.tm_model._vocab_size`.

### PR 3: Remove `to_legacy_config()` and `_copy_*_fields()`

**Files:** `lmdeploy/turbomind/deploy/spec.py`, `lmdeploy/turbomind/deploy/target_model/base.py`, `lmdeploy/turbomind/deploy/converter.py`

Remove from `spec.py`:
- `to_legacy_config()` — only served YAML serialization
- `_copy_template_fields()` — copies C++ config template fields into Python `ModelConfig`
- `_copy_orchestration_fields()` — copies scalar fields into `ModelConfig`
- `_copy_perlayer_fields()` — copies per-layer lists into `ModelConfig`

Keep `_build_attention_config()` — still needed for `attention_config` YAML section.

Update `BaseOutputModel.finalize_config()`:
- Call `spec._build_attention_config()` directly instead of `spec.to_legacy_config()`
- No longer produces or installs a `ModelConfig`

Update `converter.py`:
- Stop creating empty `TurbomindModelConfig` and populating `model_config` fields
- Only build `AttentionConfig` for the YAML

### PR 4: Remove `ModelConfig` dataclass

**Files:** `lmdeploy/turbomind/deploy/config.py`, `lmdeploy/turbomind/deploy/converter.py`

- Delete `ModelConfig` class (~45 fields)
- Delete `ModelConfig.verify()` — no longer needed
- Simplify `TurbomindModelConfig`: remove `model_config` field, keep only `attention_config` and `lora_config`
- Simplify `TurbomindModelConfig.update_from_engine_config()`: remove model_config patching (5 fields), keep attention_config patching (`cache_block_seq_len`, `use_logn_attn`, `rope_scaling_factor` → rope_param)
- Remove convenience properties: `session_len`, `group_size`, `vocab_size` (consumers migrated in PR 2)
- Simplify `config_to_dict()` — no longer handles `ModelConfig`
- Remove `config_from_dict()` usage for `ModelConfig`

## Files Changed (all PRs)

| File | PR | Change |
|---|---|---|
| `lmdeploy/turbomind/turbomind.py` | 1,2 | Stop serializing model_config; migrate consumers to engine_config/weights |
| `lmdeploy/turbomind/deploy/config.py` | 1,4 | to_dict drops model_config; delete ModelConfig class; simplify TurbomindModelConfig |
| `lmdeploy/turbomind/deploy/spec.py` | 3 | Remove to_legacy_config and _copy_*_fields methods |
| `lmdeploy/turbomind/deploy/target_model/base.py` | 2,3 | Migrate consumers; simplify finalize_config |
| `lmdeploy/turbomind/deploy/load_context.py` | 2 | Accept dtype directly instead of model_config |
| `lmdeploy/turbomind/deploy/converter.py` | 3,4 | Stop populating ModelConfig; simplify construction |

## Scope

Python-side only. No C++ changes. No behavioral changes — model output is identical. Each PR leaves the codebase in a working, testable state.
