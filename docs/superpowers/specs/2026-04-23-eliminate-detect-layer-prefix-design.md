# Eliminate detect_layer_prefix

## Problem

`detect_layer_prefix` is a runtime function that sniffs checkpoint weight keys to
decide whether a model's layers live under `model.layers` or
`model.language_model.layers`. This is an anti-pattern because:

1. **Runtime detection of static knowledge** — a spec knows its own checkpoint
   structure; it shouldn't need to inspect weights to find it.
2. **Dual-phase state** — `_layer_prefix`, `_embed_key`, `_norm_key` are set to
   defaults in `__init__`, then potentially overwritten in `set_params`. The spec
   has invalid state between construction and weight loading.
3. **`_pin_layer_prefix` escape hatch** — 3 of 4 specs set `_pin_layer_prefix = True`
   to opt out of re-detection. The abstraction is overhead for the common case.
4. **Defaults by coincidence** — `_parse_base` sets `model.*` defaults that happen
   to match most specs. Programming by accident is not allowed.

## Design

Each spec explicitly declares its own prefix in `__init__`. No defaults, no
detection, no dual-phase state.

### Changes

**Delete `detect_layer_prefix`** from `source_model/utils.py`.

**Delete `_pin_layer_prefix`** class variable from `TextModelSpec`.

**Remove prefix fields from `_parse_base`** — `_layer_prefix`, `_embed_key`,
`_norm_key` are not "parsed from config." They are structural knowledge each
spec owns.

**Simplify `set_params`** — remove the re-detection block. The method becomes
just `self.params = params`.

**Each spec declares its prefix explicitly in `__init__`:**

| Spec | _layer_prefix | _embed_key | _norm_key |
|------|--------------|------------|-----------|
| gpt_oss_spec | `model.layers` | `model.embed_tokens.weight` | `model.norm.weight` |
| qwen3_spec | `model.layers` | `model.embed_tokens.weight` | `model.norm.weight` |
| glm4_moe_lite_spec | `model.layers` | `model.embed_tokens.weight` | `model.norm.weight` |
| qwen3_5_spec | `model.language_model.layers` | `model.language_model.embed_tokens.weight` | `model.language_model.norm.weight` |

The three `model.layers` specs already declare these values in their `__init__`.
The change for them is removing `_pin_layer_prefix = True` (no longer needed).

`qwen3_5_spec` currently relies on the base-class re-detection. After this
change it declares its prefix explicitly: `model.language_model.*`. All Qwen3.5
models are multimodal, so this is always correct.

### Files touched

- `lmdeploy/turbomind/deploy/source_model/utils.py` — delete `detect_layer_prefix`
- `lmdeploy/turbomind/deploy/spec.py` — delete `_pin_layer_prefix`, remove prefix
  fields from `_parse_base`, simplify `set_params`
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — delete
  `_pin_layer_prefix = True`
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — delete
  `_pin_layer_prefix = True`
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — delete
  `_pin_layer_prefix = True`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — add explicit
  `_layer_prefix`, `_embed_key`, `_norm_key` in `__init__`
