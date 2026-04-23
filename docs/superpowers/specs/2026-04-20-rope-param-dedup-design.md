# Deduplicate RoPE Parameter Copying

## Problem

Four spec files contain identical 15-line blocks that copy `self._rope` (Python `RopeParam`) field-by-field to `_attn_cfg.rope` (C++ config), including type-specific branching for yarn/llama3/mrope.

## Solution

Add `_apply_rope(target)` method to `TextModelSpec` base class (`spec.py`). Replaces all 4 duplicated blocks with a single call.

### Method signature

```python
def _apply_rope(self, rope_cfg):
    """Copy self._rope fields into a C++ rope config object."""
```

### Field mapping (unchanged)

| `self._rope` field | `rope_cfg` field | Condition |
|---|---|---|
| `.type` (via `rope_type_to_int`) | `.type` | Always |
| `.base` | `.base` | Always |
| `.dim` | `.dim` | Always |
| `.factor` | `.factor` | Always |
| `self._max_position_embeddings` | `.max_position_embeddings` | Always |
| `.attention_factor` | `.yarn_attention_factor` | `type == 'yarn'` |
| `.beta_fast` | `.yarn_beta_fast` | `type == 'yarn'` |
| `.beta_slow` | `.yarn_beta_slow` | `type == 'yarn'` |
| `.low_freq_factor` | `.llama3_low_freq_factor` | `type == 'llama3'` |
| `.high_freq_factor` | `.llama3_high_freq_factor` | `type == 'llama3'` |
| `.original_max_position_embeddings` | `.llama3_original_max_position_embeddings` | `type == 'llama3'` |
| `.mrope_section` | `.mrope_section` | `type == 'mrope'` |

### Files changed

1. **`spec.py`** — add `_apply_rope(self, target)` method, import `rope_type_to_int`
2. **`qwen3_spec.py`** — replace lines 54-68 with `self._apply_rope(self._attn_cfg.rope)`, remove `rope_type_to_int` import
3. **`qwen3_5_spec.py`** — replace lines 69-83 with `self._apply_rope(self._attn_cfg.rope)`, remove `rope_type_to_int` import
4. **`gpt_oss_spec.py`** — replace lines 60-74 with `self._apply_rope(self._attn_cfg.rope)`, remove `rope_type_to_int` import
5. **`glm4_moe_lite_spec.py`** — replace lines 88-102 with `self._apply_rope(self._attn_cfg.rope)`, remove `rope_type_to_int` import

### Testing

Run `scripts/test_turbomind_model.py` against a RoPE-using model (e.g. Qwen3) to verify identical behavior.
