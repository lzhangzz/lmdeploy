# Unify reorder_rotary_emb: Single Entry Point for Linear and Tensor

## Problem

`reorder_rotary_emb` (operates on `torch.Tensor`) and `reorder_rotary_emb_linear`
(operates on `Linear`) are separate functions that spec authors must choose
between based on the input type. Every call site already knows `head_dim` and
`rope_dim`; the only reason for two names is the input type distinction.

## Design

Merge both into a single public function `reorder_rotary_emb` that accepts
either a `Linear` or a `torch.Tensor` and does the right thing.

### Naming

- **`_reorder_rotary_emb(x, head_dim, rope_dim)`** — private helper.
  Contains the current element-level interleave-transpose logic (partial RoPE
  vs full RoPE code paths). No "linear" or "tensor" in the name.

- **`reorder_rotary_emb(x, head_dim, rope_dim, *, data_type=None)`** — public
  function. Checks `isinstance(x, Linear)`:
  - **Linear path**: raises `TypeError` if `data_type is None` (dequantization
    requires it). Then the quantization-aware logic (block alignment check,
    dequantization fallback, per-tensor iteration calling
    `_reorder_rotary_emb` for each). This is the current
    `reorder_rotary_emb_linear` body inlined.
  - **Otherwise**: delegates to `_reorder_rotary_emb`.

- **`reorder_rotary_emb_linear`** is deleted entirely.

### Internal logic (unchanged)

No behavioral changes. The core interleave-transpose and the quantization-aware
block-shuffle logic remain identical to today — only the API surface changes.

### Call site changes

In `qwen3_spec.py`, `qwen3_5_spec.py`, `gpt_oss_spec.py`:

- Replace all `reorder_rotary_emb_linear(x, ...)` calls with
  `reorder_rotary_emb(x, ...)`.
- Replace all `reorder_rotary_emb(x, ...)` calls on tensors with the same
  `reorder_rotary_emb(x, ...)` (no change needed beyond import cleanup).
- Remove `reorder_rotary_emb_linear` from imports in all three files.

### Files changed

| File | Change |
|------|--------|
| `lmdeploy/turbomind/deploy/source_model/utils.py` | Rename tensor function to `_reorder_rotary_emb`, inline Linear logic into `reorder_rotary_emb`, delete `reorder_rotary_emb_linear` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Update imports, unify calls to `reorder_rotary_emb` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Update imports, unify calls to `reorder_rotary_emb` |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Update imports, unify calls to `reorder_rotary_emb` |

## Testing

Run existing model tests via `scripts/test_turbomind_model.py` for Qwen3 and
GPT-OSS models to confirm no behavioral regression. The refactor is purely
structural — no new test cases needed.
