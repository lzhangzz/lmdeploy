# add_projections Refactor Design

## Goal

Decompose `MLABuilder._fold_and_pad` / `_fold_and_pad_hf` (90-line combined monolith) into a pipeline of named standalone functions operating in TM layout, mirroring the pattern established by the `add_qkv_proj` and `add_input_projections` refactorings. Remove all defensive guards.

## Background

After the `add_qkv_proj` and `add_input_projections` refactorings, attention QKV fusion and GDN input projection fusion follow a clean pipeline of standalone functions. Each function has a clear contract: inputs in, outputs out, no in-place dict mutation.

`MLABuilder.add_projections` currently delegates to `_fold_and_pad` which transposes all weight tensors to HF layout `[out, in]`, calls `_fold_and_pad_hf`, then transposes back. The inner method is a 67-line monolith that handles: config extraction, condition checks, splitting kv_b into key/value components, folding kc into q_b (matmul), folding vc into wo (matmul), and padding wo.

## Changes

### 1. New `fold_kv_b` function

Rewrites the fold math in TM layout `[in, out]` directly, eliminating the transpose-to-HF-and-back dance.

```python
def fold_kv_b(q_b: Linear, kv_b: Linear, wo: Linear, *,
              cfg) -> tuple[Linear, Linear]:
    """Fold kv_b into q_b and wo. Returns (q_b_folded, wo_folded).

    Splits kv_b into key-compressed (kc) and value-compressed (vc) components.
    Folds kc into q_b via matmul (q_nope @ kc^T per head).
    Folds vc into wo via matmul (vc @ wo per head).

    All arithmetic in TM layout [in, out].
    """
```

TM-layout fold math (derived from HF equivalents):

**Fold kc into q_b** — per head:
- `q_nope`: `[kv_lora_rank, orig_qk_nope_dim]`
- `kc`: `[kv_lora_rank, orig_qk_nope_dim]`
- `result = q_nope @ kc^T` = `[kv_lora_rank, kv_lora_rank]`
- Reassemble: cat with q_rope `[kv_lora_rank, qk_rope_dim]` → `[kv_lora_rank, size_per_head]`

**Fold vc into wo** — per head:
- `vc`: `[kv_lora_rank, orig_v_head_dim]`
- `wo_h`: `[orig_v_head_dim, hidden_size]`
- `result = vc @ wo_h` = `[kv_lora_rank, hidden_size]`

Both use `torch.bmm` with `head_num` as batch dim.

### 2. New `pad_wo_input` function

Simple shape manipulation in TM layout. Pads wo's input dim from `head_num * kv_lora_rank` to `head_num * size_per_head`.

```python
def pad_wo_input(wo: Linear, *, cfg) -> Linear:
    """Pad wo input dim from head_num * cur_dim to head_num * size_per_head."""
```

### 3. Revised `add_projections`

```python
def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                    wo):
    """Apply MLA fold+pad, then commit each projection."""
    q_b_proj, wo = fold_kv_b(q_b_proj, kv_b_proj, wo, cfg=self.config)
    wo = pad_wo_input(wo, cfg=self.config)

    model_dtype = self.config.data_type
    for name, lin, side in [
        ("q_a_proj", q_a_proj, SplitSide.OUTPUT),
        ("q_b_proj", q_b_proj, SplitSide.OUTPUT),
        ("kv_a_proj", kv_a_proj, SplitSide.OUTPUT),
        ("wo", wo, SplitSide.INPUT),
    ]:
        self._commit_linear(name, lin, split_side=side,
                            model_dtype=model_dtype)
```

### 4. Revised `add_norms`

```python
def add_norms(self, *, q_a_norm, kv_a_norm, data_type):
    self._add_norm_child('q_a_layernorm', q_a_norm, data_type=data_type)
    self._add_norm_child('kv_a_layernorm', kv_a_norm, data_type=data_type)
```

`data_type` is now required (was `data_type=None`).

### 5. Remove `LMDEPLOY_MLA_FOLD` env var

The env var in `glm4_moe_lite_spec.py:model_info()` that can disable folding is dead code. The fold always happens for MLA models. Remove the env var, the `disable_mla_fold` variable, the `os` import, and the conditional branches in `model_info()`.

## Code to Delete

| What | File | Why |
|------|------|-----|
| `_MLA_TP_RULES` | `mla.py` | Inlined into commit loop |
| `_fold_and_pad` | `mla.py` | Replaced by standalone functions |
| `_fold_and_pad_hf` | `mla.py` | Replaced by TM-layout `fold_kv_b` |
| `LMDEPLOY_MLA_FOLD` env var | `glm4_moe_lite_spec.py` | Dead code |
| `import os` | `glm4_moe_lite_spec.py` | Only used for env var |

## Scope

Internal refactoring of `MLABuilder` pipeline only. No spec interface changes (same kwargs for `add_projections`). No behavioral changes. Two new standalone functions (`fold_kv_b`, `pad_wo_input`) replace three methods (`_fold_and_pad`, `_fold_and_pad_hf`, inline pad).
