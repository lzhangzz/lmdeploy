# Spec-Driven Loading Cleanup Design

Date: 2026-04-15
Status: Draft

## Problem

The spec-driven loading feature (spec `2026-04-15-builder-driven-loading-design`) has been implemented through Tasks 1-8, but the cleanup Tasks 9-10 were never completed. This leaves:

1. **Duplicate `SplitSide`** enum in both `spec.py` and `builder.py`, with specs importing from both and aliasing one as `BSplitSide`
2. **Residual cross-module dependencies** — `builder.py` still imports `_ATTN_TP_RULES` from `commit.py` and `fuse_ffn_linears` from `transforms.py`
3. **Dead files** — `distributor.py` and `transforms.py` are only used by the legacy fallback path in `TextModelLoader`
4. **Oversized `TextModelLoader`** — 365 lines instead of the target ~20, dominated by legacy `_process_*` methods
5. **Monolithic spec methods** — all four specs use `_build_*` private helpers that couple builder creation with parent assignment, instead of the factory-method pattern from the original design

## Design

### 1. Consolidate SplitSide

Delete `SplitSide` from `spec.py`. Keep only the definition in `builder.py`. Update all importers:

| File | Current import | New import |
|------|---------------|------------|
| `text_model_loader.py:15` | `from .spec import SplitSide` | `from .builder import SplitSide` |
| `commit.py:10` | `from .spec import SplitSide` | `from .builder import SplitSide` |
| `qwen3_5_spec.py:18` | `from ..spec import TextModelSpec, SplitSide` | `from ..spec import TextModelSpec` + `from ..builder import SplitSide` |
| `gpt_oss_spec.py:22` | `from ..spec import TextModelSpec, SplitSide` | `from ..spec import TextModelSpec` + `from ..builder import SplitSide` |
| `qwen3_5_spec.py:276` | `from ..builder import Builder, SplitSide as BSplitSide` | Remove — no longer needed |
| `qwen3_spec.py:37` | `from ..builder import ... SplitSide` | Already correct |
| `glm4_moe_lite_spec.py:48` | `from ..builder import ... SplitSide` | Already correct |

Remove all `BSplitSide` aliasing and `BSplitSide(x.value)` conversions. There is only one `SplitSide` now.

### 2. Absorb Remaining Dependencies into builder.py

**2a. Move `_ATTN_TP_RULES` into `AttentionBuilder`**

The `_ATTN_TP_RULES` table in `commit.py` maps attention linear names to TP split rules. Merge these into `AttentionBuilder` as a class-level constant. The `add_linear` method becomes self-contained — no import from `commit.py`.

**2b. Move `fuse_ffn_linears` and `_should_fuse_silu` into builder.py**

Copy these functions from `transforms.py` into `builder.py` as module-level functions. After this, `builder.py` has zero imports from `commit.py` or `transforms.py`.

**2c. Add `LinearBuilder`**

A lightweight builder for standalone linear layers (embeddings, lm_head). Provides `set_weight(tensor, split_side, model_dtype)` method. This replaces the ad-hoc `Builder` + `_commit_linear` usage in the spec `token_embeds()`/`lm_head()` factory methods.

```python
class LinearBuilder(Builder):
    def set_weight(self, tensor, split_side=None, model_dtype=None):
        linear = Linear({'weight': tensor})
        self._commit_linear('weight', linear, split_side, model_dtype)
```

### 3. Delete Legacy Files and Gut TextModelLoader

**3a. Simplify `TextModelLoader` to ~20 lines**

Remove all `_process_*` and `_load_*` methods. Remove the `NotImplementedError` try/except fallback. All four specs implement `model()`.

```python
class TextModelLoader:
    def __init__(self, model):
        handles = [model.root(gpu) for gpu in range(model.gpu_count)
                   if model.root(gpu) is not None]
        contexts = [model.context(gpu) for gpu in range(model.gpu_count)]
        self._handles = handles
        self._contexts = contexts

    def __call__(self, layer, spec):
        spec._contexts = self._contexts
        spec._root_handles = self._handles
        spec.configure(spec._config)
        spec.model()
        return 1
```

Remove imports of `Distributor`, `fuse_ffn_linears`, `SplitSide`, and all `commit` functions.

**3b. Keep `target_model/base.py` export() as-is**

The existing delegation through `TextModelLoader.__call__` works fine with the single-entry `readers()`. No changes needed.

**3c. Delete `distributor.py`**

No imports remain after TextModelLoader simplification.

**3d. Delete `transforms.py`**

`fuse_ffn_linears` moved to `builder.py`. No other imports remain.

**3e. Delete `commit.py`**

After absorbing `_ATTN_TP_RULES` and all commit functions into `builder.py`, verify no other imports exist, then delete. If `commit.py` has other symbols still imported somewhere, keep only those symbols.

**3f. Clean `spec.py`**

- Remove `SplitSide` enum
- Remove `model()` `NotImplementedError` default (or keep as safety net — harmless either way)
- Keep all weight-reading helpers (`merge_qkv_linear`, `_read_linear`, etc.)

### 4. Refactor Specs to Factory Methods

All four specs are refactored from private `_build_*(parent, layer, mc, dtype, tp, ranks, contexts)` helpers to public factory methods that return builders. The `model()` method assigns builders to parents.

**Common base pattern on `TextModelSpec`:**

```python
def model(self):
    self.configure(self._config)
    root = TextModelBuilder(self._root_handles, self._contexts)
    root.tok_embeddings = self.token_embeds()
    root.norm = self.norm('model.norm')
    root.output = self.lm_head()
    root.layers = self.layers('model.layers')

def norm(self, pfx) -> NormBuilder | None:
    tensor = self._get(f'{pfx}.weight')
    if tensor is None:
        return None
    return NormBuilder(NormConfig(dim=tensor.shape[-1], ...),
                       self._contexts).set_weight(tensor)

def attn(self, pfx, layer) -> AttentionBuilder:
    ...

def ffn(self, pfx, layer=None) -> FfnBuilder:
    ...

def moe(self, pfx, layer) -> MoeBuilder:
    # Reuses self.ffn() for each expert
    ...

def layers(self, pfx) -> ModuleListBuilder:
    m = ModuleListBuilder(ModuleListConfig(), self._contexts)
    for i in range(self.num_layers):
        d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
        d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm')
        d.attention = self.attn(f'{pfx}.{i}.self_attn', layer=i)
        d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm')
        if self.num_experts(i) > 0:
            d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', layer=i)
        else:
            d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', layer=i)
        m[str(i)] = d
    return m
```

**Key design decisions:**
- Factory methods read `_config`, `_contexts`, `_root_handles` from `self` — no long parameter lists
- `moe()` calls `self.ffn()` for each expert — no duplicate weight-reading logic
- Each spec overrides only the factory methods that differ from the base pattern
- Weight-reading helpers (`_read_linear`, `_read_ffn_linears`, `moe_gate`, etc.) stay unchanged

**Per-spec differences:**

**Qwen3Spec:** Straightforward override of `attn()` (with qk_norm), `ffn()`, `moe()` (optional), `token_embeds()`, `lm_head()`.

**GptOssSpec:** `attn()` handles attention sinks via `add_param`. `moe()` uses packed expert tensors and gate/up deinterleaving. No dense FFN — `ffn()` may not be needed.

**Qwen3_5Spec:** Has both `attn()` and `linear_attn()` factory methods (GDN layers dispatched per-layer). `norm()` applies zero-centered RMSNorm. `moe()` includes shared expert gate and packed experts.

**Glm4MoeLiteSpec:** `attn()` uses `add_linear` per projection for MLA. `moe()` includes score correction bias. Dense first-k layers handled in `layers()`.

### 5. `TextModelSpec` base class updates

Add default `layers()` and `norm()` implementations to the base class since the pattern is nearly identical across all four specs. Specs override only when they differ:

```python
class TextModelSpec(ABC):
    # ... existing fields ...

    def norm(self, pfx):
        """Default norm factory. Override for custom norm behavior."""
        tensor = self._get(f'{pfx}.weight')
        if tensor is None:
            return None
        from .builder import NormBuilder
        from .module_configs import NormConfig
        cfg = NormConfig(dim=tensor.shape[-1])
        return NormBuilder(cfg, self._contexts).set_weight(tensor)
```

## File Changes

### Files deleted

| File | Reason |
|------|--------|
| `distributor.py` | Only used by legacy TextModelLoader path |
| `transforms.py` | Absorbed into builder.py |
| `commit.py` | Absorbed into builder.py |

### Files modified

| File | Change |
|------|--------|
| `builder.py` | Absorb `_ATTN_TP_RULES`, `fuse_ffn_linears`, `_should_fuse_silu`. Add `LinearBuilder`. Remove imports from `commit.py`/`transforms.py`. |
| `spec.py` | Remove `SplitSide` enum. Add default `norm()`, `layers()` implementations. |
| `text_model_loader.py` | Gut to ~20 lines. Remove all `_process_*`/`_load_*` methods. Remove legacy imports. |
| `qwen3_spec.py` | Refactor `_build_*` to factory methods. Remove `BSplitSide` if present. |
| `gpt_oss_spec.py` | Refactor `_build_*` to factory methods. Remove `BSplitSide` aliasing. |
| `qwen3_5_spec.py` | Refactor `_build_*` to factory methods (including `linear_attn()`). Remove `SplitSide` from spec import, remove `BSplitSide` aliasing. |
| `glm4_moe_lite_spec.py` | Refactor `_build_*` to factory methods. |

### Files unchanged

| File | Reason |
|------|--------|
| `target_model/base.py` | Delegation through TextModelLoader works fine |
| `source_model/base.py` | Single-batch `readers()` already correct |
| `loader.py` | `all_items()` already correct |
| C++ files | No changes needed |
