# Spec Hierarchy Cleanup Design

Date: 2026-04-15
Status: Draft

## Problem

The spec hierarchy methods (`attn()`, `ffn()`, `moe()`, etc.) still delegate weight reading to flat helper methods on the base class (`attn_params()`, `moe_gate()`, `moe_ffn_linears()`, etc.). These helpers break the hierarchy principle: each method should read its own weights from its prefix, and prefixes should be appended along the call path.

Additionally:
- All four spec files use local imports inside factory methods instead of top-level imports
- `ffn()` accepts a `linears` parameter, violating self-containment
- `spec.py` contains builder-side logic (QKV fusion, dequant) that belongs in `builder.py`
- `permute_v2` / `_permute_qk_tensors` are on the base class but are model-specific

## Design

### 1. Strip `TextModelSpec` to bare minimum

Remove every empty placeholder and compound helper. The base class becomes:

```python
class TextModelSpec(ABC):
    params: dict[str, torch.Tensor]

    # configure() fields (set by TextModelLoader)
    _attn_tp: int = 1
    _permute_qk: bool = True
    _repeat_kv: int = 0
    _head_dim: int = 0
    _rope_dim: int = 0
    _attn_output_gate: bool = False
    _kv_head_num: int = 0

    # Injected by TextModelLoader
    _contexts: list = None
    _root_handles: list = None

    # NOTE: _linear_qkv_split, _gdn_qkv_split, _mc, _attn_cp, _mlp_tp,
    # _attn_ranks, _mlp_ranks, _repeat_kv are still injected by TextModelLoader
    # but are NOT declared on the base class.  Each spec accesses them directly.
    # This is temporary until TextModelLoader is further simplified.

    def model(self): ...                           # override in subclass
    def configure(self, cfg: SpecAttnConfig): ...  # called by TextModelLoader
    def _get(self, key) -> Tensor | None: ...      # raw tensor lookup
    def _linear(self, pfx) -> Linear | None: ...   # read one Linear weight
    @abstractmethod
    def model_info(self) -> dict: ...
    def num_experts(self, layer) -> int: return 0
```

**Deleted methods:** `_read_ffn_linears`, `attn_linears`, `_read_attn_linears`, `ffn_linears`, `moe_ffn_linears`, `linear_attn_linears`, `_read_linear_attn_linears`, `attn_norm`, `ffn_norm`, `tok_embeddings`, `output_weight`, `norm_weight`, `attn_params`, `attn_norm_children`, `moe_params`, `moe_gate`, `linear_attn_params`, `linear_attn_norm_children`, `_permute_qk_tensors`.

**Renamed:** `_read_linear` -> `_linear`.

### 2. Move builder-side logic from `spec.py` to `builder.py`

These free functions are builder concerns (fusion, dequant, TP interleaving) and must not live in spec.py:

- `merge_qkv_v2`, `merge_qkvg_v2` -> `builder.py`
- `merge_qkv_linear` -> `builder.py` (already imported from spec.py by `AttentionBuilder.add_qkv_proj`)
- `fuse_gdn_in_proj` -> `builder.py`
- `_ensure_compatible_formats`, `_dequant_linear` -> `builder.py`
- `_block_ops_need_dequant`, `_tp_interleave_tensor` -> `builder.py`
- `_GDN_IN_PROJ_KEYS` -> `builder.py`

After this, `spec.py` has zero fusion/dequant logic. The builder is self-contained.

### 3. Move model-specific transforms to each spec

`permute_v2`, `permute_v2_partial` are renamed to `reorder_rotary_emb` and moved to each spec that uses them:

- `qwen3_spec.py` -- uses `reorder_rotary_emb` in `attn()` for QK norm permutation
- `gpt_oss_spec.py` -- same
- `qwen3_5_spec.py` -- same
- `glm4_moe_lite_spec.py` -- may not need it (MLA attention)

Each spec defines `reorder_rotary_emb` as a module-level function. `builder.py` defines its own copy since `merge_qkv_linear` uses it internally for QKV merge permutation. No shared module -- the function is small (~5 lines) and duplication keeps each file self-contained.

### 4. Each hierarchy method reads its own weights

Every factory method reads weights directly using the prefix it receives. No delegation to separate weight-reading methods.

**`attn(pfx, layer)`** -- reads Q, K, V, O via `self._linear(f'{pfx}.q_proj')` etc. Reads qk norm via `self._get(f'{pfx}.{x}_norm.weight')`. Applies `reorder_rotary_emb` if needed. Passes `Linear` objects to builder.

**`ffn(pfx, layer, inter_size=None, fused_moe=False)`** -- reads gate/down/up via `self._linear(f'{pfx}.gate_proj')` etc. No `linears` parameter. Each spec's `ffn()` knows its own weight naming (standard HF naming, packed experts, etc.).

**`moe(pfx, layer)`** -- reads gate via `self._linear(f'{pfx}.gate')`. Reads extra params inline. Calls `self.ffn(f'{pfx}.experts.{e}', layer, ...)` for each expert -- no pre-loaded linears passed.

**`norm(pfx)`** -- reads `self._get(f'{pfx}.weight')`.

**`token_embeds(pfx)`** -- reads `self._get(f'{pfx}.weight')`.

**`lm_head(pfx)`** -- reads `self._get(f'{pfx}.weight')`.

**`root_norm(pfx)`** -- reads `self._get(f'{pfx}.weight')`.

### 5. Move all imports to top of file

All four spec files move their local imports (`from ..builder import ...`) to the file's top-level import block.

### 6. `model()` passes prefixes to all methods

```python
def model(self):
    root = TextModelBuilder(self._root_handles, self._contexts)
    root.tok_embeddings = self.token_embeds('model.embed_tokens')
    root.norm = self.root_norm('model.norm')
    root.output = self.lm_head('lm_head')
    root.layers = self.layers('model.layers')
```

## File Changes

### Files modified

| File | Change |
|------|--------|
| `spec.py` | Strip to bare `TextModelSpec`. Remove all helpers, placeholders, fusion functions. Rename `_read_linear` to `_linear`. |
| `builder.py` | Absorb `merge_qkv_linear`, `merge_qkv_v2`, `merge_qkvg_v2`, `fuse_gdn_in_proj`, `_ensure_compatible_formats`, `_dequant_linear`, `_block_ops_need_dequant`, `_tp_interleave_tensor`, `_GDN_IN_PROJ_KEYS`. Remove `from .spec import merge_qkv_linear`. |
| `qwen3_spec.py` | Inline all weight reads. Add `reorder_rotary_emb`. Remove `linears` from `ffn()`. Move imports to top. Delete flat method overrides. |
| `gpt_oss_spec.py` | Same. Handle packed expert weights inline in `ffn()`. |
| `qwen3_5_spec.py` | Same. Handle GDN inline in `linear_attn()`. Handle shared expert and packed experts in `moe()`. |
| `glm4_moe_lite_spec.py` | Same. Handle MLA inline in `attn()`. |

### Files unchanged

| File | Reason |
|------|--------|
| `text_model_loader.py` | Already gutted, works as-is |
| `target_model/base.py` | Delegation unchanged |
| `source_model/base.py` | Unchanged |
