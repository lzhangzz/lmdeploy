# Spec Hierarchy Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor model specs so each hierarchy method reads its own weights from its prefix, strip the base class to bare minimum, and move builder-side logic out of spec.py.

**Architecture:** Bottom-up: first move builder-side functions to builder.py, then refactor each spec file, then strip spec.py. This avoids broken intermediate states.

**Tech Stack:** Python, TurboMind weight loading pipeline

---

## File Structure

| File | Responsibility |
|------|----------------|
| `lmdeploy/turbomind/deploy/builder.py` | Absorbs QKV fusion, dequant, TP interleave from spec.py |
| `lmdeploy/turbomind/deploy/spec.py` | Stripped to bare `TextModelSpec` with `_get`, `_linear`, `configure`, `num_experts`, `model_info` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Canonical spec — reads weights inline, top-level imports, no `linears` param |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Same pattern. Packed expert handling inlined in `ffn()` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Same pattern. GDN inlined in `linear_attn()`, shared expert + packed experts in `moe()` |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Same pattern. MLA inlined in `attn()` |

---

### Task 1: Move builder-side functions from spec.py to builder.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py` (imports + insert functions before `AttentionBuilder` at line ~700)
- Modify: `lmdeploy/turbomind/deploy/spec.py` (delete moved functions)

**Functions to move** (copy from spec.py, then delete from spec.py):

From spec.py lines 15-51: `permute_v2`, `permute_v2_partial` — rename to internal names since builder.py will have its own copy.

From spec.py lines 54-86: `merge_qkv_v2`, `merge_qkvg_v2`

From spec.py lines 349-363: `_dequant_linear`, `_ensure_compatible_formats`

From spec.py lines 370-398: `_GDN_IN_PROJ_KEYS`, `_block_ops_need_dequant`

From spec.py lines 401-509: `merge_qkv_linear`

From spec.py lines 512-516: `_tp_interleave_tensor`

From spec.py lines 519-621: `fuse_gdn_in_proj`

- [ ] **Step 1: Add imports to builder.py**

Add `TRIVIAL_FORMAT` and `kind_map` imports to builder.py's top-level import block (after line 19):

```python
from .kind_map import TRIVIAL_FORMAT
```

- [ ] **Step 2: Add `reorder_rotary_emb` helper to builder.py**

Insert before `AttentionBuilder` class definition (before line ~700). This is builder.py's private copy:

```python
def _reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Reorder rotary embedding layout for TurboMind's RoPE kernel."""
    if rope_dim < head_dim:
        # Partial permutation: only interleave the rotary portion
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), head_num, head_dim)
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
        return x.reshape(orig_shape)
    else:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        return x.view(-1, head_num, 2, head_dim // 2).transpose(2, 3).reshape(x.shape)
```

- [ ] **Step 3: Copy all fusion functions from spec.py to builder.py**

Copy these functions verbatim from spec.py into builder.py, placed before `AttentionBuilder` (after the `_reorder_rotary_emb` from step 2):

1. `_dequant_linear(linear)` — add `TRIVIAL_FORMAT` import (already done in step 1)
2. `_ensure_compatible_formats(linears)`
3. `_block_ops_need_dequant(lin, head_dim, repeat_kv, attn_output_gate, permute_qk)`
4. `merge_qkv_v2(q, k, v, tp)`
5. `merge_qkvg_v2(q, k, v, gate, tp)`
6. `merge_qkv_linear(q, k, v, tp, head_dim, rope_dim, ...)` — update to call `_reorder_rotary_emb` instead of `permute_v2`/`permute_v2_partial`
7. `_tp_interleave_tensor(t, tp, d)`
8. `_GDN_IN_PROJ_KEYS` constant
9. `fuse_gdn_in_proj(la_linears, tp, qkv_split)`

**Important change in `merge_qkv_linear`:** Replace the two calls to `permute_v2` / `permute_v2_partial` with calls to `_reorder_rotary_emb`:

```python
# Old:
if rope_dim < head_dim:
    qt = permute_v2_partial(qt, head_dim, rope_dim)
    kt = permute_v2_partial(kt, head_dim, rope_dim)
else:
    qt = permute_v2(qt, head_dim)
    kt = permute_v2(kt, head_dim)

# New:
qt = _reorder_rotary_emb(qt, head_dim, rope_dim)
kt = _reorder_rotary_emb(kt, head_dim, rope_dim)
```

- [ ] **Step 4: Update `AttentionBuilder.add_qkv_proj` to use local import**

In builder.py, the local import at line 713:
```python
from .spec import merge_qkv_linear
```
Delete this line. The function is now in the same file.

- [ ] **Step 5: Delete the moved functions from spec.py**

Delete from spec.py:
- `permute_v2` (lines 15-25)
- `permute_v2_partial` (lines 28-51)
- `merge_qkv_v2` (lines 54-68)
- `merge_qkvg_v2` (lines 71-86)
- `_dequant_linear` (lines 349-355)
- `_ensure_compatible_formats` (lines 358-363)
- `_GDN_IN_PROJ_KEYS` (line 370)
- `_block_ops_need_dequant` (lines 373-398)
- `merge_qkv_linear` (lines 401-509)
- `_tp_interleave_tensor` (lines 512-516)
- `fuse_gdn_in_proj` (lines 519-621)

Also remove the now-unused import `TRIVIAL_FORMAT` from spec.py (line 7).

- [ ] **Step 6: Verify builder.py imports**

Run: `python -c "from lmdeploy.turbomind.deploy.builder import merge_qkv_linear, fuse_gdn_in_proj, _dequant_linear; print('OK')"`

Set `PYTHONPATH=${workspace_dir}/lmdeploy:${workspace_dir}/build/lib` first.

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(deploy): move QKV fusion and dequant from spec.py to builder.py"
```

---

### Task 2: Refactor qwen3_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

This is the canonical spec. All four specs follow the same pattern; the others build on this.

- [ ] **Step 1: Add `reorder_rotary_emb` module-level function**

Add at the top of the file (after imports, before the class):

```python
def reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Reorder rotary embedding layout for TurboMind's RoPE kernel."""
    if rope_dim < head_dim:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), head_num, head_dim)
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
        return x.reshape(orig_shape)
    else:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        return x.view(-1, head_num, 2, head_dim // 2).transpose(2, 3).reshape(x.shape)
```

- [ ] **Step 2: Move all imports to top-level**

Update the import block at the top of qwen3_spec.py to:

```python
from __future__ import annotations

import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _act_type_id, _cpp_dtype as _cd,
)
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

Then delete every local import inside factory methods (`from ..builder import ...` inside `model()`, `token_embeds()`, `root_norm()`, `lm_head()`, `norm()`, `attn()`, `ffn()`, `moe()`, `layers()`, `_cpp_dtype()`).

Update `_cpp_dtype()` to use the top-level `_cd` directly:

```python
def _cpp_dtype(self):
    return _cd(self._mc.data_type)
```

- [ ] **Step 3: Update `model()` to pass prefixes**

```python
def model(self):
    root = TextModelBuilder(self._root_handles, self._contexts)
    root.tok_embeddings = self.token_embeds('model.embed_tokens')
    root.norm = self.root_norm('model.norm')
    root.output = self.lm_head('lm_head')
    root.layers = self.layers('model.layers')
```

- [ ] **Step 4: Update `token_embeds(pfx)` to read weights inline**

Change signature from `token_embeds(self)` to `token_embeds(self, pfx)` and replace `emb = self.tok_embeddings()` with:

```python
def token_embeds(self, pfx):
    emb = self._get(f'{pfx}.weight')
    if emb is None:
        return None
    # ... rest unchanged (padding, LinearBuilder creation, etc.)
```

- [ ] **Step 5: Update `root_norm(pfx)` to read weights inline**

Change signature from `root_norm(self)` to `root_norm(self, pfx)` and replace `w = self.norm_weight()` with:

```python
def root_norm(self, pfx):
    w = self._get(f'{pfx}.weight')
    if w is None:
        return None
    # ... rest unchanged
```

- [ ] **Step 6: Update `lm_head(pfx)` to read weights inline**

Change signature from `lm_head(self)` to `lm_head(self, pfx)` and replace `output = self.output_weight()` with:

```python
def lm_head(self, pfx):
    tie = self.cfg.get("tie_word_embeddings", False)
    key = "model.embed_tokens.weight" if tie else f"{pfx}.weight"
    output = self._get(key)
    if output is None:
        return None
    # ... rest unchanged
```

- [ ] **Step 7: Update `attn(pfx, layer)` to inline weight reads**

Replace the calls to `self.attn_params(layer)` and `self.attn_norm_children(layer)` with inline reads:

```python
def attn(self, pfx, layer):
    q = self._linear(f"{pfx}.q_proj")
    k = self._linear(f"{pfx}.k_proj")
    v = self._linear(f"{pfx}.v_proj")
    o = self._linear(f"{pfx}.o_proj")

    if q is None and k is None and v is None and o is None:
        return None

    mc = self._mc
    tp = self._attn_tp
    dtype = self._cpp_dtype()

    window_size = 0
    ws_list = mc.window_size
    if ws_list and layer < len(ws_list):
        window_size = ws_list[layer]

    attn_cfg = AttentionConfig.from_model_config(
        mc, tp_size=tp, tp_rank=0, dtype=dtype,
        window_size=window_size,
        rope_dim=self._rope_dim,
        permute_qk=self._permute_qk,
        repeat_kv=self._repeat_kv)
    attn = AttentionBuilder(attn_cfg, self._contexts,
                            tp=tp, ranks=self._attn_ranks)

    if q is not None and k is not None and v is not None:
        attn.add_qkv_proj(q, k, v)
    if o is not None:
        attn.add_o_proj(o)

    # Inline qk norm (was attn_norm_children)
    q_norm = self._get(f"{pfx}.q_norm.weight")
    k_norm = self._get(f"{pfx}.k_norm.weight")
    if q_norm is not None and k_norm is not None:
        if self._permute_qk:
            q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope_dim)
            k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope_dim)
    if q_norm is not None or k_norm is not None:
        attn.add_qk_norm(q_norm, k_norm)

    return attn
```

Note: `self.attn_params(layer)` returned `{}` in qwen3_spec, so it had no extra params. No inline replacement needed — just delete the loop.

- [ ] **Step 8: Update `ffn(pfx, layer)` to remove `linears` param**

Change signature to remove `linears`:

```python
def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
```

Replace:
```python
if linears is None:
    linears = self._read_ffn_linears(pfx)
```
With:
```python
w1 = self._linear(f"{pfx}.gate_proj")
w3 = self._linear(f"{pfx}.up_proj")
w2 = self._linear(f"{pfx}.down_proj")
linears = {}
if w1 is not None: linears['w1'] = w1
if w3 is not None: linears['w3'] = w3
if w2 is not None: linears['w2'] = w2
```

- [ ] **Step 9: Update `moe(pfx, layer)` to inline weight reads**

Replace `self.moe_gate(layer)`, `self.moe_params(layer)`, and `self.moe_ffn_linears(layer, e)` with inline reads:

```python
def moe(self, pfx, layer):
    if self.num_experts(layer) <= 0:
        return None

    mc = self._mc
    tp = self._mlp_tp
    dtype = self._cpp_dtype()

    expert_num = 0
    en_list = mc.expert_num
    if en_list and layer < len(en_list):
        expert_num = en_list[layer]

    moe_cfg = MoeConfig.from_model_config(
        mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
        act_type=_act_type_id(mc.activation_type),
        fuse_silu=True, expert_num=expert_num)
    m = MoeBuilder(moe_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)

    # Inline gate read (was moe_gate)
    if self._n_experts > 0:
        gate_w = self._get(f'{pfx}.gate.weight')
        if gate_w is not None:
            gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
            m.add_gate('gate', Linear({"weight": gate_w}), model_dtype=dtype)

    # No moe_params for qwen3 — was empty

    expert_inter = mc.expert_inter_size or 0
    experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
    for e in range(self.num_experts(layer)):
        expert = self.ffn(
            f'{pfx}.experts.{e}', layer,
            inter_size=expert_inter, fused_moe=True)
        if expert is not None:
            experts[str(e)] = expert

    m.experts = experts
    return m
```

- [ ] **Step 10: Delete all flat method overrides**

Delete these methods from `Qwen3TextSpec` (lines 284-344):
- `ffn_linears()` (284-288)
- `moe_ffn_linears()` (292-294)
- `attn_norm()` (301-302)
- `ffn_norm()` (304-305)
- `attn_params()` (307-308)
- `attn_norm_children()` (310-320)
- `moe_gate()` (322-330)
- `moe_params()` (332-333)
- `tok_embeddings()` (335-336)
- `output_weight()` (338-341)
- `norm_weight()` (343-344)

Keep: `num_experts()` (296-297), `model_info()` (348-374), `Qwen3InputModel` class (377+).

Also change `self._read_linear(` to `self._linear(` everywhere in the file (rename per spec).

- [ ] **Step 11: Verify import**

Run: `python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3TextSpec; print('OK')"`

- [ ] **Step 12: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "refactor(deploy): refactor Qwen3Spec to self-contained hierarchy methods"
```

---

### Task 3: Refactor gpt_oss_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

Same pattern as Task 2. Key differences from qwen3_spec:

- **Has `reorder_rotary_emb`**: needed for attention (same as qwen3)
- **No qk norm**: `attn()` does not read qk norm weights
- **Has attention sinks**: `attn_params()` returns `{"sinks": ...}` — inline this
- **Packed expert weights**: `moe_ffn_linears()` deinterleaves packed expert tensors. This logic must be inlined into `moe()` or into a private helper on the spec class. The `ffn()` method for experts must handle the packed format.
- **Has `ffn_linears()`**: called by `ffn()` when `linears is None` and `self.ffn_linears(layer)` is non-empty. Inline the FFN weight read (standard HF naming).
- **`moe_gate()`**: returns gate with transpose, and shared_gate if present — inline into `moe()`

- [ ] **Step 1: Add `reorder_rotary_emb` module-level function**

Same as Task 2 Step 1.

- [ ] **Step 2: Move all imports to top-level**

Current top-level imports include `from ..builder import SplitSide, _cpp_dtype, _act_type_id, LinearBuilder`. Add all other builder imports used locally:

```python
from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _act_type_id, _cpp_dtype as _cd,
)
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

Delete all local imports inside factory methods.

- [ ] **Step 3: Update `model()` to pass prefixes**

Same as Task 2 Step 3 — pass `'model.embed_tokens'`, `'model.norm'`, `'lm_head'` as prefixes.

- [ ] **Step 4: Update `token_embeds(pfx)`, `root_norm(pfx)`, `lm_head(pfx)`**

Same pattern as Task 2 Steps 4-6 — read via `self._get(f'{pfx}.weight')` instead of calling `self.tok_embeddings()` etc.

- [ ] **Step 5: Update `attn(pfx, layer)` to inline weight reads**

Replace `self.attn_params(layer)` with inline reads. GptOss has attention sinks:

```python
# Inline attn_params: sinks
sinks = self._get(f'{pfx}.sinks')
if sinks is not None:
    attn.add_param('sinks', sinks)
```

Note: No `attn_norm_children` call for gpt_oss (it doesn't override it).

- [ ] **Step 6: Update `ffn(pfx, layer)` to remove `linears` param**

Remove `linears` parameter. Inline the FFN weight reading:

```python
def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
    w1 = self._linear(f"{pfx}.gate_proj")
    w3 = self._linear(f"{pfx}.up_proj")
    w2 = self._linear(f"{pfx}.down_proj")
    linears = {}
    if w1 is not None: linears['w1'] = w1
    if w3 is not None: linears['w3'] = w3
    if w2 is not None: linears['w2'] = w2
    if not linears:
        return None
    # ... rest unchanged
```

- [ ] **Step 7: Update `moe(pfx, layer)` to inline weight reads**

Inline `moe_gate()`: read `self._get(f'{pfx}.gate.weight')`, transpose, wrap in Linear. Also check for `shared_gate`.

Inline packed expert handling: the current `moe_ffn_linears()` deinterleaves packed expert tensors. Move this logic into `moe()` as a private inline operation, or keep as a private `_deinterleave_expert()` helper on the class. Then call `self.ffn(f'{pfx}.experts.{e}', layer, ...)` without passing linears.

**Key change**: For packed experts, `ffn()` needs to handle the non-standard weight layout. Since GptOss's `moe_ffn_linears()` builds a prefix from `_layer_prefix` and does custom deinterleaving, this logic should be inlined into `moe()` or kept as a private method `_read_expert_linears(pfx, expert_idx)`.

The simplest approach: keep the deinterleave logic as a private method `_deinterleave(linears)` on the spec class (not a base class method), and have `moe()` call it before passing to `ffn()`. But since `ffn()` should be self-contained, the cleanest approach is to override `ffn()` logic for packed experts directly:

```python
# In moe(), read expert weights inline
for e in range(self.num_experts(layer)):
    e_w1 = self._linear(f'{pfx}.experts.{e}.gate_proj')
    e_w3 = self._linear(f'{pfx}.experts.{e}.up_proj')
    e_w2 = self._linear(f'{pfx}.experts.{e}.down_proj')
    if e_w1 is not None: e_w1 = _deinterleave(e_w1)
    # ... etc.
```

Actually, GptOss's packed experts use a different format — the weights are packed across all experts. The `moe_ffn_linears()` method extracts and deinterleaves per-expert slices. Since this is complex, keep the deinterleave as a private method `_deinterleave_expert(self, pfx, expert_idx)` on the spec class, and call it from `moe()`.

- [ ] **Step 8: Delete all flat method overrides**

Delete these methods:
- `ffn_linears()` (327-328)
- `moe_ffn_linears()` (330-341)
- `attn_norm()` (346-348)
- `ffn_norm()` (350-352)
- `attn_params()` (354-360)
- `moe_gate()` (362-374)
- `moe_params()` (376-377)
- `tok_embeddings()` (379-380)
- `output_weight()` (382-385)
- `norm_weight()` (387-388)

Keep: `num_experts()` (343-344), `model_info()`, `GptOssInputModel`.

Rename `self._read_linear(` to `self._linear(` everywhere.

- [ ] **Step 9: Verify import**

Run: `python -c "from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec; print('OK')"`

- [ ] **Step 10: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(deploy): refactor GptOssSpec to self-contained hierarchy methods"
```

---

### Task 4: Refactor qwen3_5_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

Most complex spec. Key differences:

- **Has `linear_attn()` factory method** for GDN layers, dispatched per-layer in `layers()`
- **Has `_zero_centered()`** transform on norm weights
- **GDN fusion** (`fuse_gdn_in_proj`) was in spec.py, now in builder.py. `linear_attn()` needs to call builder-side fusion.
- **Has shared expert** in MoE — `moe()` creates both routed experts and shared expert
- **Has packed expert** handling similar to GptOss

- [ ] **Step 1: Add `reorder_rotary_emb` module-level function**

Same as Task 2 Step 1.

- [ ] **Step 2: Move all imports to top-level**

```python
from ..builder import (
    AttentionBuilder, Builder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _LINEAR_ATTN_TP_RULES, _act_type_id, _cpp_dtype as _cd,
    fuse_gdn_in_proj,
)
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, DeltaNetConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

Note: `fuse_gdn_in_proj` is now imported from `..builder` (moved in Task 1).

Delete all local imports inside factory methods.

- [ ] **Step 3: Update `model()` to pass prefixes**

Same as Task 2 Step 3.

- [ ] **Step 4: Update `token_embeds(pfx)`, `root_norm(pfx)`, `lm_head(pfx)`**

Same pattern as Task 2. For `root_norm`: read `self._get(f'{pfx}.weight')` and apply `self._zero_centered()`.

- [ ] **Step 5: Update `norm(pfx)` — applies `_zero_centered`**

```python
def norm(self, pfx):
    w = self._zero_centered(self._get(f'{pfx}.weight'))
    if w is None:
        return None
    # ... rest unchanged
```

- [ ] **Step 6: Update `attn(pfx, layer)` to inline weight reads**

Same as Task 2 Step 7 — inline qk norm reads with `reorder_rotary_emb`.

- [ ] **Step 7: Update `linear_attn(pfx, layer)` to inline weight reads**

Replace calls to `self.linear_attn_linears(layer)`, `self.linear_attn_params(layer)`, `self.linear_attn_norm_children(layer)` with inline reads.

The GDN fusion (`fuse_gdn_in_proj`) is now in builder.py. Call it directly:

```python
# Read GDN input projections inline
raw_linears = {}
for key in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a']:
    lin = self._linear(f'{pfx}.{key}')
    if lin is not None:
        raw_linears[key] = lin
# Apply GDN fusion
if raw_linears:
    raw_linears = fuse_gdn_in_proj(raw_linears, self._attn_tp,
                                    qkv_split=self._linear_qkv_split)
```

Then read params (`A_log`, `dt_bias`, `conv1d`, `D`) inline from prefix instead of `self.linear_attn_params(layer)`.

- [ ] **Step 8: Update `ffn(pfx, layer)` to remove `linears` param**

Same as Task 2 Step 8 — inline the reads.

- [ ] **Step 9: Update `moe(pfx, layer)` to inline weight reads**

Inline gate read, shared expert gate, and params. Handle shared expert inline. For packed experts, keep a private `_read_packed_expert()` helper if needed (not a base class method).

Call `self.ffn(f'{pfx}.experts.{e}', layer, inter_size=expert_inter, fused_moe=True)` without passing linears.

- [ ] **Step 10: Delete all flat method overrides**

Delete these methods:
- `_read_linear_attn_linears()` (421-430)
- `ffn_linears()` (434-438)
- `_shared_expert_linears()` (440-442)
- `moe_ffn_linears()` (444-449)
- `_packed_moe_expert()` (451-473)
- `attn_norm()` (480-482)
- `ffn_norm()` (484-486)
- `norm_weight()` (488-489)
- `attn_params()` (491-492)
- `attn_norm_children()` (494-507)
- `moe_gate()` (509-522)
- `moe_params()` (524-525)
- `linear_attn_params()` (527-555)
- `linear_attn_norm_children()` (557-565)
- `tok_embeddings()` (567-568)
- `output_weight()` (570-573)

Keep: `_zero_centered()` (414-417), `_is_linear_attn()` (110-112), `_is_moe_layer()` (114-115), `num_experts()` (475-476), `model_info()`, `Qwen3_5InputModel`.

Rename `self._read_linear(` to `self._linear(` everywhere.

- [ ] **Step 11: Verify import**

Run: `python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec; print('OK')"`

- [ ] **Step 12: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(deploy): refactor Qwen3_5Spec to self-contained hierarchy methods"
```

---

### Task 5: Refactor glm4_moe_lite_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

Key differences:
- **MLA attention**: `attn()` reads q_b_proj, kv_b_proj, o_proj (not q/k/v/o) and applies `_mla_fold_and_pad`
- **Has `_mla_fold_and_pad()`** private method — keep as private method on the spec class
- **Has shared expert** in MoE — `moe()` creates shared expert via `ffn()`
- **Dense first-k layers**: `layers()` dispatches dense vs MoE per layer

- [ ] **Step 1: No `reorder_rotary_emb` needed**

Glm4MoeLite uses MLA attention which doesn't need RoPE permutation. Skip this step.

- [ ] **Step 2: Move all imports to top-level**

```python
from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _act_type_id, _cpp_dtype as _cd,
)
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import get_yarn_params, load_model_config, parse_rope_param
```

Delete all local imports inside factory methods.

- [ ] **Step 3: Update `model()` to pass prefixes**

Same as Task 2 Step 3.

- [ ] **Step 4: Update `token_embeds(pfx)`, `root_norm(pfx)`, `lm_head(pfx)`**

Same pattern as Task 2.

- [ ] **Step 5: Update `attn(pfx, layer)` to inline weight reads**

MLA attention reads projections via `add_linear` (not `add_qkv_proj`). Inline the reads:

```python
def attn(self, pfx, layer):
    # Read 6 MLA projections inline
    raw = {}
    for name in ['q_b_proj', 'kv_b_proj', 'o_proj']:
        lin = self._linear(f'{pfx}.{name}')
        if lin is not None:
            raw[name] = lin
    if not raw:
        return None

    # Apply MLA fold and pad
    raw = self._mla_fold_and_pad(raw)
    # ... create AttentionBuilder, call add_linear for each projection
```

Inline `self.attn_params(layer)` and `self.attn_norm_children(layer)` with direct reads.

- [ ] **Step 6: Update `ffn(pfx, layer)` to remove `linears` param**

Same as Task 2 Step 8.

- [ ] **Step 7: Update `moe(pfx, layer)` to inline weight reads**

Inline gate read and params. Handle shared expert inline (read from prefix, call `self.ffn()`).

- [ ] **Step 8: Delete all flat method overrides**

Delete these methods:
- `ffn_linears()` (397-401)
- `_shared_expert_linears()` (403-405)
- `moe_ffn_linears()` (409-411)
- `attn_norm()` (420-421)
- `ffn_norm()` (423-424)
- `attn_params()` (426-427)
- `attn_norm_children()` (429-439)
- `moe_gate()` (441-454)
- `moe_params()` (456-463)
- `tok_embeddings()` (465-466)
- `output_weight()` (468-469)
- `norm_weight()` (471-472)

Keep: `_mla_fold_and_pad()` (311-333), `_mla_fold_and_pad_hf()` (335-393), `num_experts()` (413-416), `model_info()`, `Glm4MoeLiteInputModel`.

Rename `self._read_linear(` to `self._linear(` everywhere.

- [ ] **Step 9: Verify import**

Run: `python -c "from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec; print('OK')"`

- [ ] **Step 10: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(deploy): refactor Glm4MoeLiteSpec to self-contained hierarchy methods"
```

---

### Task 6: Strip spec.py base class

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

After all four specs are refactored and no longer call any base class methods, strip the base class.

- [ ] **Step 1: Rewrite spec.py**

The new spec.py should contain only:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from .linear import Linear
from .module_configs import SpecAttnConfig


class TextModelSpec(ABC):
    """Declarative weight mapping for a model architecture.

    Subclasses define how to read and transform weights for a specific model.
    The ``model()`` method is the entry point, called by TextModelLoader.
    """

    params: dict[str, torch.Tensor]

    # Default values for configure() fields; overwritten by configure().
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

    def model(self):
        """Build the full model hierarchy using builders.

        Override in subclasses.
        """

    def configure(self, cfg: SpecAttnConfig):
        """Set TP and model parameters. Called by TextModelLoader."""
        self._attn_tp = cfg.tp
        self._permute_qk = cfg.permute_qk
        self._repeat_kv = cfg.repeat_kv
        self._head_dim = cfg.head_dim
        self._rope_dim = cfg.rope_dim if cfg.rope_dim else cfg.head_dim
        self._attn_output_gate = cfg.output_gate
        self._kv_head_num = cfg.kv_head_num

    def _get(self, key: str) -> torch.Tensor | None:
        """Get a raw tensor from the checkpoint params."""
        return self.params.get(key)

    def _linear(self, pfx: str) -> Linear | None:
        """Read a Linear bundle from the checkpoint at *pfx*."""
        from .kind_map import build_linear
        return build_linear(self.params, pfx)

    @abstractmethod
    def model_info(self) -> dict:
        """Return model metadata (num_layer, head_num, etc.)."""

    def num_experts(self, layer: int) -> int:
        return 0
```

- [ ] **Step 2: Verify all specs still import correctly**

Run:
```bash
python -c "
from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3TextSpec
from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec
from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec
from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec
print('All specs import OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(deploy): strip TextModelSpec to bare minimum base class"
```

---

### Task 7: Verify with model tests

**Files:** No changes — verification only.

- [ ] **Step 1: Check GPU availability**

Use `get_gpu_usage` MCP tool to confirm an empty GPU.

- [ ] **Step 2: Test Qwen3-MoE**

Use the turbomind-tester agent to test `Qwen3-235B-A22B` with a 128+ token generation prompt. Verify the response is coherent (not gibberish).

- [ ] **Step 3: Test at least one other model**

Test one more model from the available cache to verify the refactoring didn't break other architectures. Pick based on GPU availability and cache presence.

- [ ] **Step 4: Report results**

Report pass/fail for each model tested.
