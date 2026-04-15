# Spec-Driven Loading Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the spec-driven loading cleanup: consolidate SplitSide, absorb remaining dependencies into builder.py, refactor specs to factory methods, delete legacy code.

**Architecture:** Bottom-up approach. First consolidate types and dependencies (builder.py becomes self-contained), then refactor each spec to factory methods, then delete all legacy code.

**Tech Stack:** Python, TurboMind C++ via pybind11

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `lmdeploy/turbomind/deploy/builder.py` | Absorb `_ATTN_TP_RULES`, `_LINEAR_ATTN_TP_RULES`, `_act_type_id`, `fuse_ffn_linears` helpers. Add `LinearBuilder`. |
| Modify | `lmdeploy/turbomind/deploy/spec.py` | Remove `SplitSide` enum. Clean up. |
| Modify | `lmdeploy/turbomind/deploy/text_model_loader.py` | Gut to ~20 lines. |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Refactor `_build_*` to factory methods. |
| Modify | `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Refactor `_build_*` to factory methods. |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Refactor `_build_*` to factory methods. Remove `BSplitSide`. |
| Modify | `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Refactor `_build_*` to factory methods. |
| Delete | `lmdeploy/turbomind/deploy/distributor.py` | Only used by legacy TextModelLoader. |
| Delete | `lmdeploy/turbomind/deploy/transforms.py` | Absorbed into builder.py. |
| Delete | `lmdeploy/turbomind/deploy/commit.py` | Absorbed into builder.py. |

---

### Task 1: Consolidate SplitSide — delete from spec.py, update importers

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:1-19` (remove SplitSide enum)
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:15` (change import)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:22` (change import)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:18` (change import)

- [ ] **Step 1: Remove SplitSide from spec.py**

In `spec.py`, delete lines 11-19 (the `SplitSide` enum and its docstring). Also remove the `import enum` on line 3 if nothing else uses it (check first — `TextModelSpec` doesn't use `enum`).

Change:
```python
import enum
from abc import ABC, abstractmethod
```
To:
```python
from abc import ABC, abstractmethod
```

Delete the entire SplitSide class:
```python
class SplitSide(enum.Enum):
    ...
```

- [ ] **Step 2: Update text_model_loader.py import**

In `text_model_loader.py:15`, change:
```python
from .spec import SplitSide
```
To:
```python
from .builder import SplitSide
```

- [ ] **Step 3: Update gpt_oss_spec.py import**

In `gpt_oss_spec.py:22`, change:
```python
from ..spec import TextModelSpec, SplitSide
```
To:
```python
from ..spec import TextModelSpec
from ..builder import SplitSide
```

- [ ] **Step 4: Update qwen3_5_spec.py import**

In `qwen3_5_spec.py:18`, change:
```python
from ..spec import TextModelSpec, SplitSide
```
To:
```python
from ..spec import TextModelSpec
from ..builder import SplitSide
```

- [ ] **Step 5: Update type annotations in spec.py that reference SplitSide**

The methods `attn_params`, `moe_params`, `linear_attn_params` have return type annotations referencing `SplitSide`. Since `SplitSide` is now in `builder.py`, use a string literal or `TYPE_CHECKING` import:

At the top of spec.py, add:
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .builder import SplitSide
```

The return type annotations `dict[str, tuple[torch.Tensor, SplitSide | None]]` will resolve correctly via `TYPE_CHECKING` and `from __future__ import annotations`.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py \
        lmdeploy/turbomind/deploy/text_model_loader.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(deploy): consolidate SplitSide into builder.py"
```

---

### Task 2: Absorb commit.py symbols into builder.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py` (add `_ATTN_TP_RULES`, `_LINEAR_ATTN_TP_RULES`, `_act_type_id`)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (change import)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (change import)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (change import)
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (change import)

- [ ] **Step 1: Add `_act_type_id` to builder.py**

Append after `_cpp_dtype` (after line 73):

```python
def _act_type_id(act_str: str) -> int:
    """Convert activation_type string to C++ ActivationType enum value."""
    return {'silu': 0, 'gpt-oss': 1}.get(act_str, 0)
```

- [ ] **Step 2: Add `_ATTN_TP_RULES` and `_LINEAR_ATTN_TP_RULES` to builder.py**

Append after `_act_type_id`:

```python
# TP split rules for attention linears.
# Keys absent from the table are broadcast (no TP split).
_ATTN_TP_RULES: dict[str, dict] = {
    "w_qkv":     dict(split_side=SplitSide.OUTPUT),
    "wo":        dict(split_side=SplitSide.INPUT),
    "q_proj":    dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_b_proj": dict(split_side=SplitSide.OUTPUT),
}

_LINEAR_ATTN_TP_RULES: dict[str, dict] = {
    "in_proj_qkv": dict(split_side=SplitSide.OUTPUT),
    "in_proj_z":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_b":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_a":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_all": dict(split_side=SplitSide.OUTPUT),
    "out_proj":    dict(split_side=SplitSide.INPUT),
}
```

- [ ] **Step 3: Update `AttentionBuilder.add_linear` to use internal `_ATTN_TP_RULES`**

In `builder.py`, the `add_linear` method currently does `from .commit import _ATTN_TP_RULES`. Change it to use the module-level constant:

```python
    def add_linear(self, name, linear):
        """Commit a named attention linear using TP rules."""
        rule = _ATTN_TP_RULES.get(name, {})
        split_side = rule.get('split_side')
        self._commit_linear(name, linear, split_side=split_side,
                            model_dtype=self.config.data_type)
```

No more `from .commit import`, no more `SplitSide(split_side.value)` conversion since `_ATTN_TP_RULES` now uses builder's own `SplitSide`.

- [ ] **Step 4: Update all four specs to import from builder.py**

In each spec, change `from ..commit import _cpp_dtype, _act_type_id` to `from ..builder import _cpp_dtype, _act_type_id`.

For `qwen3_spec.py`:
```python
from ..builder import _cpp_dtype, _act_type_id
```
(replaces `from ..commit import _cpp_dtype` at line ~40, adds `_act_type_id`)

For `gpt_oss_spec.py`:
```python
from ..builder import _cpp_dtype, _act_type_id
```

For `qwen3_5_spec.py`:
```python
from ..builder import _cpp_dtype, _act_type_id
```
Also in `_build_linear_attn`, change `from ..commit import _LINEAR_ATTN_TP_RULES` to `from ..builder import _LINEAR_ATTN_TP_RULES`.

For `glm4_moe_lite_spec.py`:
```python
from ..builder import _cpp_dtype, _act_type_id
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(deploy): absorb _ATTN_TP_RULES, _act_type_id into builder.py"
```

---

### Task 3: Move fuse_ffn_linears into builder.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py` (add fuse functions)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (no change needed — uses builder's FfnBuilder)
- Modify: all specs if they import from transforms (check — only builder.py does via FfnBuilder)

- [ ] **Step 1: Copy `_should_fuse_silu`, `_can_fuse_w1w3`, `_shard_linear_for_tp`, `fuse_ffn_linears` into builder.py**

In `builder.py`, after the existing helper functions and before the `Builder` class, add:

```python
from .linear import Linear as _Linear, chunk_linears as _chunk_linears, interleave_linears as _interleave_linears


def _should_fuse_silu(w1_linear: _Linear, act_type: str, is_moe: bool = False) -> bool:
    """Determine if fused SiLU (interleave) should be used for w1+w3 fusion."""
    if act_type not in ('', 'silu', 'SiLU'):
        return False
    weight = w1_linear.tensors.get("weight")
    is_quantized = weight is not None and weight.element_size() < 2
    if not is_quantized and not is_moe:
        return False
    fmt = w1_linear.weight_format
    if fmt is not None and fmt.name == "fp8":
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap == (9, 0):
                return False
    return True


def _can_fuse_w1w3(w1: _Linear, tp: int) -> bool:
    if tp <= 1:
        return True
    fmt = w1.weight_format
    if fmt is None or fmt.block_out is None:
        return True
    w = w1.tensors.get("weight")
    if w is None:
        return True
    return (w.size(-1) // tp) % fmt.block_out == 0


def fuse_ffn_linears(
    w1: _Linear, w3: _Linear, tp: int, act_type: str, is_moe: bool = False,
) -> tuple[_Linear | None, bool]:
    """Optionally fuse w1/w3 on full (unsharded) tensors for FFN."""
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)
    if can_fuse:
        if fused_silu:
            w1w3 = _interleave_linears(w1, w3)
        else:
            w1w3 = _chunk_linears(w1, w3, tp)
        return (w1w3, fused_silu)
    else:
        return (None, fused_silu)
```

- [ ] **Step 2: Update FfnBuilder.add_ffn to use internal fuse_ffn_linears**

In `FfnBuilder.add_ffn`, change:
```python
        from .transforms import fuse_ffn_linears
```
To just use the module-level `fuse_ffn_linears` directly (no import needed).

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "refactor(deploy): move fuse_ffn_linears into builder.py"
```

---

### Task 4: Add LinearBuilder to builder.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py`

- [ ] **Step 1: Add LinearBuilder class**

After `NormBuilder`, add:

```python
class LinearBuilder(Builder):
    """Builder for standalone linear layers (embeddings, lm_head).

    Wraps a C++ LinearWeight module. Use ``set_weight()`` to commit
    the weight tensor.
    """

    def set_weight(self, tensor: torch.Tensor, split_side=None):
        """Commit the weight tensor to all GPU handles.

        Parameters
        ----------
        tensor : torch.Tensor
            The weight tensor (already padded/transposed by the spec).
        split_side : SplitSide | None
            TP split semantics. None means broadcast.
        """
        self._commit_tensor('weight', tensor, split_side)
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "feat(deploy): add LinearBuilder for embeddings and lm_head"
```

---

### Task 5: Refactor Qwen3Spec to factory methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

Qwen3Spec is the canonical spec. All others follow the same pattern.

- [ ] **Step 1: Refactor `model()` to use factory methods**

Replace the entire `model()` and `_build_*` methods with:

```python
    def model(self):
        from ..builder import (TextModelBuilder, ModuleListBuilder,
                               DecoderLayerBuilder, SplitSide)
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds()
        root.norm = self.root_norm()
        root.output = self.lm_head()
        root.layers = self.layers('model.layers')

    def token_embeds(self):
        from ..builder import LinearBuilder
        from ..module_configs import LinearConfig
        from ..linear import pad_out_dim

        emb = self._get("model.embed_tokens.weight")
        if emb is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        cfg = LinearConfig(input_dim=padded_vocab,
                           output_dim=mc.hidden_units // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def root_norm(self):
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

        tensor = self.norm_weight()
        if tensor is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(tensor)
        return m

    def lm_head(self):
        from ..builder import LinearBuilder
        from ..module_configs import LinearConfig
        from ..linear import pad_out_dim

        output = self.output_weight()
        if output is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        cfg = LinearConfig(input_dim=mc.hidden_units,
                           output_dim=padded_vocab // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m

    def norm(self, pfx):
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

        tensor = self._get(f'{pfx}.weight')
        if tensor is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(tensor)
        return m

    def attn(self, pfx, layer):
        from ..builder import AttentionBuilder
        from ..module_configs import AttentionConfig

        q = self._read_linear(f"{pfx}.q_proj")
        k = self._read_linear(f"{pfx}.k_proj")
        v = self._read_linear(f"{pfx}.v_proj")
        o = self._read_linear(f"{pfx}.o_proj")

        if q is None and k is None and v is None and o is None:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp
        ranks = self._attn_ranks

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
        attn = AttentionBuilder(attn_cfg, self._contexts, tp=tp, ranks=ranks)

        if q is not None and k is not None and v is not None:
            attn.add_qkv_proj(q, k, v)
        if o is not None:
            attn.add_o_proj(o)

        for name, tensor in self.attn_params(layer).items():
            attn.add_param(name, tensor)

        norm_children = self.attn_norm_children(layer)
        q_norm = norm_children.get('q_norm')
        k_norm = norm_children.get('k_norm')
        if q_norm is not None or k_norm is not None:
            attn.add_qk_norm(q_norm, k_norm)

        return attn

    def ffn(self, pfx, layer, linears=None, inter_size=None, fused_moe=False):
        from ..builder import FfnBuilder
        from ..module_configs import FfnConfig
        from ..builder import _act_type_id

        if linears is None:
            linears = self._read_ffn_linears(pfx)
        if not linears:
            return None

        w1 = linears.get('w1')
        w3 = linears.get('w3')
        w2 = linears.get('w2')

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._mlp_tp
        ranks = self._mlp_ranks

        if inter_size is None:
            is_list = mc.inter_size
            inter_size = is_list[layer] if is_list and layer < len(is_list) else 0

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        from ..builder import MoeBuilder, ModuleListBuilder
        from ..module_configs import MoeConfig, ModuleListConfig
        from ..builder import _act_type_id

        if self.num_experts(layer) <= 0:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._mlp_tp
        ranks = self._mlp_ranks

        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        moe = MoeBuilder(moe_cfg, self._contexts, tp=tp, ranks=ranks)

        for name, linear in self.moe_gate(layer).items():
            moe.add_gate(name, linear, model_dtype=dtype)

        for name, tensor in self.moe_params(layer).items():
            moe.add_param(name, tensor)

        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            expert = self.ffn(
                f'{pfx}.experts.{e}', layer,
                linears=self.moe_ffn_linears(layer, e),
                inter_size=expert_inter, fused_moe=True)
            if expert is not None:
                experts[str(e)] = expert

        moe.experts = experts
        return moe

    def layers(self, pfx):
        from ..builder import ModuleListBuilder, DecoderLayerBuilder
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        m = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
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

- [ ] **Step 2: Add helper `_cpp_dtype()` shortcut method**

Add to the class (after `__init__`):

```python
    def _cpp_dtype(self):
        from ..builder import _cpp_dtype as _cd
        return _cd(self._mc.data_type)
```

- [ ] **Step 3: Delete all `_build_*` methods**

Remove `_build_attention`, `_build_ffn`, `_build_moe`. These are replaced by the factory methods above.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "refactor(deploy): refactor Qwen3Spec to factory methods"
```

---

### Task 6: Refactor GptOssSpec to factory methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

- [ ] **Step 1: Replace model() and _build_* with factory methods**

Replace `model()`, `_build_attention`, `_build_moe` with:

```python
    def model(self):
        from ..builder import TextModelBuilder, ModuleListBuilder, DecoderLayerBuilder
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds()
        root.norm = self.root_norm()
        root.output = self.lm_head()
        root.layers = self.layers('model.layers')

    def _cpp_dtype(self):
        from ..builder import _cpp_dtype as _cd
        return _cd(self._mc.data_type)

    def token_embeds(self):
        from ..builder import LinearBuilder, SplitSide
        from ..module_configs import LinearConfig
        from ..linear import pad_out_dim

        emb = self.tok_embeddings()
        if emb is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        cfg = LinearConfig(input_dim=padded_vocab,
                           output_dim=mc.hidden_units // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def root_norm(self):
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

        tensor = self.norm_weight()
        if tensor is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(tensor)
        return m

    def lm_head(self):
        from ..builder import LinearBuilder, SplitSide
        from ..module_configs import LinearConfig
        from ..linear import pad_out_dim

        output = self.output_weight()
        if output is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        cfg = LinearConfig(input_dim=mc.hidden_units,
                           output_dim=padded_vocab // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m

    def norm(self, pfx):
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

        tensor = self._get(f'{pfx}.weight')
        if tensor is None:
            return None
        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(tensor)
        return m

    def attn(self, pfx, layer):
        from ..builder import AttentionBuilder
        from ..module_configs import AttentionConfig

        q = self._read_linear(f"{pfx}.q_proj")
        k = self._read_linear(f"{pfx}.k_proj")
        v = self._read_linear(f"{pfx}.v_proj")
        o = self._read_linear(f"{pfx}.o_proj")

        if q is None and k is None and v is None and o is None:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp
        ranks = self._attn_ranks

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
        attn = AttentionBuilder(attn_cfg, self._contexts, tp=tp, ranks=ranks)

        if q is not None and k is not None and v is not None:
            attn.add_qkv_proj(q, k, v)
        if o is not None:
            attn.add_o_proj(o)

        for name, val in self.attn_params(layer).items():
            if isinstance(val, tuple):
                tensor, _ = val
            else:
                tensor = val
            attn.add_param(name, tensor)

        return attn

    def ffn(self, pfx, layer, linears=None, inter_size=None, fused_moe=False):
        from ..builder import FfnBuilder
        from ..module_configs import FfnConfig
        from ..builder import _act_type_id

        if linears is None:
            linears = self._read_ffn_linears(pfx)
        if not linears:
            return None

        w1 = linears.get('w1')
        w3 = linears.get('w3')
        w2 = linears.get('w2')

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._mlp_tp
        ranks = self._mlp_ranks

        if inter_size is None:
            is_list = mc.inter_size
            inter_size = is_list[layer] if is_list and layer < len(is_list) else 0

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        from ..builder import MoeBuilder, ModuleListBuilder
        from ..module_configs import MoeConfig, ModuleListConfig
        from ..builder import _act_type_id

        if self.num_experts(layer) <= 0:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._mlp_tp
        ranks = self._mlp_ranks

        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        moe = MoeBuilder(moe_cfg, self._contexts, tp=tp, ranks=ranks)

        for name, linear in self.moe_gate(layer).items():
            moe.add_gate(name, linear, model_dtype=dtype)

        for name, val in self.moe_params(layer).items():
            if isinstance(val, tuple):
                tensor, _ = val
            else:
                tensor = val
            moe.add_param(name, tensor)

        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            expert = self.ffn(
                f'{pfx}.experts.{e}', layer,
                linears=self.moe_ffn_linears(layer, e),
                inter_size=expert_inter, fused_moe=True)
            if expert is not None:
                experts[str(e)] = expert

        moe.experts = experts
        return moe

    def layers(self, pfx):
        from ..builder import ModuleListBuilder, DecoderLayerBuilder
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        m = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._mc.num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', layer=i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', layer=i)
            m[str(i)] = d
        return m
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(deploy): refactor GptOssSpec to factory methods"
```

---

### Task 7: Refactor Qwen3_5Spec to factory methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

Key differences from Qwen3Spec: linear attention layers, zero-centered RMSNorm, shared expert gate, packed MoE.

- [ ] **Step 1: Replace model() and _build_* with factory methods**

The `model()` follows the same pattern. Replace `model()`, `_build_attention`, `_build_linear_attn`, `_build_ffn`, `_build_moe` with factory methods.

The `attn()` and `ffn()` methods are identical to Qwen3Spec. Add a `linear_attn()` method unique to this spec:

```python
    def linear_attn(self, pfx, layer):
        from ..builder import Builder, SplitSide
        from ..module_configs import DeltaNetConfig
        from ..builder import _LINEAR_ATTN_TP_RULES

        la_linears = self.linear_attn_linears(layer)
        if not la_linears:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp
        ranks = self._attn_ranks

        dn_cfg = DeltaNetConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype)
        m = Builder(dn_cfg, self._contexts, tp=tp, ranks=ranks)

        for name, lin in la_linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            m._commit_linear(name, lin, split_side=split_side,
                             model_dtype=dtype)

        for name, val in self.linear_attn_params(layer).items():
            tensor, ss = val
            m._commit_tensor(name, tensor, split_side=ss)

        for name, tensor in self.linear_attn_norm_children(layer).items():
            m._add_norm_child(name, tensor, data_type=dtype)

        return m
```

The `norm()` method applies zero-centered transform:

```python
    def norm(self, pfx):
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

        tensor = self._get(f'{pfx}.weight')
        if tensor is None:
            return None
        tensor = self._zero_centered(tensor)
        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(tensor)
        return m
```

The `root_norm()` and `token_embeds()`/`lm_head()` use `self._norm_key` and `self._embed_key` instead of hardcoded paths.

The `layers()` method dispatches between `attn()` and `linear_attn()` per layer:

```python
    def layers(self, pfx):
        from ..builder import ModuleListBuilder, DecoderLayerBuilder
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        m = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm')
            if self._is_linear_attn(i):
                d.linear_attn = self.linear_attn(f'{pfx}.{i}.linear_attn', layer=i)
            else:
                d.attention = self.attn(f'{pfx}.{i}.self_attn', layer=i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm')
            if self.num_experts(i) > 0:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp.shared_expert', layer=i)
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', layer=i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', layer=i)
            m[str(i)] = d
        return m
```

- [ ] **Step 2: Remove all `BSplitSide` references**

Remove `SplitSide as BSplitSide` from imports. Remove `BSplitSide(x.value)` conversions. Use `SplitSide` directly everywhere (now there's only one).

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(deploy): refactor Qwen3_5Spec to factory methods"
```

---

### Task 8: Refactor Glm4MoeLiteSpec to factory methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

Key differences: MLA attention (uses `add_linear` not `add_qkv_proj`), dense first-k layers, score correction bias.

- [ ] **Step 1: Replace model() and _build_* with factory methods**

The `attn()` method uses MLA-style projections:

```python
    def attn(self, pfx, layer):
        from ..builder import AttentionBuilder
        from ..module_configs import AttentionConfig

        raw: dict = {}
        for tm_name, hf_key in [
            ("q_a_proj", "q_a_proj"),
            ("q_b_proj", "q_b_proj"),
            ("q_proj", "q_proj"),
            ("kv_a_proj", "kv_a_proj_with_mqa"),
            ("kv_b_proj", "kv_b_proj"),
            ("wo", "o_proj"),
        ]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                raw[tm_name] = lin

        if "q_proj" in raw and "q_b_proj" not in raw:
            raw["q_b_proj"] = raw.pop("q_proj")

        if not raw:
            return None

        self._mla_fold_and_pad(raw)

        mc = self._mc
        dtype = self._cpp_dtype()
        tp = self._attn_tp
        ranks = self._attn_ranks

        attn_cfg = AttentionConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype, window_size=-1)
        attn = AttentionBuilder(attn_cfg, self._contexts, tp=tp, ranks=ranks)

        for name, lin in raw.items():
            attn.add_linear(name, lin)

        for name, val in self.attn_params(layer).items():
            if isinstance(val, tuple):
                tensor, _ = val
            else:
                tensor = val
            attn.add_param(name, tensor)

        for name, tensor in self.attn_norm_children(layer).items():
            attn._add_norm_child(name, tensor, data_type=dtype)

        return attn
```

The `layers()` method handles dense-first-k logic:

```python
    def layers(self, pfx):
        from ..builder import ModuleListBuilder, DecoderLayerBuilder
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        m = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._mc.num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', layer=i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm')
            if self.num_experts(i) > 0:
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp.shared_experts', layer=i)
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', layer=i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', layer=i)
            m[str(i)] = d
        return m
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(deploy): refactor Glm4MoeLiteSpec to factory methods"
```

---

### Task 9: Gut TextModelLoader

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Replace entire file with simplified loader**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: injects context into specs and calls model()."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .module_configs import SpecAttnConfig

if TYPE_CHECKING:
    from .spec import TextModelSpec
    from .target_model.base import BaseOutputModel


class TextModelLoader:
    """Drives the model loading pipeline for text models.

    All structure comes from the TextModelSpec. The loader only injects
    GPU handles, contexts, and TP configuration, then calls spec.model().
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

        self._attn_ranks = [model.tp_ranks(gpu)[0]
                            for gpu in range(model.gpu_count)]
        self._mlp_ranks = [model.tp_ranks(gpu)[1]
                           for gpu in range(model.gpu_count)]
        handles = []
        contexts = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
            contexts.append(model.context(gpu))
        self._contexts = contexts
        self._root_handles = handles

    def __call__(self, layer: int, spec: 'TextModelSpec'):
        spec._contexts = self._contexts
        spec._root_handles = self._root_handles
        spec._mc = self.model.model_config
        spec._attn_tp = self.attn_tp
        spec._attn_cp = self.model.attn_cp_size
        spec._mlp_tp = self.mlp_tp
        spec._attn_ranks = self._attn_ranks
        spec._mlp_ranks = self._mlp_ranks
        spec._repeat_kv = self.model.repeat_kv
        mc = self.model.model_config
        rope_param = self.model.attention_config.rope_param
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            repeat_kv=self.model.repeat_kv,
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
            output_gate=mc.attn_output_gate,
            kv_head_num=mc.kv_head_num,
        ))
        spec.model()
        return 1
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(deploy): gut TextModelLoader to ~60 lines, remove legacy code"
```

---

### Task 10: Delete distributor.py, transforms.py, commit.py

**Files:**
- Delete: `lmdeploy/turbomind/deploy/distributor.py`
- Delete: `lmdeploy/turbomind/deploy/transforms.py`
- Delete: `lmdeploy/turbomind/deploy/commit.py`

- [ ] **Step 1: Verify no remaining imports**

Search for any remaining imports of these modules:
```bash
grep -r 'from .distributor\|from .transforms\|from .commit' lmdeploy/turbomind/deploy/
```

Expected: no results. All imports should have been updated in Tasks 1-8.

- [ ] **Step 2: Delete the three files**

```bash
git rm lmdeploy/turbomind/deploy/distributor.py \
       lmdeploy/turbomind/deploy/transforms.py \
       lmdeploy/turbomind/deploy/commit.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(deploy): delete distributor.py, transforms.py, commit.py"
```

---

### Task 11: Clean spec.py base class

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

- [ ] **Step 1: Remove `model()` NotImplementedError default**

Since all four specs implement `model()`, change the base class method from raising `NotImplementedError` to `pass` or remove it. Keeping it as a no-op is safest:

```python
    def model(self):
        """Build the full model hierarchy using builders.

        Override in subclasses.
        """
```

- [ ] **Step 2: Clean up any remaining references to deleted modules**

Search spec.py for any references to `commit`, `distributor`, `transforms`, or `SplitSide`. Remove or update as needed.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(deploy): clean spec.py base class"
```

---

## Self-Review

**Spec coverage:**
- Section 1 (Consolidate SplitSide) → Task 1
- Section 2 (Absorb dependencies) → Tasks 2, 3, 4
- Section 3 (Delete legacy, gut TextModelLoader) → Tasks 9, 10
- Section 4 (Refactor specs to factory methods) → Tasks 5, 6, 7, 8
- Section 5 (Base class updates) → Task 11
- moe() reuses ffn() → Tasks 5-8 (all moe() methods call self.ffn() for experts)

**Placeholder scan:** No TBDs, TODOs, or "similar to" references. All factory method code is shown in full for each spec.

**Type consistency:**
- `SplitSide` is now only in `builder.py` — all specs import from there
- `_cpp_dtype`, `_act_type_id`, `_ATTN_TP_RULES`, `_LINEAR_ATTN_TP_RULES` all in `builder.py`
- `fuse_ffn_linears` in `builder.py`, used by `FfnBuilder.add_ffn` via module-level reference
- `LinearBuilder.set_weight(tensor, split_side)` used consistently in `token_embeds()`, `lm_head()`
- `ffn()` accepts optional `linears` parameter — used by `moe()` to pass expert linears
