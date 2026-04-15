# Spec Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove copy-paste duplication across 4 model spec files by moving shared factory methods to the base class, killing dead code, and eliminating pointless None guards.

**Architecture:** Base class `TextModelSpec` gains concrete factory methods (`token_embeds`, `lm_head`, `output_norm`, `norm`, `_cpp_dtype`) that take checkpoint keys as parameters. Each spec deletes its copy-pasted implementations and calls the base methods from `model()`. The `reorder_rotary_emb` function moves to a shared module. `InputModel.__init__` is absorbed into `BaseInputModel`.

**Tech Stack:** Python, PyTorch, TurboMind deploy pipeline.

---

### Task 1: Add `reorder_rotary_emb` to `source_model/utils.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py`

- [ ] **Step 1: Add the function to utils.py**

Append the following to the end of `lmdeploy/turbomind/deploy/source_model/utils.py`:

```python
def reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Reorder rotary embedding layout for TurboMind's RoPE kernel."""
    import torch
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

Note: `torch` is already available at module level via the existing `import math` — but it is NOT imported yet. Add `import torch` to the top-level imports of utils.py (after `import math`).

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "refactor(deploy): move reorder_rotary_emb to shared utils"
```

---

### Task 2: Add shared factory methods to `spec.py` base class

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

This task adds 5 concrete methods to `TextModelSpec` and removes the dead `model_info()` abstract method.

- [ ] **Step 1: Update imports**

Replace the current imports in `spec.py`:

```python
from abc import ABC, abstractmethod
import torch
from .linear import Linear
from .module_configs import SpecAttnConfig
```

With:

```python
from abc import ABC

import torch

from .builder import LinearBuilder, NormBuilder, SplitSide, _cpp_dtype as _cd
from .linear import Linear, pad_out_dim
from .module_configs import LinearConfig, NormConfig, SpecAttnConfig
```

Note: `abstractmethod` is no longer needed (we're removing the only abstract method).

- [ ] **Step 2: Add injected field declarations**

After the existing injected fields (`_contexts`, `_root_handles`), add the missing injected fields that the new base methods need:

```python
    # Injected by TextModelLoader
    _contexts: list = None
    _root_handles: list = None
    _mc = None  # ModelConfig
    _attn_cp: int = 1
    _attn_ranks: list = None
    _mlp_tp: int = 1
    _mlp_ranks: list = None
```

Delete the old `_contexts` and `_root_handles` lines (they're replaced by the block above).

- [ ] **Step 3: Add the 5 concrete methods**

Add these methods to `TextModelSpec`, after `configure()` and before `_get()`:

```python
    def _cpp_dtype(self):
        return _cd(self._mc.data_type)

    def token_embeds(self, key):
        emb = self._get(key)
        mc = self._mc
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        dtype = self._cpp_dtype()
        cfg = LinearConfig(input_dim=padded_vocab,
                           output_dim=mc.hidden_units // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def output_norm(self, key):
        w = self._get(key)
        cfg = NormConfig(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def lm_head(self, key):
        output = self._get(key)
        mc = self._mc
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        dtype = self._cpp_dtype()
        cfg = LinearConfig(input_dim=mc.hidden_units,
                           output_dim=padded_vocab // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m

    def norm(self, key):
        w = self._get(key)
        cfg = NormConfig(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m
```

- [ ] **Step 4: Remove `model_info()` abstract method**

Delete:

```python
    @abstractmethod
    def model_info(self) -> dict:
        """Return model metadata (num_layer, head_num, etc.)."""
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(deploy): add shared factory methods to TextModelSpec base"
```

---

### Task 3: Absorb InputModel `__init__()` into `BaseInputModel`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/base.py`

- [ ] **Step 1: Update BaseInputModel.__init__**

Replace the current `__init__` in `BaseInputModel`:

```python
    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        """Constructor for BaseInputModel.

        Args:
            model_path (str): the path of the model.
            tokenizer_path (str): the path of the tokenizer model.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
```

With:

```python
    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_config = load_model_config(model_path)
        self.model_format = kwargs.get('model_format')
        self.fp8_quant = kwargs.get('fp8_quant', False)
```

- [ ] **Step 2: Add the import for `load_model_config`**

Add to the top of `base.py`:

```python
from .utils import load_model_config
```

- [ ] **Step 3: Remove `model_info()` abstract method from BaseInputModel**

Since `model_info()` is still needed on InputModel subclasses (it IS called by `finalize_config`), keep the abstract declaration. No change needed here — just confirming the abstract `model_info()` stays.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/base.py
git commit -m "refactor(deploy): absorb InputModel init into BaseInputModel"
```

---

### Task 4: Refactor `qwen3_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

This spec is the simplest case — all 5 deleted methods map directly to base class defaults.

- [ ] **Step 1: Update imports**

Replace the current import block:

```python
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
from ..kind_map import build_linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

With:

```python
import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig,
    ModuleListConfig, MoeConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param, reorder_rotary_emb
```

Removed: `LinearBuilder`, `NormBuilder`, `SplitSide`, `_cpp_dtype as _cd` (no longer used directly). `pad_out_dim` (no longer used). `LinearConfig`, `NormConfig` (no longer used). `build_linear` (only used in deleted `_linear` override). Added: `reorder_rotary_emb` from utils.

- [ ] **Step 2: Delete the `reorder_rotary_emb` local function**

Delete lines 29-47 (the entire `reorder_rotary_emb` function definition before the class).

- [ ] **Step 3: Update `model()`**

Replace:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens')
        root.norm = self.root_norm('model.norm')
        root.output = self.lm_head('lm_head')
        root.layers = self.layers('model.layers')
```

With:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens.weight')
        root.norm = self.output_norm('model.norm.weight')
        tie = self.cfg.get("tie_word_embeddings", False)
        lm_key = "model.embed_tokens.weight" if tie else "lm_head.weight"
        root.output = self.lm_head(lm_key)
        root.layers = self.layers('model.layers')
```

- [ ] **Step 4: Delete these methods entirely**

Delete the following methods from `Qwen3TextSpec`:
- `_cpp_dtype(self)` (2 lines)
- `token_embeds(self, pfx)` (~15 lines)
- `root_norm(self, pfx)` (~10 lines)
- `lm_head(self, pfx)` (~15 lines)
- `norm(self, pfx)` (~8 lines)
- `_linear(self, prefix)` (3 lines)
- `model_info(self)` (~25 lines)

- [ ] **Step 5: Remove None guards from `attn()`**

Remove the early-return guard:
```python
        if q is None and k is None and v is None and o is None:
            return None
```

Make the qkv/o adds unconditional:
```python
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)
```

- [ ] **Step 6: Remove None guards from `ffn()`**

Remove the `linears` dict and the early-return:
```python
        linears = {}
        if w1 is not None: linears['w1'] = w1
        if w3 is not None: linears['w3'] = w3
        if w2 is not None: linears['w2'] = w2
        if not linears:
            return None
```

Replace `m.add_ffn(linears.get('w1'), linears.get('w2'), linears.get('w3'))` with `m.add_ffn(w1, w2, w3)`.

The method should read:

```python
    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f"{pfx}.gate_proj")
        w3 = self._linear(f"{pfx}.up_proj")
        w2 = self._linear(f"{pfx}.down_proj")

        mc = self._mc
        tp = self._mlp_tp
        dtype = self._cpp_dtype()

        if inter_size is None:
            is_list = mc.inter_size
            inter_size = is_list[layer] if is_list and layer < len(
                is_list) else 0

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m
```

- [ ] **Step 7: Remove None guards from `moe()`**

Remove the None guard on gate_w:
```python
        if gate_w is not None:
            gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
            m.add_gate('gate', Linear({"weight": gate_w}), model_dtype=dtype)
```

Replace with unconditional:
```python
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({"weight": gate_w}), model_dtype=dtype)
```

Keep the `if self.num_experts(layer) <= 0: return None` guard (this is structural, not a None guard).

Remove the None guard on expert:
```python
        if expert is not None:
            experts[str(e)] = expert
```

Replace with unconditional:
```python
        experts[str(e)] = self.ffn(...)
```

- [ ] **Step 8: Update `layers()` norm calls to pass full keys**

Replace:
```python
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm')
```
With:
```python
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm.weight')
```

Replace:
```python
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm')
```
With:
```python
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm.weight')
```

- [ ] **Step 9: Delete InputModel `__init__` from `Qwen3InputModel`**

Delete the `__init__` method (3 lines) from `Qwen3InputModel`. The base class now handles it.

- [ ] **Step 10: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "refactor(deploy): deduplicate qwen3_spec using base class methods"
```

---

### Task 5: Refactor `gpt_oss_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

Same pattern as Task 4, with the same set of deletions. gpt_oss does NOT have `reorder_rotary_emb` — no change for that.

- [ ] **Step 1: Update imports**

Replace:

```python
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
from ..kind_map import build_linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

With:

```python
import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig,
    ModuleListConfig, MoeConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

Note: `build_linear` IS still needed by `_read_packed_expert()` — keep it:
```python
from ..kind_map import build_linear
```

- [ ] **Step 2: Update `model()`**

Replace:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens')
        root.norm = self.root_norm('model.norm')
        root.output = self.lm_head('lm_head')
        root.layers = self.layers('model.layers')
```

With:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens.weight')
        root.norm = self.output_norm('model.norm.weight')
        tie = self.cfg.get("tie_word_embeddings", False)
        lm_key = "model.embed_tokens.weight" if tie else "lm_head.weight"
        root.output = self.lm_head(lm_key)
        root.layers = self.layers('model.layers')
```

- [ ] **Step 3: Delete these methods from `GptOssSpec`**
- `_cpp_dtype(self)`
- `token_embeds(self, pfx)`
- `root_norm(self, pfx)`
- `lm_head(self, pfx)`
- `norm(self, pfx)`
- `_linear(self, prefix)`
- `model_info(self)`

- [ ] **Step 4: Remove None guards from `attn()`**

Remove the early-return guard:
```python
        if q is None and k is None and v is None and o is None:
            return None
```

Make qkv/o adds unconditional:
```python
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)
```

Remove the None guard on sinks:
```python
        if sinks is not None:
            attn.add_param('sinks', sinks)
```
Replace with:
```python
        attn.add_param('sinks', self._get(f'{pfx}.sinks'))
```

Wait — attention sinks may genuinely be absent in some checkpoints. However, the design says "crash immediately." Follow the design: make unconditional. If sinks are always present for gpt-oss models, this is correct.

- [ ] **Step 5: Remove None guards from `ffn()`**

Same pattern as Task 4 Step 6: remove `linears` dict, remove early-return, call `m.add_ffn(w1, w2, w3)` directly.

- [ ] **Step 6: Remove None guards from `moe()`**

Remove the None guard on gate_w:
```python
        if gate_w is not None:
            gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
            tensors = {"weight": gate_w}
            gate_bias = self._get(f'{pfx}.router.bias')
            if gate_bias is not None:
                tensors["bias"] = gate_bias
            m.add_gate('gate', Linear(tensors), model_dtype=dtype)
```

Replace with unconditional:
```python
        gate_w = self._get(f'{pfx}.router.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {"weight": gate_w}
        gate_bias = self._get(f'{pfx}.router.bias')
        if gate_bias is not None:
            tensors["bias"] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)
```

Note: keep the `gate_bias` None check — bias is genuinely optional for some models.

Remove the expert None guard:
```python
        if expert is not None:
            experts[str(e)] = expert
```
Replace with:
```python
        experts[str(e)] = self._moe_expert_ffn(...)
```

- [ ] **Step 7: Update `layers()` norm calls to pass full keys**

Same as Task 4 Step 8 — append `.weight` to all `self.norm(...)` calls.

- [ ] **Step 8: Delete InputModel `__init__` from `GptOssInputModel`**

Delete the `__init__` method.

- [ ] **Step 9: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(deploy): deduplicate gpt_oss_spec using base class methods"
```

---

### Task 6: Refactor `qwen3_5_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

Same pattern as Task 4, but qwen3_5 writes its own `norm()` and `output_norm()` (with `_zero_centered` transform) instead of using the base class defaults.

- [ ] **Step 1: Update imports**

Replace:

```python
import torch

from ..builder import (
    AttentionBuilder, Builder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _LINEAR_ATTN_TP_RULES, _act_type_id, _cpp_dtype as _cd,
    fuse_gdn_in_proj,
)
from ..kind_map import build_linear
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, DeltaNetConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param
```

With:

```python
import torch

from ..builder import (
    AttentionBuilder, Builder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _LINEAR_ATTN_TP_RULES, _act_type_id,
    fuse_gdn_in_proj,
)
from ..kind_map import build_linear
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, DeltaNetConfig, FfnConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param, reorder_rotary_emb
```

Removed: `LinearBuilder`, `LinearConfig`, `_cpp_dtype as _cd` (no longer used directly). Added: `reorder_rotary_emb` from utils.

Note: `build_linear` IS still used by `_packed_moe_expert_indexed` — keep it.
Note: `pad_out_dim` IS still used by `token_embeds` — wait, `token_embeds` is deleted. Check if `pad_out_dim` is used elsewhere... It is NOT used elsewhere in qwen3_5. Remove it.

Wait — `pad_out_dim` was only used in `token_embeds` and `lm_head` which are now deleted. Remove the import.

Corrected imports:

```python
import torch

from ..builder import (
    AttentionBuilder, Builder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _LINEAR_ATTN_TP_RULES, _act_type_id,
    fuse_gdn_in_proj,
)
from ..kind_map import build_linear
from ..linear import Linear
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, DeltaNetConfig, FfnConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param, reorder_rotary_emb
```

- [ ] **Step 2: Delete the `reorder_rotary_emb` local function**

Delete lines 44-62 (the entire `reorder_rotary_emb` function definition).

- [ ] **Step 3: Update `model()`**

Replace:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens')
        root.norm = self.root_norm('model.norm')
        root.output = self.lm_head('lm_head')
        root.layers = self.layers(self._layer_prefix)
```

With:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        tie = self.cfg.get("tie_word_embeddings", False)
        lm_key = self._embed_key if tie else "lm_head.weight"
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)
```

Note: qwen3_5 uses `self._embed_key` and `self._norm_key` (model-specific keys set in `__init__`).

- [ ] **Step 4: Delete these methods from `Qwen3_5Spec`**
- `_cpp_dtype(self)`
- `token_embeds(self, pfx)`
- `lm_head(self, pfx)`
- `_linear(self, prefix)`
- `model_info(self)`

Do NOT delete `root_norm` and `norm` yet — they need to be rewritten (next step).

- [ ] **Step 5: Rewrite `norm()` and `output_norm()`**

Replace `root_norm` with `output_norm` (renamed), and rewrite `norm`. Both apply `_zero_centered`:

```python
    def output_norm(self, key):
        w = self._zero_centered(self._get(key))
        cfg = NormConfig(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        w = self._zero_centered(self._get(key))
        cfg = NormConfig(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m
```

Delete the old `root_norm(self, pfx)` method entirely.

- [ ] **Step 6: Remove None guards from `attn()`**

Same as Task 4 Step 5: remove the `if q is None and k is None and v is None and o is None: return None` guard and make qkv/o adds unconditional.

Also update the qk norm block to remove None guards:

```python
        q_norm = self._zero_centered(self._get(f"{pfx}.q_norm.weight"))
        k_norm = self._zero_centered(self._get(f"{pfx}.k_norm.weight"))
        if self._permute_qk:
            q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope_dim)
            k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope_dim)
        attn.add_qk_norm(q_norm, k_norm)
```

Note: the `if q_norm is not None and k_norm is not None` and `if q_norm is not None or k_norm is not None` guards are removed. If qk norm weights are missing, crash.

Wait — the `if self._permute_qk:` check IS structural (some models don't permute), keep it. Only remove the `is not None` checks.

- [ ] **Step 7: Remove None guards from `ffn()`**

Same pattern as Task 4 Step 6.

- [ ] **Step 8: Remove None guards from `moe()`**

Same pattern as Task 5 Step 6, but with qwen3_5's shared_expert_gate:

```python
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({"weight": gate_w}), model_dtype=dtype)
        sg = self._get(f'{pfx}.shared_expert_gate.weight')
        sg = sg.t() if sg.dim() > 1 else sg
        m.add_gate('shared_gate', Linear({"weight": sg}), model_dtype=dtype)
```

Note: for qwen3_5 MoE, both gate and shared_gate are always present. No None guards.

Remove the expert None guard too:
```python
        if expert is not None:
            experts[str(e)] = expert
```
Replace with:
```python
        experts[str(e)] = self._moe_expert_ffn(...)
```

- [ ] **Step 9: Update `layers()` norm calls to pass full keys**

Same as Task 4 Step 8 — append `.weight` to all `self.norm(...)` calls.

- [ ] **Step 10: Remove None guards from `linear_attn()`**

Remove the `if not la_linears: return None` guard.

Remove `if t is not None:` guards for inline params (A_log, dt_bias, conv1d, D, norm).

For conv1d: the shape check `if conv1d is not None and conv1d.ndim == 3 and conv1d.shape[1] == 1` is structural, keep it. But the outer `if conv1d is not None:` becomes unconditional (just process conv1d).

- [ ] **Step 11: Remove None guards from `_moe_expert_ffn()`**

Remove `if result is not None: return result` guard. Just return the result of `self.ffn(...)` directly, or fall through to packed format. Actually — this is a genuine fallback (try standard format first, fall back to packed). The check `if result is not None` is NOT a pointless None guard — it's a format probe. Keep it.

- [ ] **Step 12: Delete InputModel `__init__` from both `Qwen3_5InputModel` and `Qwen3_5MoeInputModel`**

Delete the `__init__` method from both classes.

- [ ] **Step 13: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(deploy): deduplicate qwen3_5_spec using base class methods"
```

---

### Task 7: Refactor `glm4_moe_lite_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

Same pattern as Task 4. No `reorder_rotary_emb` (MLA attention, no RoPE permutation).

- [ ] **Step 1: Update imports**

Replace:

```python
import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _act_type_id, _cpp_dtype as _cd,
)
from ..kind_map import build_linear
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import get_yarn_params, load_model_config, parse_rope_param
```

With:

```python
import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig,
    ModuleListConfig, MoeConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import get_yarn_params, load_model_config, parse_rope_param
```

Removed: `LinearBuilder`, `NormBuilder`, `SplitSide`, `_cpp_dtype as _cd`, `build_linear`, `pad_out_dim`, `LinearConfig`, `NormConfig`.

Wait — `build_linear` is NOT used elsewhere in glm4 (no packed expert helpers). Remove it.
`Linear` IS used in `moe()` for gate. Keep it.

- [ ] **Step 2: Update `model()`**

Replace:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens')
        root.norm = self.root_norm('model.norm')
        root.output = self.lm_head('lm_head')
        root.layers = self.layers('model.layers')
```

With:

```python
    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens.weight')
        root.norm = self.output_norm('model.norm.weight')
        root.output = self.lm_head('lm_head.weight')
        root.layers = self.layers('model.layers')
```

Note: glm4 does NOT have tie_word_embeddings, so no tie resolution needed.

- [ ] **Step 3: Delete these methods from `Glm4MoeLiteSpec`**
- `_cpp_dtype(self)`
- `token_embeds(self, pfx)`
- `root_norm(self, pfx)`
- `lm_head(self, pfx)`
- `norm(self, pfx)`
- `_linear(self, prefix)`
- `model_info(self)`

- [ ] **Step 4: Remove None guards from `attn()`**

Remove:
```python
        if not raw:
            return None
```

Remove `if lin is not None:` guard in the projection read loop:
```python
        for tm_name, hf_key in [
            ("q_a_proj", "q_a_proj"),
            ...
        ]:
            lin = self._linear(f"{pfx}.{hf_key}")
            raw[tm_name] = lin
```

Remove `if norm_tensor is not None:` guard on norm children:
```python
        for norm_name, norm_key in [
            ("q_a_layernorm", "q_a_layernorm.weight"),
            ("kv_a_layernorm", "kv_a_layernorm.weight"),
        ]:
            norm_tensor = self._get(f"{pfx}.{norm_key}")
            attn._add_norm_child(norm_name, norm_tensor, data_type=dtype)
```

Wait — q_a_layernorm and kv_a_layernorm may not be present in all GLM4 checkpoints. But the design says crash if missing. Follow the design.

- [ ] **Step 5: Remove None guards from `ffn()`**

Same pattern as Task 4 Step 6. Also note: glm4 `ffn()` uses `linears.get('w1')` etc. in `m.add_ffn()` — change to direct `m.add_ffn(w1, w2, w3)`.

- [ ] **Step 6: Remove None guards from `moe()`**

Remove None guard on gate_w:
```python
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {"weight": gate_w}
        gate_bias = self._get(f'{pfx}.gate.bias')
        if gate_bias is not None:
            tensors["bias"] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)
```

Note: keep the `gate_bias` None check — bias is genuinely optional for GLM4.

Remove None guard on score correction bias:
```python
        correction = self._get(f'{pfx}.gate.e_score_correction_bias')
        if correction is not None:
            m.add_param("score_correction_bias", correction)
```

Wait — score correction bias may genuinely be absent. But the design says crash. Follow the design: make unconditional. If it's always present for GLM4 MoE, this is correct.

Actually, let me reconsider. The design says "Remove all `if x is None: return None` from factory methods." The score correction bias check is `if correction is not None: m.add_param(...)` — it's not a return-None guard, it's a conditional add. This is more like the `gate_bias` pattern — genuinely optional field. Keep both `gate_bias` and `score_correction_bias` None checks.

Remove the expert None guard:
```python
        if expert is not None:
            experts[str(e)] = expert
```

- [ ] **Step 7: Update `layers()` norm calls to pass full keys**

Same as Task 4 Step 8.

- [ ] **Step 8: Delete InputModel `__init__` from `Glm4MoeLiteInputModel`**

Delete the `__init__` method.

- [ ] **Step 9: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(deploy): deduplicate glm4_moe_lite_spec using base class methods"
```

---

### Task 8: Verify with model tests

**Files:**
- No code changes — verification only.

- [ ] **Step 1: Build**

Run `ninja` from the `build` folder to ensure the project compiles.

- [ ] **Step 2: Test one model per spec**

Test at least one model for each of the 4 specs using the turbomind-tester agent or `scripts/test_turbomind_model.py`. Models to test (from the cached model list):

- qwen3_spec: Qwen3-8B or Qwen3-30B-A3B
- gpt_oss_spec: gpt-oss model
- qwen3_5_spec: Qwen3.5 model
- glm4_moe_lite_spec: GLM-4-Flash (GLM-4.7-Flash)

Each model must respond with meaningful text to a test prompt of at least 128 tokens.

- [ ] **Step 3: Check GPU availability before testing**

Use `get_gpu_usage` MCP tool to verify empty GPUs before each test run.
