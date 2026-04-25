# model_dtype Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the redundant `model_dtype` parameter threaded through every `_add_linear` and `_add_tensor` call by reading `self.config.data_type` internally instead.

**Architecture:** Single source of truth — `spec._dtype` → `config.data_type` on every C++ config → `self.config.data_type` read by builder methods. `model_dtype` parameter dropped from all signatures. `_add_linear` reads config; `_add_tensor` passes `alloc_dtype=None` preserving tensor native dtype.

**Tech Stack:** C++ (x-macro config system), Python (builder pattern), `_turbomind` C extension

---

### Task 1: Add `data_type` to `ModelWeightConfig` (C++)

**Files:**
- Modify: `src/turbomind/models/model_weight.h:16-18`
- Modify: `src/turbomind/models/model_weight.cc:10-15`

- [ ] **Step 1: Add `X(DataType, data_type)` to `MODEL_WEIGHT_FIELDS`**

In `src/turbomind/models/model_weight.h`, add the new field between `tp_size` and `tp_rank`:

```cpp
#define MODEL_WEIGHT_FIELDS(X) \
    X(int, tp_size) \
    X(DataType, data_type) \
    X(int, tp_rank)
```

- [ ] **Step 2: Copy `cfg.data_type` in `ModelWeight` constructor**

In `src/turbomind/models/model_weight.cc`, append to the member initializer list:

```cpp
ModelWeight::ModelWeight(const core::ModelWeightConfig& cfg)
    : tp_size(cfg.tp_size)
    , data_type(cfg.data_type)
    , tp_rank(cfg.tp_rank)
{
}
```

Note: `ModelWeight` already has a `DataType data_type{}` member (line 60 in the header). It is currently set in `prepare()` from the first attention layer. The constructor copy sets it early from config; `prepare()` will set the same value later.

- [ ] **Step 3: Build C++ to verify compilation**

Run: `ninja` from the `build` directory
Expected: Success (no compiler errors)

---

### Task 2: Remove `_infer_compute_dtype` and update `_add_linear` signature

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:92-114, 423-425, 455-456, 499-501`

- [ ] **Step 1: Remove `_infer_compute_dtype` function**

Delete lines 92-114 from `_base.py` (the entire `_infer_compute_dtype` function and its docstring). This function is only called at line 456 as a fallback when `model_dtype is None` — a path that no longer exists.

- [ ] **Step 2: Remove `model_dtype` parameter from `_add_linear`**

Change the signature at line 423-425 from:
```python
def _add_linear(self, name: str, linear: Linear,
                   split_side: SplitSide | None = None,
                   model_dtype=None):
```
To:
```python
def _add_linear(self, name: str, linear: Linear,
                   split_side: SplitSide | None = None):
```

- [ ] **Step 3: Replace `compute_dtype` derivation**

Replace lines 455-456:
```python
compute_dtype = (model_dtype if model_dtype is not None
                 else _infer_compute_dtype(linear))
```
With:
```python
compute_dtype = self.config.data_type
```

- [ ] **Step 4: Replace alloc_dtype fallback for weight tensors**

Replace lines 499-501:
```python
if alloc_dtype is None and kind == 'weight' \
        and model_dtype is not None:
    alloc_dtype = model_dtype
```
With:
```python
if alloc_dtype is None and kind == 'weight':
    alloc_dtype = self.config.data_type
```

---

### Task 3: Update `_add_tensor` and `_commit_tensor` signatures

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:511-513, 520, 558-559, 592-594, 616`

- [ ] **Step 1: Remove `model_dtype` from `_add_tensor` signature**

Change lines 511-513 from:
```python
def _add_tensor(self, name: str, tensor: torch.Tensor | None,
                   split_side: SplitSide | None = None, *,
                   model_dtype=None):
```
To:
```python
def _add_tensor(self, name: str, tensor: torch.Tensor | None,
                   split_side: SplitSide | None = None):
```

- [ ] **Step 2: Remove `model_dtype` from `_pending_tensors` storage**

Change line 520 from:
```python
self._pending_tensors[name] = (tensor, split_side, model_dtype)
```
To:
```python
self._pending_tensors[name] = (tensor, split_side)
```

- [ ] **Step 3: Remove `model_dtype` from drain loop in `build()`**

Change lines 558-559 from:
```python
for name, (tensor, split_side, model_dtype) in self._pending_tensors.items():
    self._commit_tensor(name, tensor, split_side, model_dtype)
```
To:
```python
for name, (tensor, split_side) in self._pending_tensors.items():
    self._commit_tensor(name, tensor, split_side)
```

- [ ] **Step 4: Remove `model_dtype` from `_commit_tensor` signature and body**

Change lines 592-594 from:
```python
def _commit_tensor(self, name: str, tensor: torch.Tensor,
                  split_side: SplitSide | None = None,
                  model_dtype=None):
```
To:
```python
def _commit_tensor(self, name: str, tensor: torch.Tensor,
                  split_side: SplitSide | None = None):
```

Delete the `model_dtype` parameter description from the docstring (lines 605-606):
```python
    model_dtype : C++ DataType value | None
        Override dtype for the C++ allocation.
```

Change line 616 from:
```python
                _copy_shard_to_param(handle, name, shard,
                                     alloc_dtype=model_dtype)
```
To:
```python
                _copy_shard_to_param(handle, name, shard,
                                     alloc_dtype=None)
```

---

### Task 4: Update `TextModelBuilder`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:637-642, 662-664, 684-686`

- [ ] **Step 1: Remove `data_type` from `TextModelBuilder.__init__`**

Change lines 637-642 from:
```python
def __init__(self, config, contexts, *, root_handles,
             tp, ranks, vocab_size, data_type):
    super().__init__(config=config, contexts=contexts, tp=tp, ranks=ranks)
    self._root_handles = root_handles
    self._vocab_size = vocab_size
    self._data_type = data_type
```
To:
```python
def __init__(self, config, contexts, *, root_handles,
             tp, ranks, vocab_size):
    super().__init__(config=config, contexts=contexts, tp=tp, ranks=ranks)
    self._root_handles = root_handles
    self._vocab_size = vocab_size
```

- [ ] **Step 2: Remove `model_dtype` from `add_token_embeds`**

Change lines 662-664 from:
```python
self._add_tensor('tok_embeddings', tensor,
                    split_side=SplitSide.OUTPUT,
                    model_dtype=self._data_type)
```
To:
```python
self._add_tensor('tok_embeddings', tensor,
                    split_side=SplitSide.OUTPUT)
```

- [ ] **Step 3: Remove `model_dtype` from `add_lm_head`**

Change lines 684-686 from:
```python
self._add_linear('output', padded,
                    split_side=SplitSide.OUTPUT,
                    model_dtype=self._data_type)
```
To:
```python
self._add_linear('output', padded,
                    split_side=SplitSide.OUTPUT)
```

---

### Task 5: Remove `model_dtype` from sub-builders

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:123-129`
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py:200-210`
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py:93, 100-101`
- Modify: `lmdeploy/turbomind/deploy/builder/moe.py:15-18`
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py:129-133`

- [ ] **Step 1: `attention.py` — remove `model_dtype=` from `add_qkv_proj` and `add_o_proj`**

`add_qkv_proj` (line 123-124):
```python
self._add_linear('w_qkv', merged, SplitSide.OUTPUT,
                    model_dtype=self.config.data_type)
```
→
```python
self._add_linear('w_qkv', merged, SplitSide.OUTPUT)
```

`add_o_proj` (line 128-129):
```python
self._add_linear('wo', o, SplitSide.INPUT,
                    model_dtype=self.config.data_type)
```
→
```python
self._add_linear('wo', o, SplitSide.INPUT)
```

Note: `dequant_mixed(q, k, v, gate, data_type=self.config.data_type)` at line 116 stays — this is a dataflow helper, not a builder method.

- [ ] **Step 2: `ffn.py` — remove `model_dtype=` from `add_ffn`**

Remove the local variable at line 200:
```python
model_dtype = self.config.data_type
```

Remove `model_dtype=model_dtype` from all four `_add_linear` calls:
- Line 202-203: `self._add_linear('w1w3', fused, SplitSide.OUTPUT)`
- Line 205-206: `self._add_linear('w1', w1, SplitSide.OUTPUT)`
- Line 207-208: `self._add_linear('w3', w3, SplitSide.OUTPUT)`
- Line 209-210: `self._add_linear('w2', w2, SplitSide.INPUT)`

- [ ] **Step 3: `mla.py` — remove `model_dtype=` from `add_projections`**

Remove the local variable at line 93:
```python
model_dtype = self.config.data_type
```

Remove `model_dtype=model_dtype` from the `_add_linear` call in the loop (line 100-101):
```python
self._add_linear(name, lin, split_side=side)
```

- [ ] **Step 4: `moe.py` — remove `model_dtype` parameter from `add_gate`**

Change lines 15-18 from:
```python
def add_gate(self, name, linear, model_dtype=None):
    """Commit a gate linear (broadcast, no split)."""
    self._add_linear(name, linear, split_side=None,
                        model_dtype=model_dtype)
```
To:
```python
def add_gate(self, name, linear):
    """Commit a gate linear (broadcast, no split)."""
    self._add_linear(name, linear, split_side=None)
```

- [ ] **Step 5: `deltanet.py` — remove `model_dtype=` from `add_input_projections`**

`in_proj_all` (line 129-130):
```python
self._add_linear("in_proj_all", fused, SplitSide.OUTPUT,
                    model_dtype=self.config.data_type)
```
→
```python
self._add_linear("in_proj_all", fused, SplitSide.OUTPUT)
```

`out_proj` (line 132-133):
```python
self._add_linear("out_proj", out_proj, SplitSide.INPUT,
                    model_dtype=self.config.data_type)
```
→
```python
self._add_linear("out_proj", out_proj, SplitSide.INPUT)
```

---

### Task 6: Store `self._dtype` in `TextModelSpec`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:61, 167`

- [ ] **Step 1: Add `self._dtype` to base `TextModelSpec.__init__`**

In `lmdeploy/turbomind/deploy/spec.py`, add one line after `self._resolver = resolver` (line 60):

```python
self._resolver = resolver
self._dtype = self._cpp_dtype()
self._parse_base(hf_cfg)
```

- [ ] **Step 2: Use `self._dtype` as `norm()` default**

Change line 167 from:
```python
data_type=data_type if data_type is not None else self._cpp_dtype(),
```
To:
```python
data_type=data_type if data_type is not None else self._dtype,
```

---

### Task 7: Update model specs — Qwen3

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:41, 104-110, 179-180`

- [ ] **Step 1: Use `self._dtype` in `__init__`**

Change line 41 from:
```python
dtype = self._cpp_dtype()
```
To:
```python
dtype = self._dtype
```

- [ ] **Step 2: Set `cfg.data_type` on `ModelWeightConfig` and remove `data_type=` from `TextModelBuilder`**

In the `model()` method (lines 102-110), add `cfg.data_type = self._dtype` and remove the `data_type=` kwarg:

```python
def model(self):
    ec = self.engine_cfg
    cfg = _tm.ModelWeightConfig()
    cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
    cfg.data_type = self._dtype
    root = TextModelBuilder(
        cfg, self._contexts,
        root_handles=self._root_handles,
        tp=ec.attn_tp_size * ec.attn_cp_size,
        ranks=self._model_tp_ranks,
        vocab_size=self._vocab_size)
```

- [ ] **Step 3: Remove `model_dtype=` from `add_gate()` call**

In the `moe()` method, change lines 179-180 from:
```python
m.add_gate('gate', self._linear(f'{pfx}.gate'),
           model_dtype=self._cpp_dtype())
```
To:
```python
m.add_gate('gate', self._linear(f'{pfx}.gate'))
```

---

### Task 8: Update model specs — Qwen3.5

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:50, 145-151, 267-271`

- [ ] **Step 1: Use `self._dtype` in `__init__`**

Change line 50 from:
```python
dtype = self._cpp_dtype()
```
To:
```python
dtype = self._dtype
```

- [ ] **Step 2: Set `cfg.data_type` on `ModelWeightConfig` and remove `data_type=` from `TextModelBuilder`**

In the `model()` method (lines 141-151), add `cfg.data_type = self._dtype` and remove the `data_type=` kwarg:

```python
def model(self):
    ec = self.engine_cfg
    cfg = _tm.ModelWeightConfig()
    cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
    cfg.data_type = self._dtype
    root = TextModelBuilder(
        cfg, self._contexts,
        root_handles=self._root_handles,
        tp=ec.attn_tp_size * ec.attn_cp_size,
        ranks=self._model_tp_ranks,
        vocab_size=self._vocab_size)
```

- [ ] **Step 3: Remove `model_dtype=` from both `add_gate()` calls**

Lines 267-268 from:
```python
m.add_gate('gate', self._linear(f'{pfx}.gate'),
           model_dtype=self._cpp_dtype())
```
To:
```python
m.add_gate('gate', self._linear(f'{pfx}.gate'))
```

Lines 270-271 from:
```python
m.add_gate('shared_gate', self._linear(f'{pfx}.shared_expert_gate'),
           model_dtype=self._cpp_dtype())
```
To:
```python
m.add_gate('shared_gate', self._linear(f'{pfx}.shared_expert_gate'))
```

---

### Task 9: Update model specs — GPT-OSS

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:43, 109-117, 186-187`

- [ ] **Step 1: Use `self._dtype` in `__init__`**

Change line 43 from:
```python
dtype = self._cpp_dtype()
```
To:
```python
dtype = self._dtype
```

- [ ] **Step 2: Set `cfg.data_type` on `ModelWeightConfig` and remove `data_type=` from `TextModelBuilder`**

In the `model()` method (lines 107-117), add `cfg.data_type = self._dtype` and remove the `data_type=` kwarg:

```python
def model(self):
    ec = self.engine_cfg
    cfg = _tm.ModelWeightConfig()
    cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
    cfg.data_type = self._dtype
    root = TextModelBuilder(
        cfg, self._contexts,
        root_handles=self._root_handles,
        tp=ec.attn_tp_size * ec.attn_cp_size,
        ranks=self._model_tp_ranks,
        vocab_size=self._vocab_size)
```

- [ ] **Step 3: Remove `model_dtype=` from `add_gate()` call**

Lines 186-187 from:
```python
m.add_gate('gate', self._linear(f'{pfx}.router'),
           model_dtype=self._cpp_dtype())
```
To:
```python
m.add_gate('gate', self._linear(f'{pfx}.router'))
```

---

### Task 10: Update model specs — GLM-4 MoE Lite

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:33, 150-158, 225-226`

- [ ] **Step 1: Use `self._dtype` in `__init__`**

Change line 33 from:
```python
dtype = self._cpp_dtype()
```
To:
```python
dtype = self._dtype
```

- [ ] **Step 2: Set `cfg.data_type` on `ModelWeightConfig` and remove `data_type=` from `TextModelBuilder`**

In the `model()` method (lines 148-158), add `cfg.data_type = self._dtype` and remove the `data_type=` kwarg:

```python
def model(self):
    ec = self.engine_cfg
    cfg = _tm.ModelWeightConfig()
    cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
    cfg.data_type = self._dtype
    root = TextModelBuilder(
        cfg, self._contexts,
        root_handles=self._root_handles,
        tp=ec.attn_tp_size * ec.attn_cp_size,
        ranks=self._model_tp_ranks,
        vocab_size=self._vocab_size)
```

- [ ] **Step 3: Remove `model_dtype=` from `add_gate()` call**

Lines 225-226 from:
```python
m.add_gate('gate', self._linear(f'{pfx}.gate'),
           model_dtype=self._cpp_dtype())
```
To:
```python
m.add_gate('gate', self._linear(f'{pfx}.gate'))
```

---

### Task 11: Build and test

- [ ] **Step 1: Build C++**

Run: `ninja` from the `build` directory
Expected: Success

- [ ] **Step 2: Find model paths and cache dirs**

Use the `list_models` and `get_model_cache_path` MCP tools to find available Qwen3 and Qwen3-MoE models and their cache directories.

- [ ] **Step 3: Verify no remaining `model_dtype` references**

```bash
grep -rn "model_dtype" lmdeploy/turbomind/deploy/ --include="*.py"
```

Expected: zero matches. If any remain (e.g., in comment or unrelated), verify they are not parameter-passing usages.

- [ ] **Step 4: Run model test — dense (e.g., Qwen3) with TP=1**

Check `get_gpu_usage` for an empty GPU first, then:

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> 1 <gpu_id>
```

Note: `<model_path>`, `<cache_dir>`, and `<gpu_id>` come from Step 2 and `get_gpu_usage`.

Expected: model loads and produces meaningful text (at least 128 tokens, non-gibberish). Check the `--- response begin ---` section in stdout.

- [ ] **Step 5: Run model test — MoE (e.g., Qwen3-MoE) with TP=1**

Same workflow as Step 4 with a MoE variant model.

Expected: model loads and produces meaningful text (at least 128 tokens, non-gibberish).

---

### Task 12: Commit

- [ ] **Step 1: Stage and commit all changes**

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc
git add lmdeploy/turbomind/deploy/
git commit -m "$(cat <<'EOF'
refactor: remove redundant model_dtype threading from builder pipeline

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```
