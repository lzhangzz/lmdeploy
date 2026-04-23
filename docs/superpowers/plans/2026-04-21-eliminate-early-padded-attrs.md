# Eliminate Early Padded Attributes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `_kv_head_num_padded`, `_expert_inter_size_padded`, `_inter_sizes_padded` from model specs; C++ weight classes derive padded dimensions from actual weight tensors in `prepare()`.

**Architecture:** Specs store raw HF values. Python builders pad weight tensors for TP sharding. C++ constructors store raw values without division; `prepare()` computes effective per-rank dimensions from weight children.

**Tech Stack:** C++ (weight classes), Python (specs, builders), math.lcm for block alignment.

---

### Task 1: C++ FfnWeight — defer inter_size to prepare()

**Files:**
- Modify: `src/turbomind/models/ffn_weight.cc`
- Modify: `src/turbomind/models/ffn_weight.h`

- [ ] **Step 1: Update FfnWeight constructor to store raw inter_size**

In `src/turbomind/models/ffn_weight.cc`, replace the constructor body:

```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , inter_size_{cfg.inter_size}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu}
    , is_fused_moe_{cfg.fused_moe}
{
}
```

Removed: `TM_CHECK(inter_size_ % tp_size_ == 0)` and `inter_size_ /= tp_size_`.

- [ ] **Step 2: Update FfnWeight::prepare() to derive inter_size from weights**

Replace the `prepare()` body in `src/turbomind/models/ffn_weight.cc`:

```cpp
void FfnWeight::prepare()
{
    // Derive per-rank inter_size from actual weight dimensions.
    // Weight tensors are already TP-sharded by the Python builder,
    // so w1/w3 output_dim (or w1w3 output_dim) equals per-rank inter_size.
    if (w1w3) {
        inter_size_ = w1w3->output_dim;
    } else if (w1) {
        inter_size_ = w1->output_dim;
    }

    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Propagate grouped-GEMM flag for MoE expert weights
    if (is_fused_moe_) {
        auto set_grouped = [](const char*, Module* m) {
            if (auto* lw = dynamic_cast<LinearWeight*>(m)) {
                lw->set_grouped(true);
            }
        };
        for_each_child(set_grouped);
    }

    Module::prepare();  // recurse into children
}
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd build && ninja`

Expected: Clean build, no errors.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/ffn_weight.cc
git commit -m "refactor: defer FfnWeight inter_size to prepare()"
```

---

### Task 2: C++ MoeWeight — defer inter_size to prepare()

**Files:**
- Modify: `src/turbomind/models/moe_weight.cc`

- [ ] **Step 1: Update MoeWeight constructor to store raw inter_size**

In `src/turbomind/models/moe_weight.cc`, change the constructor to store raw inter_size without dividing:

```cpp
MoeWeight::MoeWeight(const core::MoeConfig& cfg)
{
    layer_id_ = cfg.layer_id;
    method_ = static_cast<MoeMethod>(cfg.method);
    experts_per_token = cfg.experts_per_token;
    norm_topk_prob = cfg.norm_topk_prob;
    use_shared_gate = cfg.shared_gate;
    routed_scale = static_cast<float>(cfg.routed_scale);
    router_bias = cfg.router_bias;
    topk_group = cfg.topk_group;
    topk_method = cfg.topk_method;
    n_group = cfg.n_group;
    scoring_func = cfg.scoring_func;
    router_n_groups = cfg.router_n_groups;
    hidden_dim = cfg.hidden_dim;
    inter_size = cfg.inter_size;  // raw, not divided by tp
    mlp_bias_ = cfg.mlp_bias;
    data_type_ = cfg.data_type;
    tp_size_ = cfg.tp_size;
    tp_rank_ = cfg.tp_rank;
    act_type_ = static_cast<ActivationType>(cfg.act_type);
    fuse_silu_act_ = cfg.fuse_silu;
    expert_num = cfg.expert_num;
}
```

Changed: `inter_size = cfg.inter_size / cfg.tp_size` → `inter_size = cfg.inter_size`.

- [ ] **Step 2: Update MoeWeight::prepare() to derive inter_size from expert children**

In `src/turbomind/models/moe_weight.cc`, update `prepare()`:

```cpp
void MoeWeight::prepare()
{
    // First prepare all children (experts, gate, etc.)
    Module::prepare();

    // Create batched block view for fused MoE path
    if (expert_num > 0 && method() == MoeMethod::kFused) {
        // Derive per-rank inter_size from first expert's weights.
        // Each expert FfnWeight's prepare() has already computed its
        // per-rank inter_size from w1/w3 output dimensions.
        if (auto* e0 = expert(0)) {
            inter_size = e0->inter_size();
        }

        core::FfnConfig block_cfg;
        block_cfg.hidden_dim = hidden_dim;
        block_cfg.inter_size = inter_size * tp_size_;
        block_cfg.has_bias   = mlp_bias_;
        block_cfg.tp_size    = tp_size_;
        block_cfg.tp_rank    = tp_rank_;
        block_cfg.data_type  = data_type_;
        block_cfg.act_type   = static_cast<int>(act_type_);
        block_cfg.fuse_silu  = fuse_silu_act_;
        block_ = std::make_unique<FfnWeight>(block_cfg);

        // Link each linear in the block to the corresponding expert linears
        auto get_expert_w1w3 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w1w3.get() : nullptr;
        };
        auto get_expert_w1 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w1.get() : nullptr;
        };
        auto get_expert_w3 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w3.get() : nullptr;
        };
        auto get_expert_w2 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w2.get() : nullptr;
        };

        if (get_expert_w1w3(0)) {
            block_->add_child("w1w3", std::make_unique<LinearWeight>());
            LinkLinearExperts(get_expert_w1w3, expert_num, *block_->w1w3);
        }
        else {
            block_->add_child("w1", std::make_unique<LinearWeight>());
            block_->add_child("w3", std::make_unique<LinearWeight>());
            if (get_expert_w1(0)) {
                LinkLinearExperts(get_expert_w1, expert_num, *block_->w1);
            }
            if (get_expert_w3(0)) {
                LinkLinearExperts(get_expert_w3, expert_num, *block_->w3);
            }
        }

        block_->add_child("w2", std::make_unique<LinearWeight>());
        if (get_expert_w2(0)) {
            LinkLinearExperts(get_expert_w2, expert_num, *block_->w2);
        }

        if (auto* e0 = expert(0)) {
            block_->set_fused_silu(e0->is_fused_silu());
        }
    }
}
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd build && ninja`

Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/moe_weight.cc
git commit -m "refactor: defer MoeWeight inter_size to prepare()"
```

---

### Task 3: C++ AttentionWeight — derive kv_head_num from weights in prepare()

**Files:**
- Modify: `src/turbomind/models/attention_weight.cc`

The w_qkv output_dim is per-shard (after TP split by Python builder).
Per-shard layout: `(head_num + 2 * padded_kv_heads) / tp` heads.
So: `padded_kv_heads = (local_total * tp - head_num) / 2`.

- [ ] **Step 1: Update AttentionWeight::prepare() to derive kv_head_num from weights**

Replace `prepare()` in `src/turbomind/models/attention_weight.cc`:

```cpp
void AttentionWeight::prepare()
{
    Module::prepare();

    // Derive kv_head_num from actual weight tensor dimensions.
    // Python's repeat_kv_for_tp() physically pads KV heads, and w_qkv
    // is TP-sharded, so output_dim is per-shard.
    int local_total = w_qkv->output_dim / head_dim;
    kv_head_num = (local_total * tp_size - head_num) / 2;
}
```

- [ ] **Step 2: Build and verify compilation**

Run: `cd build && ninja`

Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/attention_weight.cc
git commit -m "refactor: derive AttentionWeight kv_head_num from weights in prepare()"
```

---

### Task 4: Python — remove _kv_head_num_padded from spec base

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

- [ ] **Step 1: Remove _kv_head_num_padded from spec.py**

In `lmdeploy/turbomind/deploy/spec.py`:

1. Remove `_pad_kv_head` from the import on line 15:
```python
from .source_model.utils import (detect_layer_prefix,
                                 parse_rope_param, rope_type_to_int)
```

2. Remove lines 99-100 (`self._kv_head_num_padded = _pad_kv_head(...)`).

3. Update the docstring at line 74-78 to remove `_kv_head_num_padded` from the populated list. Change to:
```python
        Populated:
          _num_layer, _vocab_size, _norm_eps, _head_num, _kv_head_num,
          _head_dim, _hidden_units, _rope,
```

4. Update the `_parse_base` docstring comment at line 57-58 to remove the group_size padding mention:
```python
        ``group_size`` is the quantization group size the converter resolves
        from ``engine_cfg.model_format`` plus any user override. It lands on
        ``self._group_size`` so build_linear() can use it during weight loading.
```

- [ ] **Step 2: Update all 4 spec subclasses to use _kv_head_num instead of _kv_head_num_padded**

In each spec file, replace `self._kv_head_num_padded` with `self._kv_head_num`:

- `qwen3_spec.py` line 51: `self._attn_cfg.kv_head_num = self._kv_head_num`
- `qwen3_5_spec.py` line 65: `self._attn_cfg.kv_head_num = self._kv_head_num`
- `gpt_oss_spec.py` line 57: `self._attn_cfg.kv_head_num = self._kv_head_num`
- `glm4_moe_lite_spec.py` line 55: remove `self._kv_head_num_padded = _pad_kv_head(1, engine_cfg.attn_tp_size)` and on line 78: `self._attn_cfg.kv_head_num = self._kv_head_num`

For `glm4_moe_lite_spec.py`, also remove `_pad_kv_head` from the import on line 14:
```python
from .utils import get_yarn_params, layer_progress, parse_rope_param
```

- [ ] **Step 3: Verify no references to _kv_head_num_padded remain**

Run: `grep -rn '_kv_head_num_padded' lmdeploy/`

Expected: No matches.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/deploy/source_model/qwen3_spec.py lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor: remove _kv_head_num_padded from specs"
```

---

### Task 5: Python — remove _inter_sizes_padded and _expert_inter_size_padded from specs

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

- [ ] **Step 1: Update qwen3_spec.py**

1. Remove `_pad_inter_size` from the import on line 19:
```python
from .utils import layer_progress, reorder_rotary_emb, reorder_rotary_emb_linear
```

2. Replace lines 90-94 (expert inter size):
```python
            self._expert_inter_size = hf_cfg.get('moe_intermediate_size', 768)
        else:
            self._expert_inter_size = 0
```

3. Replace lines 97-102 (inter sizes):
```python
        raw_inter = hf_cfg.get('intermediate_size', 0) if self._n_experts == 0 else 0
        self._inter_sizes = [raw_inter] * self._num_layer
```

4. In `ffn()` method (line 170-171), change:
```python
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
```

5. In `moe()` method (line 188), change:
```python
        cfg.inter_size = self._expert_inter_size
```

6. In `moe()` method (line 203), change:
```python
                inter_size=self._expert_inter_size, fused_moe=True)
```

- [ ] **Step 2: Update qwen3_5_spec.py**

1. Remove `_pad_inter_size` from import on line 20:
```python
from .utils import layer_progress, reorder_rotary_emb, reorder_rotary_emb_linear
```

2. Replace lines 127-133 (expert inter size):
```python
            self._expert_inter_size = hf_cfg['moe_intermediate_size']
            raw_shared = hf_cfg.get('shared_expert_intermediate_size', 0)
        else:
            self._expert_inter_size = 0
            raw_shared = hf_cfg.get('intermediate_size', 0)
```

3. Replace lines 136-140 (inter sizes):
```python
        self._inter_sizes = [raw_shared] * self._num_layer
```

4. In `ffn()` method (line 248-249), change:
```python
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
```

5. In `moe()` method (line 266), change:
```python
        cfg.inter_size = self._expert_inter_size
```

6. In `moe()` method (line 284), change:
```python
            experts[str(e)] = self._moe_expert_ffn(
                pfx, layer, e, self._expert_inter_size)
```

- [ ] **Step 3: Update gpt_oss_spec.py**

1. Remove `_pad_inter_size` from import on line 19:
```python
from .utils import layer_progress, reorder_rotary_emb_linear
```

2. Replace lines 94-96 (expert inter size):
```python
        self._expert_inter_size = hf_cfg['intermediate_size']
```

3. Replace line 106 (inter sizes):
```python
        self._inter_sizes = [0] * self._num_layer
```

4. In `ffn()` method (line 171-172), change:
```python
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
```

5. In `moe()` method (line 189), change:
```python
        cfg.inter_size = self._expert_inter_size
```

6. In `moe()` method (line 207), change:
```python
            experts[str(e)] = self._packed_expert_ffn(
                f'{pfx}.experts.{e}', self._expert_inter_size)
```

- [ ] **Step 4: Update glm4_moe_lite_spec.py**

1. Remove `_pad_inter_size` from import on line 14:
```python
from .utils import get_yarn_params, layer_progress, parse_rope_param
```

2. Replace lines 123-127 (expert inter size):
```python
            self._expert_inter_size = hf_cfg['moe_intermediate_size']
        else:
            self._expert_inter_size = 0
```

3. Replace lines 137-140 (inter sizes):
```python
        self._inter_sizes = raw_inter
```

4. In `ffn()` method (line 214-215), change:
```python
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
```

5. In `moe()` method (line 232), change:
```python
        cfg.inter_size = self._expert_inter_size
```

6. In `moe()` method (line 254), change:
```python
                inter_size=self._expert_inter_size, fused_moe=True)
```

- [ ] **Step 5: Verify no references to padded attrs remain**

Run: `grep -rn '_inter_sizes_padded\|_expert_inter_size_padded' lmdeploy/`

Expected: No matches.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor: remove _inter_sizes_padded and _expert_inter_size_padded from specs"
```

---

### Task 6: Python — add weight padding to FfnBuilder

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py`

- [ ] **Step 1: Add pad_for_tp helper to ffn.py**

Add `import math` at the top of `lmdeploy/turbomind/deploy/builder/ffn.py`.

Update the existing `from ..linear import ...` line to include `pad_in_dim` and `pad_out_dim`:

```python
from ..linear import (Linear, chunk_linears as _chunk_linears,
                       interleave_linears as _interleave_linears,
                       pad_in_dim, pad_out_dim)
```

Add new import after existing imports:

```python
from ..source_model.utils import _pad_inter_size
```

Add a helper function before the FfnBuilder class:

```python
def _pad_ffn_for_tp(w1: Linear, w2: Linear, w3: Linear,
                     tp: int) -> tuple[Linear, Linear, Linear, int]:
    """Pad w1/w3 output dim and w2 input dim for TP sharding.

    Returns the padded (w1, w2, w3) and the padded inter_size.
    Padding uses lcm(block_in, block_out) * tp as the alignment target.
    """
    if tp <= 1:
        w = w1.tensors.get('weight') or w3.tensors.get('weight')
        raw_inter = w.size(-1) if w is not None else 0
        return w1, w2, w3, raw_inter

    w = w1.tensors.get('weight') or w3.tensors.get('weight')
    if w is None:
        return w1, w2, w3, 0

    raw_inter = w.size(-1)
    fmt = w1.weight_format
    block_out = (fmt.block_out or 1) if fmt else 1
    block_in = (fmt.block_in or 1) if fmt else 1
    effective_block = math.lcm(block_in, block_out) if block_in != block_out else block_out

    padded_inter = _pad_inter_size(raw_inter, effective_block, tp)
    if padded_inter == raw_inter:
        return w1, w2, w3, raw_inter

    # Pad w1/w3 output dim (axis -1)
    def _pad_linear_out(lin: Linear, target: int) -> Linear:
        new_tensors = {}
        for kind, t in lin.tensors.items():
            dim = t.dim() - 1
            new_tensors[kind] = pad_out_dim(t, target, dim=dim)
        return Linear(tensors=new_tensors,
                       weight_format=lin.weight_format,
                       data_format=lin.data_format)

    # Pad w2 input dim (axis 0)
    def _pad_linear_in(lin: Linear, target: int) -> Linear:
        new_tensors = {}
        for kind, t in lin.tensors.items():
            if t.dim() < 2:
                new_tensors[kind] = t
            else:
                new_tensors[kind] = pad_in_dim(t, target, dim=0)
        return Linear(tensors=new_tensors,
                       weight_format=lin.weight_format,
                       data_format=lin.data_format)

    w1 = _pad_linear_out(w1, padded_inter)
    w3 = _pad_linear_out(w3, padded_inter)
    w2 = _pad_linear_in(w2, padded_inter)
    return w1, w2, w3, padded_inter
```

- [ ] **Step 2: Update FfnBuilder.add_ffn() to call _pad_ffn_for_tp**

Replace the `add_ffn` method in the FfnBuilder class:

```python
    def add_ffn(self, w1, w2, w3):
        """Pad weights for TP alignment, fuse w1+w3 if possible, then shard and commit."""
        # Pad weights for TP alignment before any fusion or sharding
        w1, w2, w3, padded_inter = _pad_ffn_for_tp(
            w1, w2, w3, self._tp)

        fused = None
        fused_silu = False
        if w1 is not None and w3 is not None:
            act_type = getattr(self.config, 'act_type', 0)
            if isinstance(act_type, int):
                act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self._tp, act_type,
                is_moe=getattr(self.config, 'fused_moe', False))

        self.config.fuse_silu = fused_silu

        model_dtype = self.config.data_type
        if fused is not None:
            self._commit_linear('w1w3', fused, SplitSide.OUTPUT,
                                model_dtype=model_dtype)
        else:
            if w1 is not None:
                self._commit_linear('w1', w1, SplitSide.OUTPUT,
                                    model_dtype=model_dtype)
            if w3 is not None:
                self._commit_linear('w3', w3, SplitSide.OUTPUT,
                                    model_dtype=model_dtype)
        if w2 is not None:
            self._commit_linear('w2', w2, SplitSide.INPUT,
                                model_dtype=model_dtype)
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/ffn.py
git commit -m "feat: add weight padding to FfnBuilder for TP sharding"
```

---

### Task 7: Test with a model

**Files:** None (verification only)

- [ ] **Step 1: Check GPU availability**

Run the `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 2: Build**

Run: `cd build && ninja`

- [ ] **Step 3: Test a dense model (Qwen3)**

Run the test script with a Qwen3 model (e.g. `Qwen/Qwen3-8B`) with at least 128 tokens. Verify the response contains meaningful words.

- [ ] **Step 4: Test an MoE model**

Run the test script with an MoE model (e.g. Qwen3 MoE or gpt-oss). Verify the response.

- [ ] **Step 5: Fix any bugs**

If any test fails, debug and fix. The model must produce coherent text.

- [ ] **Step 6: Commit fixes if any**

```bash
git add -u
git commit -m "fix: address issues from model testing"
```
