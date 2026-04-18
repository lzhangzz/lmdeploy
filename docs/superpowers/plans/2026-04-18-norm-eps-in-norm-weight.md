# Move norm_eps into NormWeight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `norm_eps` from `ModelParam` into `NormWeight`. Each NormWeight instance carries its own eps. Layer constructors lose their `norm_eps` parameter and cached member. No behavioral changes.

**Architecture:** Add `norm_eps` to the `NormConfig` X-macro and `NormWeight` public field. Thread it through the Python builder layer (`make_norm_config` → `_add_norm_child` → builder methods → spec files). Remove cached `norm_eps_`/`rmsnorm_eps_` members from 3 C++ layers. At each `invokeRMSNorm` call site, read eps from the NormWeight being used.

**Tech Stack:** C++17, CUDA, Python, ninja build

---

### Task 1: Add norm_eps to NormConfig X-macro and NormWeight

**Files:**
- Modify: `src/turbomind/models/norm_weight.h` (lines 12–14, line 55)
- Modify: `src/turbomind/models/norm_weight.cc` (line 28)

- [ ] **Step 1: Add norm_eps to NormConfig X-macro**

In `src/turbomind/models/norm_weight.h`, replace lines 12–14:

```cpp
    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type)
```

With:

```cpp
    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type) \
        X(float,    norm_eps, 0.f)
```

- [ ] **Step 2: Add public norm_eps_ field to NormWeight**

In `src/turbomind/models/norm_weight.h`, add `float norm_eps_{};` after the `TM_MODULE_DECLARE` line (53) and before `private:` (55):

```cpp
    TM_MODULE_DECLARE(NormWeight, NORM_WEIGHT_CHILDREN, NORM_WEIGHT_PARAMS)

    float norm_eps_{};

private:
```

- [ ] **Step 3: Populate norm_eps_ from config in constructor**

In `src/turbomind/models/norm_weight.cc`, replace the constructor (lines 26–29):

```cpp
NormWeight::NormWeight(const core::NormConfig& cfg)
{
    configure(cfg.dim, cfg.data_type);
}
```

With:

```cpp
NormWeight::NormWeight(const core::NormConfig& cfg)
{
    configure(cfg.dim, cfg.data_type);
    norm_eps_ = cfg.norm_eps;
}
```

- [ ] **Step 4: Build to verify no breakage**

Run: `cd build && ninja`
Expected: clean build (no code reads `norm_eps_` externally yet)

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/norm_weight.h src/turbomind/models/norm_weight.cc
git commit -m "refactor(norm): add norm_eps to NormConfig X-macro and NormWeight public field"
```

---

### Task 2: Update Python builder and spec files to pass norm_eps

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/norm.py` (lines 9–13)
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py` (lines 476–506)
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py` (lines 266–271)
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py` (lines 103–108)
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py` (lines 142–144)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (lines 127, 161)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (lines 179, 216, 238–239)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (line 134)
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (lines 170, 197–201)

- [ ] **Step 1: Update make_norm_config**

In `lmdeploy/turbomind/deploy/builder/norm.py`, replace lines 9–13:

```python
def make_norm_config(*, dim, data_type):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    return cfg
```

With:

```python
def make_norm_config(*, dim, data_type, norm_eps):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    cfg.norm_eps = norm_eps
    return cfg
```

- [ ] **Step 2: Update _add_norm_child**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace lines 476–493:

```python
    def _add_norm_child(self, name: str, tensor: torch.Tensor,
                        data_type=None):
        """Create a NormConfig child and commit weight tensor.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"attention_norm"``).
        tensor : torch.Tensor
            The norm weight tensor.
        data_type : C++ DataType value | None
            Compute dtype for the norm.  Defaults to FP32 if not set.
        """
        self._ensure_handles()
        from .norm import make_norm_config
        if data_type is None:
            data_type = _tm.DataType.TYPE_FP32
        norm_cfg = make_norm_config(dim=tensor.shape[-1], data_type=data_type)
```

With:

```python
    def _add_norm_child(self, name: str, tensor: torch.Tensor,
                        data_type=None, *, norm_eps):
        """Create a NormConfig child and commit weight tensor.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"attention_norm"``).
        tensor : torch.Tensor
            The norm weight tensor.
        data_type : C++ DataType value | None
            Compute dtype for the norm.  Defaults to FP32 if not set.
        norm_eps : float
            RMS norm epsilon.  Required.
        """
        self._ensure_handles()
        from .norm import make_norm_config
        if data_type is None:
            data_type = _tm.DataType.TYPE_FP32
        norm_cfg = make_norm_config(dim=tensor.shape[-1], data_type=data_type, norm_eps=norm_eps)
```

- [ ] **Step 3: Update AttentionBuilder.add_qk_norm**

In `lmdeploy/turbomind/deploy/builder/attention.py`, replace lines 266–271:

```python
    def add_qk_norm(self, q, k):
        """Create NormConfig children for q_norm, k_norm, commit tensors."""
        if q is not None:
            self._add_norm_child('q_norm', q, data_type=self.config.data_type)
        if k is not None:
            self._add_norm_child('k_norm', k, data_type=self.config.data_type)
```

With:

```python
    def add_qk_norm(self, q, k, *, norm_eps):
        """Create NormConfig children for q_norm, k_norm, commit tensors."""
        if q is not None:
            self._add_norm_child('q_norm', q, data_type=self.config.data_type, norm_eps=norm_eps)
        if k is not None:
            self._add_norm_child('k_norm', k, data_type=self.config.data_type, norm_eps=norm_eps)
```

- [ ] **Step 4: Update MLABuilder.add_norms**

In `lmdeploy/turbomind/deploy/builder/mla.py`, replace lines 103–108:

```python
    def add_norms(self, *, q_a_norm, kv_a_norm, data_type):
        """Create norm children for q_a_layernorm and kv_a_layernorm."""
        self._add_norm_child('q_a_layernorm', q_a_norm,
                             data_type=data_type)
        self._add_norm_child('kv_a_layernorm', kv_a_norm,
                             data_type=data_type)
```

With:

```python
    def add_norms(self, *, q_a_norm, kv_a_norm, data_type, norm_eps):
        """Create norm children for q_a_layernorm and kv_a_layernorm."""
        self._add_norm_child('q_a_layernorm', q_a_norm,
                             data_type=data_type, norm_eps=norm_eps)
        self._add_norm_child('kv_a_layernorm', kv_a_norm,
                             data_type=data_type, norm_eps=norm_eps)
```

- [ ] **Step 5: Update DeltaNetBuilder.add_norm**

In `lmdeploy/turbomind/deploy/builder/deltanet.py`, replace lines 142–144:

```python
    def add_norm(self, norm_weight, data_type):
        """Add inline norm child."""
        self._add_norm_child("norm", norm_weight, data_type=data_type)
```

With:

```python
    def add_norm(self, norm_weight, data_type, *, norm_eps):
        """Add inline norm child."""
        self._add_norm_child("norm", norm_weight, data_type=data_type, norm_eps=norm_eps)
```

- [ ] **Step 6: Update qwen3_spec.py**

In `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`:

Replace line 127:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
```
With:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
```

Replace line 161:
```python
        attn.add_qk_norm(q_norm, k_norm)
```
With:
```python
        attn.add_qk_norm(q_norm, k_norm, norm_eps=self._norm_eps)
```

- [ ] **Step 7: Update qwen3_5_spec.py**

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`:

Replace line 179:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
```
With:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
```

Replace line 216:
```python
        attn.add_qk_norm(q_norm, k_norm)
```
With:
```python
        attn.add_qk_norm(q_norm, k_norm, norm_eps=self._norm_eps)
```

Replace lines 238–239:
```python
        builder.add_norm(
            self._get(f'{pfx}.norm.weight'), data_type=self._cpp_dtype())
```
With:
```python
        builder.add_norm(
            self._get(f'{pfx}.norm.weight'), data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
```

- [ ] **Step 8: Update gpt_oss_spec.py**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`:

Replace line 134:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
```
With:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
```

- [ ] **Step 9: Update glm4_moe_lite_spec.py**

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`:

Replace line 170:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
```
With:
```python
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
```

Replace lines 197–201:
```python
        builder.add_norms(
            q_a_norm=self._get(f'{pfx}.q_a_layernorm.weight'),
            kv_a_norm=self._get(f'{pfx}.kv_a_layernorm.weight'),
            data_type=self._cpp_dtype(),
        )
```
With:
```python
        builder.add_norms(
            q_a_norm=self._get(f'{pfx}.q_a_layernorm.weight'),
            kv_a_norm=self._get(f'{pfx}.kv_a_layernorm.weight'),
            data_type=self._cpp_dtype(),
            norm_eps=self._norm_eps,
        )
```

- [ ] **Step 10: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/norm.py \
        lmdeploy/turbomind/deploy/builder/_base.py \
        lmdeploy/turbomind/deploy/builder/attention.py \
        lmdeploy/turbomind/deploy/builder/mla.py \
        lmdeploy/turbomind/deploy/builder/deltanet.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(python): thread norm_eps through builder layer to NormConfig"
```

---

### Task 3: Remove norm_eps from C++ layers, read from NormWeight

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.h` (lines 56, 85)
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc` (lines 104, 644, 655, 694, 698)
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.h` (lines 22, 42)
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc` (lines 12, 20, 315)
- Modify: `src/turbomind/models/llama/unified_decoder.h` (line 40, lines 53–60)
- Modify: `src/turbomind/models/llama/unified_decoder.cc` (lines 46, 56, 69, 97, 111, 127, 197, 243–250, 284–291)

- [ ] **Step 1: Update UnifiedAttentionLayer header**

In `src/turbomind/models/llama/unified_attention_layer.h`:

Remove `float norm_eps` from the constructor (line 56). Replace lines 56–65:

```cpp
    UnifiedAttentionLayer(float                   norm_eps,
                          int                     quant_policy,
                          const std::vector<int>& layer_types,
                          int                     layer_num,
                          const RopeParam&        rope,
                          int                     cache_block_seq_len,
                          const EngineParam&      engine,
                          const Context&          context,
                          int                     phases,
                          bool                    init);
```

With:

```cpp
    UnifiedAttentionLayer(int                     quant_policy,
                          const std::vector<int>& layer_types,
                          int                     layer_num,
                          const RopeParam&        rope,
                          int                     cache_block_seq_len,
                          const EngineParam&      engine,
                          const Context&          context,
                          int                     phases,
                          bool                    init);
```

Remove `const float norm_eps_;` member (line 85).

- [ ] **Step 2: Update UnifiedAttentionLayer implementation**

In `src/turbomind/models/llama/unified_attention_layer.cc`:

Update the constructor definition. Remove `float norm_eps` parameter and the `norm_eps_{norm_eps}` initializer (line 104).

Update all 4 `norm_eps_` references:

Line 644 — replace:
```cpp
        invokeRMSNorm(q_a, q_a, w.q_a_layernorm->weight, norm_eps_, stream);
```
With:
```cpp
        invokeRMSNorm(q_a, q_a, w.q_a_layernorm->weight, w.q_a_layernorm->norm_eps_, stream);
```

Line 655 — replace:
```cpp
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm->weight, norm_eps_, stream);
```
With:
```cpp
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm->weight, w.kv_a_layernorm->norm_eps_, stream);
```

Line 694 — replace:
```cpp
    invokeRMSNormQK(q, weights.q_norm->weight, norm_eps_, stream);
```
With:
```cpp
    invokeRMSNormQK(q, weights.q_norm->weight, weights.q_norm->norm_eps_, stream);
```

Line 698 — replace:
```cpp
    invokeRMSNormQK(k, weights.k_norm->weight, norm_eps_, aux_stream_);
```
With:
```cpp
    invokeRMSNormQK(k, weights.k_norm->weight, weights.k_norm->norm_eps_, aux_stream_);
```

- [ ] **Step 3: Update GatedDeltaNetLayer header**

In `src/turbomind/models/llama/GatedDeltaNetLayer.h`:

Remove `float norm_eps` from the constructor (line 22). Replace lines 22–27:

```cpp
    GatedDeltaNetLayer(float                   norm_eps,
                       DataType                state_dtype,
                       const std::vector<int>& layer_types,
                       const EngineParam&      engine,
                       const Context&          ctx,
                       int                     phases);
```

With:

```cpp
    GatedDeltaNetLayer(DataType                state_dtype,
                       const std::vector<int>& layer_types,
                       const EngineParam&      engine,
                       const Context&          ctx,
                       int                     phases);
```

Remove `float norm_eps_;` member (line 42).

- [ ] **Step 4: Update GatedDeltaNetLayer implementation**

In `src/turbomind/models/llama/GatedDeltaNetLayer.cc`:

Remove `float norm_eps` from the constructor definition (line 12). Remove `norm_eps_(norm_eps),` (line 20).

Line 315 — replace:
```cpp
        invokeRMSNormGated(hidden_view, gate, weights.norm->weight, norm_eps_, stream);
```
With:
```cpp
        invokeRMSNormGated(hidden_view, gate, weights.norm->weight, weights.norm->norm_eps_, stream);
```

- [ ] **Step 5: Update UnifiedDecoder header**

In `src/turbomind/models/llama/unified_decoder.h`:

Remove `const float rmsnorm_eps_;` (line 40).

Add `float eps` parameter to `AllreduceResidualRMSnorm`. Replace lines 53–60:

```cpp
    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums);
```

With:

```cpp
    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  float         eps,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums);
```

- [ ] **Step 6: Update UnifiedDecoder implementation**

In `src/turbomind/models/llama/unified_decoder.cc`:

Remove `rmsnorm_eps_(model.norm_eps),` from the constructor initializer list (line 46).

Update constructor call sites — remove `model.norm_eps` from both:

Replace line 56:
```cpp
        model.norm_eps,
```
(Remove this line entirely — `model.quant_policy` becomes the first arg.)

Replace lines 69:
```cpp
            model.norm_eps, model.linear_state_dtype, model.layer_types,
```
With:
```cpp
            model.linear_state_dtype, model.layer_types,
```

Update `AllreduceResidualRMSnorm` definition — add `float eps` parameter after `const Tensor& weight`. Replace all 3 internal uses of `rmsnorm_eps_` with `eps`:

Line 97 — replace `rmsnorm_eps_,` with `eps,`
Line 111 — replace `rmsnorm_eps_,` with `eps,`
Line 127 — replace `rmsnorm_eps_,` with `eps,`

Update all call sites of `AllreduceResidualRMSnorm` to pass eps from the NormWeight:

Line 197 (direct invokeRMSNorm, not AllreduceResidualRMSnorm) — replace:
```cpp
    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->attention_norm->weight, rmsnorm_eps_, stream);
```
With:
```cpp
    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->attention_norm->weight, weights.at(0)->attention_norm->norm_eps_, stream);
```

Lines 243–250 — add eps parameter:
```cpp
        AllreduceResidualRMSnorm(global_hidden_states,
                                 local_residual,
                                 out_bias,
                                 weights.at(layer)->ffn_norm->weight,
                                 weights.at(layer)->ffn_norm->norm_eps_,
                                 local_token_num,
                                 attn_tp_group_,
                                 0,
                                 local_token_nums.data());
```

Lines 284–291 — add eps parameter. Use `weights.at(layer)->ffn_norm->norm_eps_` (same value for all norms, and this NormWeight is always available):
```cpp
        AllreduceResidualRMSnorm(global_hidden_states,
                                 local_residual,
                                 {},
                                 scale_weight,
                                 weights.at(layer)->ffn_norm->norm_eps_,
                                 local_token_num,
                                 0,
                                 attn_tp_group_,
                                 local_token_nums.data());
```

- [ ] **Step 7: Build to verify clean compile**

Run: `cd build && ninja`
Expected: clean build with no warnings

- [ ] **Step 8: Commit**

```bash
git add src/turbomind/models/llama/unified_attention_layer.h \
        src/turbomind/models/llama/unified_attention_layer.cc \
        src/turbomind/models/llama/GatedDeltaNetLayer.h \
        src/turbomind/models/llama/GatedDeltaNetLayer.cc \
        src/turbomind/models/llama/unified_decoder.h \
        src/turbomind/models/llama/unified_decoder.cc
git commit -m "refactor(decoder): remove norm_eps from layer constructors, read from NormWeight"
```

---

### Task 4: Smoke test with a model

**Files:**
- None (verification only)

- [ ] **Step 1: Check GPU availability**

Run the `get_gpu_usage` MCP tool to confirm a free GPU.

- [ ] **Step 2: Test with a non-MoE model**

Run:
```bash
cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py Qwen/Qwen3-4B --prompt "Hello, how are you?" --request-output-len 128
```

Expected: Model produces meaningful human-language response, no CUDA errors, no assertion failures.

- [ ] **Step 3: Test with a MoE model**

Run:
```bash
cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py Qwen/Qwen3.5-35B-A3B-AWQ --prompt "Hello, how are you?" --request-output-len 128
```

Expected: Same — meaningful response, no errors.

- [ ] **Step 4: Commit (if any fixup was needed)**

Only commit if fixes were required during testing.
