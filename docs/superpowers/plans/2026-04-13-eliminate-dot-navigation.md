# Eliminate Dot Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dot-navigation loops in `text_model_loader.py` with explicit norm-children and direct-params methods on specs.

**Architecture:** Split each spec's `attn_params()` / `linear_attn_params()` so that norm submodule weights (always broadcast, single `"weight"` param) are returned separately from direct params (arbitrary split_side). The loader then iterates each category with the correct API call — no string splitting.

**Tech Stack:** Python, torch, lmdeploy deploy module.

---

### Task 1: Add base class defaults for new methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:249-291`

- [ ] **Step 1: Add `attn_norm_children` and `linear_attn_norm_children` defaults**

In `spec.py`, after the existing `attn_params` method (around line 259), add:

```python
def attn_norm_children(self, layer: int) -> dict[str, torch.Tensor]:
    """Return norm submodule weights for the attention block.

    Each entry is ``{child_name: tensor}``.  The loader creates a
    ``NormConfig`` child and commits the tensor as ``"weight"``.
    All norm children are broadcast (no TP split).
    """
    return {}
```

After the existing `linear_attn_params` method (around line 291), add:

```python
def linear_attn_norm_children(self, layer: int) -> dict[str, torch.Tensor]:
    """Return norm submodule weights for the linear-attention block.

    Each entry is ``{child_name: tensor}``.  The loader creates a
    ``NormConfig`` child and commits the tensor as ``"weight"``.
    All norm children are broadcast (no TP split).
    """
    return {}
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(spec): add attn_norm_children and linear_attn_norm_children base defaults"
```

---

### Task 2: Split qwen3_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:71-81`

- [ ] **Step 1: Replace `attn_params` and add `attn_norm_children`**

Replace the existing `attn_params` method (lines 71-81) with:

```python
def attn_params(self, layer):
    return {}

def attn_norm_children(self, layer):
    params = {}
    q = self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight")
    k = self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight")
    if q is not None and k is not None:
        q, k = self._permute_qk_tensors(q, k)
    if q is not None:
        params["q_norm"] = q
    if k is not None:
        params["k_norm"] = k
    return params
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "refactor(qwen3): split attn_params into attn_norm_children"
```

---

### Task 3: Split qwen3_5_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:213-226,246-277`

- [ ] **Step 1: Replace `attn_params` and add `attn_norm_children`**

Replace the existing `attn_params` method (lines 213-226) with:

```python
def attn_params(self, layer):
    return {}

def attn_norm_children(self, layer):
    params = {}
    if not self._is_linear_attn(layer):
        q = self._zero_centered(
            self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight"))
        k = self._zero_centered(
            self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight"))
        if q is not None and k is not None:
            q, k = self._permute_qk_tensors(q, k)
        if q is not None:
            params["q_norm"] = q
        if k is not None:
            params["k_norm"] = k
    return params
```

- [ ] **Step 2: Replace `linear_attn_params` and add `linear_attn_norm_children`**

Replace the existing `linear_attn_params` method (lines 246-277) with:

```python
def linear_attn_params(self, layer):
    params = {}
    if not self._is_linear_attn(layer):
        return params
    pfx = f"{self._layer_prefix}.{layer}.linear_attn"
    for key in ["A_log", "dt_bias"]:
        t = self._get(f"{pfx}.{key}")
        if t is not None:
            params[key] = (t, SplitSide.OUTPUT)
    conv1d = self._get(f"{pfx}.conv1d.weight")
    if conv1d is not None and conv1d.ndim == 3 and conv1d.shape[1] == 1:
        conv1d = conv1d.squeeze(1)
    # C++ kernel expects [d_conv, conv_dim]; HF stores [conv_dim, d_conv].
    if conv1d is not None:
        conv1d = conv1d.t().contiguous()
        if self._attn_tp > 1 and self._linear_qkv_split is not None:
            q_dim, k_dim, v_dim = self._linear_qkv_split
            d_conv = conv1d.shape[0]
            tp = self._attn_tp
            q_part = conv1d[:, :q_dim]
            k_part = conv1d[:, q_dim:q_dim + k_dim]
            v_part = conv1d[:, q_dim + k_dim:]
            conv1d = torch.cat([
                q_part.reshape(d_conv, tp, q_dim // tp),
                k_part.reshape(d_conv, tp, k_dim // tp),
                v_part.reshape(d_conv, tp, v_dim // tp),
            ], dim=2).reshape(d_conv, -1).contiguous()
        params["conv1d"] = (conv1d, SplitSide.OUTPUT)
    return params

def linear_attn_norm_children(self, layer):
    params = {}
    if not self._is_linear_attn(layer):
        return params
    pfx = f"{self._layer_prefix}.{layer}.linear_attn"
    norm = self._get(f"{pfx}.norm.weight")
    if norm is not None:
        params["norm"] = norm
    return params
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(qwen3.5): split attn_params and linear_attn_params into norm children"
```

---

### Task 4: Split glm4_moe_lite_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:182-192`

- [ ] **Step 1: Replace `attn_params` and add `attn_norm_children`**

Replace the existing `attn_params` method (lines 182-192) with:

```python
def attn_params(self, layer):
    return {}

def attn_norm_children(self, layer):
    params = {}
    q_a = self._get(
        f"{self._layer_prefix}.{layer}.self_attn.q_a_layernorm.weight")
    kv_a = self._get(
        f"{self._layer_prefix}.{layer}.self_attn.kv_a_layernorm.weight")
    if q_a is not None:
        params["q_a_layernorm"] = q_a
    if kv_a is not None:
        params["kv_a_layernorm"] = kv_a
    return params
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(glm4_moe_lite): split attn_params into attn_norm_children"
```

---

### Task 5: Update gpt_oss_spec.py (no structural change needed)

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:125-131`

`gpt_oss_spec.py` already returns only direct params (`"sinks"` with no dots). No structural change is needed. The existing `attn_params` returns `{"sinks": (tensor, SplitSide.OUTPUT)}` — no dotted names, no norm children.

- [ ] **Step 1: Verify no changes needed**

Confirm the current `attn_params` returns only `"sinks"` (single-segment, no dots). If it does, skip this file.

---

### Task 6: Replace dot-navigation blocks in text_model_loader.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:110-118,256-264`

- [ ] **Step 1: Replace the attention dot-navigation block (lines 110-118)**

Replace:

```python
        # --- Parameters (q_norm, k_norm, sinks, etc.) ---
        for name, (tensor, split_side) in spec.attn_params(layer).items():
            parts = name.split('.')
            parent = attn
            for seg in parts[:-1]:
                parent = parent.create_child(seg, NormConfig(
                    dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                    data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

With:

```python
        # --- Direct params (sinks, etc.) ---
        for name, (tensor, split_side) in spec.attn_params(layer).items():
            attn.commit_tensor(name, tensor, split_side=split_side)

        # --- Norm children (q_norm, k_norm, etc.) ---
        for name, tensor in spec.attn_norm_children(layer).items():
            child = attn.create_child(name, NormConfig(
                dim=tensor.shape[-1],
                data_type=dtype))
            child.commit_tensor('weight', tensor)
```

- [ ] **Step 2: Replace the linear-attention dot-navigation block (lines 256-264)**

Replace:

```python
        # --- Parameters (A_log, dt_bias, conv1d, norm.weight) ---
        for name, (tensor, split_side) in spec.linear_attn_params(layer).items():
            parts = name.split('.')
            parent = linear_attn
            for seg in parts[:-1]:
                parent = parent.create_child(seg, NormConfig(
                    dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                    data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

With:

```python
        # --- Direct params (A_log, dt_bias, conv1d, etc.) ---
        for name, (tensor, split_side) in spec.linear_attn_params(layer).items():
            linear_attn.commit_tensor(name, tensor, split_side=split_side)

        # --- Norm children (norm, etc.) ---
        for name, tensor in spec.linear_attn_norm_children(layer).items():
            child = linear_attn.create_child(name, NormConfig(
                dim=tensor.shape[-1],
                data_type=dtype))
            child.commit_tensor('weight', tensor)
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): replace dot-navigation with explicit norm-children loops"
```

---

### Task 7: Test affected models

**Files:** None (verification only)

- [ ] **Step 1: Verify qwen3**

Use the turbomind-tester agent to test `Qwen/Qwen3-0.6B` with TP=1. Prompt: "What is the capital of France? Please explain in detail." Minimum 128 tokens. Verify meaningful response.

- [ ] **Step 2: Verify qwen3.5 (has both attn and linear_attn norm children)**

Use the turbomind-tester agent to test `Qwen/Qwen3.5-MoE` with TP=1. Same prompt, verify meaningful response.

- [ ] **Step 3: Verify glm4_moe_lite**

Use the turbomind-tester agent to test `THUDM/GLM-4-9B-0414` with TP=1. Same prompt, verify meaningful response.

- [ ] **Step 4: Verify gpt_oss (has direct attn params, no norm children)**

Use the turbomind-tester agent to test `openbmb/MiniCPM4-8B` with TP=1. Same prompt, verify meaningful response.
