# Uniform WeightSpec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `_infer_cpp_linear_dtype` handle all dense dtypes (including FP32) so `set_weight_spec` is always called unconditionally.

**Architecture:** Replace hard-coded BF16/FP16 checks in `_infer_cpp_linear_dtype` with a `_TORCH_TO_CPP.get()` lookup, then remove the `if cpp_dtype is not None:` guard before `set_weight_spec`.

**Tech Stack:** Python

---

### Task 1: Simplify `_infer_cpp_linear_dtype` and remove the None guard

**Files:**
- Modify: `lmdeploy/turbomind/deploy/commit.py:65-80` (`_infer_cpp_linear_dtype`)
- Modify: `lmdeploy/turbomind/deploy/commit.py:267-270` (None guard before `set_weight_spec`)

- [ ] **Step 1: Replace the hard-coded dtype checks with `_TORCH_TO_CPP` lookup**

In `lmdeploy/turbomind/deploy/commit.py`, replace lines 73-80:

```python
    # Dense (or missing format): dtype from weight tensor
    weight = linear.tensors.get("weight")
    if weight is not None:
        if weight.dtype == torch.bfloat16:
            return _tm.DataType.TYPE_BF16, 0
        if weight.dtype == torch.float16:
            return _tm.DataType.TYPE_FP16, 0
    return None, 0
```

with:

```python
    # Dense (or missing format): dtype from weight tensor
    weight = linear.tensors.get("weight")
    if weight is not None:
        return _TORCH_TO_CPP.get(weight.dtype), 0
    return None, 0
```

- [ ] **Step 2: Remove the None guard before `set_weight_spec`**

In `lmdeploy/turbomind/deploy/commit.py`, replace lines 267-270:

```python
    # Set weight spec for quantization metadata (skip for dense weights
    # with unrecognized dtype, e.g. FP32 gate weights in quantized models).
    if cpp_dtype is not None:
        linear_mod.set_weight_spec(cpp_dtype, group_size)
```

with:

```python
    linear_mod.set_weight_spec(cpp_dtype, group_size)
```

- [ ] **Step 3: Verify imports resolve**

```bash
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.commit import commit_linear; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
cd /data/lmdeploy-modeling && git add lmdeploy/turbomind/deploy/commit.py && git commit -m "refactor(commit): use _TORCH_TO_CPP lookup for dense linear dtype inference

Replace hard-coded BF16/FP16 checks in _infer_cpp_linear_dtype with
_TORCH_TO_CPP.get(), which handles all dense dtypes including FP32.
Remove the None guard before set_weight_spec since cpp_dtype is now
always non-None for linears with weight tensors."
```

---

### Task 2: Test all models with TP=1 and TP=2

**Files:** None (testing only)

- [ ] **Step 1: Build**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 2: Test all 13 models with TP=1 and TP=2**

Use the turbomind-tester agent to test all available models with both TP=1 and TP=2. Each test should use a prompt requiring at least 128 tokens. Verify responses contain meaningful human words.

Available models:
1. /data/model/Qwen3-30B-A3B-GPTQ-Int4
2. QuantTrio/GLM-4.7-Flash-AWQ
3. QuantTrio/Qwen3.5-35B-A3B-AWQ
4. Qwen/Qwen3-30B-A3B
5. Qwen/Qwen3-30B-A3B-FP8
6. Qwen/Qwen3-4B
7. Qwen/Qwen3-4B-AWQ
8. Qwen/Qwen3.5-27B
9. Qwen/Qwen3.5-35B-A3B
10. Qwen/Qwen3.5-35B-A3B-FP8
11. openai/gpt-oss-20b
12. unsloth/gpt-oss-20b-BF16
13. zai-org/GLM-4.7-Flash

Use CUDA_VISIBLE_DEVICES=2 for TP=1, CUDA_VISIBLE_DEVICES=2,3 for TP=2. Run tests sequentially.

- [ ] **Step 3: Fix any failures**

If any model fails, diagnose and fix before proceeding.
