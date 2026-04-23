# Module API Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `create_param()` to the C++ Module API and clean up `create_child()` calling conventions across Python callers.

**Architecture:** Add a new `create_param()` method to C++ Module that allocates and registers a Parameter tensor in one call, exposed to Python via pybind. Then migrate Python callers from the `create_child()` + `get()` two-step pattern to using `create_child()`'s return value directly.

**Tech Stack:** C++ (Module class, pybind11), Python (model loading pipeline)

---

### Task 1: Add `create_param()` to C++ Module

**Files:**
- Modify: `src/turbomind/core/module.h` (add declaration after `alloc()` on line 74)
- Modify: `src/turbomind/core/module.cc` (add implementation after `alloc()` on line 93)

- [ ] **Step 1: Add declaration to module.h**

In `src/turbomind/core/module.h`, add the following declaration after the existing `alloc()` declaration (after line 74):

```cpp
    /// Create and register a named parameter tensor with the given shape/dtype.
    /// Returns the allocated Tensor for the caller to fill via copy_from().
    Tensor create_param(const std::string& name,
                        const std::vector<size_t>& shape,
                        DataType dtype,
                        int group_size = 0);
```

- [ ] **Step 2: Add implementation to module.cc**

In `src/turbomind/core/module.cc`, add the following implementation after the existing `alloc()` method (after line 93):

```cpp
Tensor Module::create_param(const std::string& name,
                            const std::vector<size_t>& shape,
                            DataType dtype,
                            int group_size)
{
    auto layout = Layout{std::vector<ssize_t>(shape.begin(), shape.end())};
    auto tensor = Tensor{std::move(layout), dtype, kDEVICE};
    add_param(name, tensor);
    return tensor;
}
```

Note: Uses `kDEVICE` (same as LinearWeight's `do_allocate`) which resolves to the current CUDA device via `Context::alloc()`. The pybind `with_context` wrapper sets up the correct CUDA stream and allocator before this call. `add_param()` registers the tensor as a named parameter (stores a `Tensor*` pointer). The `group_size` parameter is accepted for API consistency with `alloc()` but unused by the base Module implementation.

- [ ] **Step 3: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja _turbomind`
Expected: Clean build with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "feat(core): add Module::create_param() for one-call parameter creation"
```

---

### Task 2: Add pybind binding for `create_param()`

**Files:**
- Modify: `src/turbomind/python/bind.cpp` (add binding after the `alloc` binding around line 562)

- [ ] **Step 1: Add pybind binding**

In `src/turbomind/python/bind.cpp`, add the following binding after the existing `alloc` binding block (after the line containing `"group_size"_a = 0` at line 562):

```cpp
        .def("create_param",
             [with_context](ft::core::Module& m,
                            const std::string& name,
                            std::vector<size_t> shape,
                            ft::DataType dtype,
                            int group_size) {
                 return with_context(m, [&] {
                     return std::make_shared<Tensor>(
                         m.create_param(name, shape, dtype, group_size));
                 });
             },
             "name"_a,
             "shape"_a,
             "dtype"_a,
             "group_size"_a = 0)
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja _turbomind`
Expected: Clean build with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "feat(bind): expose Module::create_param() to Python"
```

---

### Task 3: Clean up `create_child()` + `get()` in text_model_loader.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

This task removes the separate `get()` calls after `create_child()`, using the return value directly.

- [ ] **Step 1: Fix layer norms (lines 100-105)**

Change from:
```python
            handle.create_child('attention_norm', 'NormWeight', norm_cfg)
            handle.create_child('ffn_norm', 'NormWeight', norm_cfg)
            commit_tensor(handle.get('attention_norm'),
                                spec.attn_norm(layer), 'weight')
            commit_tensor(handle.get('ffn_norm'),
                                spec.ffn_norm(layer), 'weight')
```

To:
```python
            attention_norm = handle.create_child('attention_norm', 'NormWeight', norm_cfg)
            ffn_norm = handle.create_child('ffn_norm', 'NormWeight', norm_cfg)
            commit_tensor(attention_norm, spec.attn_norm(layer), 'weight')
            commit_tensor(ffn_norm, spec.ffn_norm(layer), 'weight')
```

- [ ] **Step 2: Fix attention module (lines 115-133)**

Change from:
```python
                handle.create_child('attention', 'AttentionWeight', {
                    ...
                })
                attn_mod = handle.get('attention')
```

To:
```python
                attn_mod = handle.create_child('attention', 'AttentionWeight', {
                    ...
                })
```

(Delete the separate `attn_mod = handle.get('attention')` line and assign the return value directly.)

- [ ] **Step 3: Fix feed_forward module (lines 149-159)**

Change from:
```python
                handle.create_child('feed_forward', 'FfnWeight', {
                    ...
                })
                ffn_mod = handle.get('feed_forward')
```

To:
```python
                ffn_mod = handle.create_child('feed_forward', 'FfnWeight', {
                    ...
                })
```

- [ ] **Step 4: Fix moe_ffn module (lines 183-206)**

Change from:
```python
                handle.create_child('moe_ffn', 'MoeWeight', {
                    ...
                })
                moe_mod = handle.get('moe_ffn')
```

To:
```python
                moe_mod = handle.create_child('moe_ffn', 'MoeWeight', {
                    ...
                })
```

- [ ] **Step 5: Fix gate LinearWeight (lines 216-221)**

The `gate` creation on line 216-221 does not use a separate `get()`, but the return value is also unused. No change needed — `commit_linear` is called on `moe_mod` which will use `module.child(name)` internally.

No change for this one.

- [ ] **Step 6: Fix shared_gate LinearWeight (lines 229-234)**

Same as gate — no separate `get()`, no change needed.

No change for this one.

- [ ] **Step 7: Fix linear_attn module (lines 275-287)**

Change from:
```python
                handle.create_child('linear_attn', 'DeltaNetWeight', {
                    ...
                })
                linear_attn_mod = handle.get('linear_attn')
```

To:
```python
                linear_attn_mod = handle.create_child('linear_attn', 'DeltaNetWeight', {
                    ...
                })
```

- [ ] **Step 8: Fix tok_embeddings (lines 351-360)**

Change from:
```python
                root.create_child('tok_embeddings', 'LinearWeight', {
                    'input_dim': padded_vocab,
                    'output_dim': hidden // tp,
                    'data_type': dtype,
                    'has_bias': False,
                })
                commit_tensor(root.get('tok_embeddings'), emb_padded,
                                     'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)
```

To:
```python
                tok_emb = root.create_child('tok_embeddings', 'LinearWeight', {
                    'input_dim': padded_vocab,
                    'output_dim': hidden // tp,
                    'data_type': dtype,
                    'has_bias': False,
                })
                commit_tensor(tok_emb, emb_padded,
                                     'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)
```

- [ ] **Step 9: Fix norm (lines 365-367)**

Change from:
```python
                root.create_child('norm', 'NormWeight',
                                  {'dim': hidden, 'data_type': dtype})
                commit_tensor(root.get('norm'), norm, 'weight')
```

To:
```python
                norm_mod = root.create_child('norm', 'NormWeight',
                                  {'dim': hidden, 'data_type': dtype})
                commit_tensor(norm_mod, norm, 'weight')
```

- [ ] **Step 10: Fix output (lines 374-382)**

Change from:
```python
                root.create_child('output', 'LinearWeight', {
                    'input_dim': hidden,
                    'output_dim': padded_vocab // tp,
                    'data_type': dtype,
                    'has_bias': False,
                })
                commit_tensor(root.get('output'), output_t, 'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)
```

To:
```python
                output_mod = root.create_child('output', 'LinearWeight', {
                    'input_dim': hidden,
                    'output_dim': padded_vocab // tp,
                    'data_type': dtype,
                    'has_bias': False,
                })
                commit_tensor(output_mod, output_t, 'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)
```

- [ ] **Step 11: Fix expert module access (line 255)**

Change from:
```python
                    expert_mod = moe_mod.get('experts').get(str(e))
```

To:
```python
                    expert_mod = moe_mod.child('experts').child(str(e))
```

This doesn't use `create_child`, but `.get()` aborts on failure while `.child()` returns `None`. Since the expert was just created in the loop above, this is safe and avoids the abort-on-missing behavior of `get()`.

- [ ] **Step 12: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): use create_child() return value directly"
```

---

### Task 4: Clean up `create_child()` + `get()` in module.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module.py` (lines 932-938)

- [ ] **Step 1: Fix commit_linear function**

In `lmdeploy/turbomind/deploy/module.py`, in the `commit_linear` function, change lines 932-938 from:

```python
        module.create_child(name, 'LinearWeight', {
            'input_dim': in_dim,
            'output_dim': out_dim,
            'data_type': compute_dtype.value if compute_dtype else 0,
            'has_bias': 1 if 'bias' in linear.tensors else 0,
        })
        linear_mod = module.get(name)
```

To:

```python
        linear_mod = module.create_child(name, 'LinearWeight', {
            'input_dim': in_dim,
            'output_dim': out_dim,
            'data_type': compute_dtype.value if compute_dtype else 0,
            'has_bias': 1 if 'bias' in linear.tensors else 0,
        })
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/module.py
git commit -m "refactor(module): use create_child() return value in commit_linear"
```

---

### Task 5: Verify with a model test

**Files:** None (testing only)

- [ ] **Step 1: Build the project**

Run: `cd /data/lmdeploy-modeling/build && ninja _turbomind`
Expected: Clean build.

- [ ] **Step 2: Check GPU availability**

Use the `get_gpu_usage` MCP tool to verify an empty GPU is available.

- [ ] **Step 3: Run model test**

Use the turbomind-tester agent to test a model (e.g., Qwen2.5-7B or any available model) with a 128+ token generation request. Verify the response contains meaningful words, not gibberish.

- [ ] **Step 4: Commit any fixes if needed**

If the test reveals issues, fix and commit. Re-run until passing.
