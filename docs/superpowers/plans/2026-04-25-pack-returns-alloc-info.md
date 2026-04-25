# Pack Returns Alloc Info Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `alloc_shape`/`alloc_dtype` metadata into `WeightFormat.pack()` so the builder no longer reconstructs format internals.

**Architecture:** `pack()` returns a `PackedTensor` NamedTuple bundling the packed tensor with optional alloc overrides. The builder reads these instead of branching on `is_quantized` and `weight_cpp_dtype`. The only branch left in the builder is the `model_dtype` deployment override.

**Tech Stack:** Python, torch, TurboMind C++ extension (`_tm`)

---

### Task 1: Add `PackedTensor` NamedTuple

**Files:**
- Modify: `lmdeploy/turbomind/deploy/weight_format.py` (after imports, before `pack_u4_row`)

- [ ] **Step 1: Add `PackedTensor`**

```python
from typing import NamedTuple

class PackedTensor(NamedTuple):
    tensor:      torch.Tensor
    alloc_shape: list[int] | None       # None = inherit from packed tensor
    alloc_dtype: "_tm.DataType | None"  # None = inherit from packed tensor
```

Place it after the existing imports block and `_tm` check, before `pack_u4_row` (line 58).

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/weight_format.py
git commit -m "feat: add PackedTensor NamedTuple for alloc metadata"
```

---

### Task 2: Update base `WeightFormat.pack()` return type

**Files:**
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:136-137`

- [ ] **Step 1: Change return type**

Replace lines 136-137:
```python
    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        return tensor
```

With:
```python
    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        return PackedTensor(tensor, None, None)
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/weight_format.py
git commit -m "feat: base WeightFormat.pack() returns PackedTensor"
```

---

### Task 3: Update quantized `pack()` overrides

**Files:**
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:226-229` (AWQFormat)
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:280-283` (GPTQFormat)
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:319-322` (CompressedTensorFormat)
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:395-398` (MXFP4Format)
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:352` (FP8Format — add new override)

These 5 formats all have `weight_dtype` set. Four already override `pack()` with the same pattern. FP8Format needs a new override.

- [ ] **Step 1: Update AWQFormat.pack (line 226)**

Replace:
```python
    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor
```

With:
```python
    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)
```

- [ ] **Step 2: Update GPTQFormat.pack (line 280)**

Replace:
```python
    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor
```

With:
```python
    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)
```

- [ ] **Step 3: Update CompressedTensorFormat.pack (line 319)**

Replace:
```python
    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor
```

With:
```python
    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)
```

- [ ] **Step 4: Update MXFP4Format.pack (line 395)**

Replace:
```python
    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor
```

With:
```python
    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)
```

- [ ] **Step 5: Add FP8Format.pack override (after line 368)**

FP8Format currently inherits identity `pack()` but has `weight_dtype = TYPE_FP8_E4M3`. Its normalize converts float8 to uint8 (view), and pack should remain identity while surfacing the alloc_dtype override. Insert after line 368 (end of `dequant`), before the blank line that precedes MXFP4Format:

```python
    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight":
            return PackedTensor(tensor, list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)
```

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/weight_format.py
git commit -m "feat: quantized pack() overrides return PackedTensor with alloc metadata"
```

---

### Task 4: Simplify builder loop in `_commit_linear`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:464-503`

- [ ] **Step 1: Replace the tensor packing line and the GPU loop**

Replace lines 464-503:
```python
        tensors = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
        is_quantized = linear.data_format.is_quantized()

        kind_split_dims = {
            kind: None if (kind == 'bias' and split_side == SplitSide.INPUT)
                  else split_dim
            for kind in tensors
        }

        if tp > 1 and split_dim is not None:
            for kind, tensor in tensors.items():
                kind_split_dim = kind_split_dims[kind]
                if kind_split_dim is not None:
                    d = tensor.shape[kind_split_dim]
                    assert d % tp == 0, (
                        f"TP split: {name}.{kind} dim {kind_split_dim} "
                        f"has size {d}, not divisible by tp={tp}.")

        # --- Per-GPU: standalone creation + tensor copy --------------------
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                rank = self._rank_for(i) if tp > 1 else 0

                mod = _tm.create_module(lin_cfg)

                for kind, tensor in tensors.items():
                    shard = _shard(tensor, kind_split_dims[kind], tp, rank)

                    if kind == 'weight' and is_quantized:
                        alloc_shape, alloc_dtype = ([in_dim, out_dim],
                                                    weight_cpp_dtype)
                    elif kind == 'weight' and model_dtype is not None:
                        alloc_shape, alloc_dtype = None, model_dtype
                    else:
                        alloc_shape, alloc_dtype = None, None

                    _copy_shard_to_param(mod, kind, shard,
                                         alloc_shape=alloc_shape,
                                         alloc_dtype=alloc_dtype)

                handles.append(mod)
```

With:
```python
        packed = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
        tensors = {k: p.tensor for k, p in packed.items()}

        kind_split_dims = {
            kind: None if (kind == 'bias' and split_side == SplitSide.INPUT)
                  else split_dim
            for kind in tensors
        }

        if tp > 1 and split_dim is not None:
            for kind, tensor in tensors.items():
                kind_split_dim = kind_split_dims[kind]
                if kind_split_dim is not None:
                    d = tensor.shape[kind_split_dim]
                    assert d % tp == 0, (
                        f"TP split: {name}.{kind} dim {kind_split_dim} "
                        f"has size {d}, not divisible by tp={tp}.")

        # --- Per-GPU: standalone creation + tensor copy --------------------
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                rank = self._rank_for(i) if tp > 1 else 0

                mod = _tm.create_module(lin_cfg)

                for kind, tensor in tensors.items():
                    shard = _shard(tensor, kind_split_dims[kind], tp, rank)

                    alloc_shape, alloc_dtype = packed[kind].alloc_shape, \
                                               packed[kind].alloc_dtype
                    if alloc_dtype is None and kind == 'weight' \
                            and model_dtype is not None:
                        alloc_dtype = model_dtype

                    _copy_shard_to_param(mod, kind, shard,
                                         alloc_shape=alloc_shape,
                                         alloc_dtype=alloc_dtype)

                handles.append(mod)
```

- [ ] **Step 2: Remove unused `weight_cpp_dtype`**

Remove line 443:
```python
        weight_cpp_dtype = linear.data_format.dtype
```

This local was only used in the old branching; no longer needed.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "refactor: builder reads alloc metadata from pack() instead of reconstructing it"
```

---

### Task 5: Verify correctness with a model test

**Files:** (none modified)

- [ ] **Step 1: Check GPU availability**

Use the `mcp__gpu-monitor__get_gpu_usage` tool to verify at least one GPU is free (low memory usage, no other processes). If a GPU is occupied by another process, wait or free it before proceeding.

- [ ] **Step 2: List available models**

Use the `mcp__model-server__list_models` tool to see which models are cached locally. Pick one unquantized model (e.g., a 7B/8B dense model) for the first test, and one quantized model (AWQ, GPTQ) if available.

- [ ] **Step 3: Test an unquantized model**

```bash
python scripts/test_turbomind_model.py <unquantized_model_id>
```

Verify the model responds with meaningful human words for at least 128 tokens.

- [ ] **Step 4: Test a quantized model**

```bash
python scripts/test_turbomind_model.py <quantized_model_id>
```

Verify the model responds with meaningful human words for at least 128 tokens.

- [ ] **Step 5: Commit** (only if any fixes were needed)

```bash
git commit -m "fix: ..."
```
