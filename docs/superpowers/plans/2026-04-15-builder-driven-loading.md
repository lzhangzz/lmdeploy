# Builder-Driven Model Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TextModelLoader's hardcoded `_process_*` methods with a Builder pattern where specs drive loading and builders own weight transforms + TP sharding + C++ commit.

**Architecture:** Builder base class absorbs Distributor + commit logic. Specs create builders directly and call methods like `add_qkv_proj()`. All weights loaded upfront via safetensors mmap, enabling a single `model()` call per spec.

**Tech Stack:** Python (pybind11), C++ (x-macro modules), TurboMind engine

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `lmdeploy/turbomind/deploy/builder.py` | Builder base + all concrete builders + commit internals |
| Modify | `src/turbomind/core/module.h` | Add `static create()` factory |
| Modify | `src/turbomind/core/module.cc` | Implement `create()` |
| Modify | `src/turbomind/python/bind.cpp` | Bind `create_module`, `add_child` |
| Modify | `lmdeploy/turbomind/deploy/loader.py` | All-upfront loading |
| Modify | `lmdeploy/turbomind/deploy/source_model/base.py` | Single-batch `readers()` |
| Modify | `lmdeploy/turbomind/deploy/target_model/base.py` | Single-call `export()` |
| Modify | `lmdeploy/turbomind/deploy/spec.py` | Add `model()`, update base class |
| Modify | `lmdeploy/turbomind/deploy/text_model_loader.py` | Simplify to ~20 lines |
| Delete | `lmdeploy/turbomind/deploy/distributor.py` | Absorbed into builder.py |
| Delete | `lmdeploy/turbomind/deploy/transforms.py` | Absorbed into builder.py |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Implement model() |
| Modify | `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Implement model() |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Implement model() |
| Modify | `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Implement model() |

---

### Task 1: C++ — Add `create_module` factory and expose `add_child`

**Files:**
- Modify: `src/turbomind/core/module.h:220-222`
- Modify: `src/turbomind/core/module.cc` (wherever `create_child` is implemented)
- Modify: `src/turbomind/python/bind.cpp:599-637`

- [ ] **Step 1: Add `Module::create()` static factory to header**

In `src/turbomind/core/module.h`, add after the `create_child` declaration (line ~222):

```cpp
    /// Create a standalone module using the type registry (no parent binding).
    /// Uses config.module_type to look up the factory.
    /// Returns owned module, or nullptr on failure.
    static std::unique_ptr<Module> create(const ModuleConfig& config);
```

- [ ] **Step 2: Implement `Module::create()` in module.cc**

Find the `Module::create_child` implementation. Extract the registry creation part:

```cpp
std::unique_ptr<Module> Module::create(const ModuleConfig& config) {
    return ModuleRegistry::instance().create(std::string(config.module_type), config);
}
```

The existing `create_child` should now call this:

```cpp
Module* Module::create_child(const std::string& name, const ModuleConfig& config) {
    auto mod = create(config);
    if (!mod) return nullptr;
    return add_child(name, std::move(mod));
}
```

- [ ] **Step 3: Bind `create_module` and `add_child` in bind.cpp**

In `src/turbomind/python/bind.cpp`, in the `Module` class binding block (after line 625, the `create_child` binding), add:

```cpp
        // Standalone module creation (no parent binding)
        m.def("create_module",
            [](turbomind::core::ModuleConfig& config) -> ft::core::Module* {
                auto mod = ft::core::Module::create(config);
                if (!mod) return nullptr;
                // Return raw pointer; Python will need to manage lifetime.
                // Store in a shared_ptr to prevent GC.
                return mod.release();  // leaks intentionally — lifetime managed by parent's add_child
            },
            py::return_value_policy::reference,
            "config"_a)
        // Bind existing add_child for deferred parent binding
        .def("add_child",
            [](ft::core::Module& m, const std::string& name,
               ft::core::Module& child) -> ft::core::Module* {
                // add_child takes unique_ptr, but Python holds raw ptr.
                // We need a version that accepts a raw pointer.
                // For now, use a helper that wraps in unique_ptr(no-op deleter).
                return m.add_child(name, std::unique_ptr<ft::core::Module>(&child));
            },
            py::return_value_policy::reference,
            "name"_a, "child"_a)
```

**Important:** The `add_child` binding above has a subtle issue — the `unique_ptr` with `&child` would call `delete` on the child when the parent is destroyed, but the child was created by `create_module` and its raw pointer was returned. We need the parent to take ownership. The cleanest approach is to have `create_module` return a `unique_ptr`-wrapped object. Let me revise:

```cpp
        // Standalone module creation — returns a new module via registry
        m.def("create_module",
            [](turbomind::core::ModuleConfig& config) -> ft::core::Module* {
                auto mod = ft::core::Module::create(config);
                if (!mod) return nullptr;
                // Release ownership; Python holds raw pointer until add_child takes it.
                return mod.release();
            },
            py::return_value_policy::take_ownership,
            "config"_a)
```

Actually, pybind ownership semantics are tricky here. The simplest approach: use a `shared_ptr` holder:

```cpp
        // Module holder for standalone-created modules
        py::class_<ft::core::Module, std::shared_ptr<ft::core::Module>>(m, "ModuleHolder")
            .def("get", [](std::shared_ptr<ft::core::Module>& p) -> ft::core::Module* { return p.get(); },
                 py::return_value_policy::reference);

        m.def("create_module",
            [](turbomind::core::ModuleConfig& config) -> std::shared_ptr<ft::core::Module> {
                auto mod = ft::core::Module::create(config);
                if (!mod) return nullptr;
                return std::shared_ptr<ft::core::Module>(mod.release());
            },
            "config"_a)
```

Wait — the `Module` pybind class is already bound without `shared_ptr` holder. We can't mix holders. The cleanest approach: `create_module` returns a raw pointer, and `add_child` accepts a raw pointer using a custom no-op deleter for the `unique_ptr`:

```cpp
        m.def("create_module",
            [](turbomind::core::ModuleConfig& config) -> ft::core::Module* {
                auto mod = ft::core::Module::create(config);
                return mod.release();
            },
            py::return_value_policy::reference,
            "config"_a)
        // add_child: transfers ownership of a previously create_module'd child
        .def("add_child_raw",
            [](ft::core::Module& parent, const std::string& name,
               ft::core::Module* child) -> ft::core::Module* {
                // Wrap raw pointer in unique_ptr with no-op deleter, then
                // transfer to parent. The parent's add_child stores it via
                // reset() on its unique_ptr member.
                // BUT: this leaks if parent is destroyed before child is added.
                // Instead, we need proper ownership transfer.
                // Solution: use a special overload that takes ownership.
                auto owned = std::unique_ptr<ft::core::Module>(child);
                return parent.add_child(name, std::move(owned));
            },
            py::return_value_policy::reference,
            "name"_a, "child"_a)
```

This works because `create_module` gives us a raw pointer (Python keeps it alive via `return_value_policy::reference`), and `add_child_raw` takes ownership via wrapping in `unique_ptr`. Python must not use the child pointer after calling `add_child_raw`.

- [ ] **Step 4: Build and verify C++ changes**

Run from `build` directory:
```bash
ninja
```

Expected: Clean build with no errors.

- [ ] **Step 5: Quick Python smoke test**

```bash
PYTHONPATH=lmdeploy:build/lib python -c "
import _turbomind as _tm
cfg = _tm.ModuleListConfig()
m = _tm.create_module(cfg)
print('create_module:', m)
print('type:', m.type())
"
```

Expected: Module created successfully, type = "ModuleList".

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc src/turbomind/python/bind.cpp
git commit -m "feat(core): add Module::create() factory and expose to Python"
```

---

### Task 2: Create `builder.py` — Builder base class with commit internals

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder.py`
- Read: `lmdeploy/turbomind/deploy/commit.py` (source for commit internals)
- Read: `lmdeploy/turbomind/deploy/distributor.py` (source for Distributor)

This task creates the `Builder` base class that absorbs `Distributor` and the core commit functions from `commit.py`. The old files are NOT deleted yet — they coexist until Task 6.

- [ ] **Step 1: Create `builder.py` with Builder base class**

```python
# lmdeploy/turbomind/deploy/builder.py
# Copyright (c) OpenMMLab. All rights reserved.
"""Builder classes for model loading. Absorbs Distributor + commit logic."""
from __future__ import annotations

import enum

import torch

import _turbomind as _tm

from .linear import Linear


class SplitSide(enum.Enum):
    """Internal TP split direction — not exposed to specs."""
    OUTPUT = "output"
    INPUT = "input"


_SPLIT_SIDE_TO_DIM: dict[SplitSide, int] = {
    SplitSide.OUTPUT: -1,
    SplitSide.INPUT: 0,
}

# --- C++ dtype mappings (from commit.py) ---

_STR_TO_DTYPE: dict[str, _tm.DataType] = {
    'float32':  _tm.DataType.TYPE_FP32,
    'float16':  _tm.DataType.TYPE_FP16,
    'bfloat16': _tm.DataType.TYPE_BF16,
}

_TORCH_TO_CPP: dict[torch.dtype, _tm.DataType] = {
    torch.float32:  _tm.DataType.TYPE_FP32,
    torch.float16:  _tm.DataType.TYPE_FP16,
    torch.bfloat16: _tm.DataType.TYPE_BF16,
    torch.int32:    _tm.DataType.TYPE_INT32,
    torch.int64:    _tm.DataType.TYPE_INT64,
    torch.int8:     _tm.DataType.TYPE_INT8,
    torch.uint8:    _tm.DataType.TYPE_UINT8,
}

_FP8_DTYPES: set[torch.dtype] = {torch.uint8}
for _fp8_attr in ('float8_e4m3fn', 'float8_e5m2fn'):
    _fp8_dt = getattr(torch, _fp8_attr, None)
    if _fp8_dt is not None:
        _FP8_DTYPES.add(_fp8_dt)


def _cpp_dtype(dtype_str: str):
    return _STR_TO_DTYPE[dtype_str]


def _torch_dtype_to_cpp(dtype: torch.dtype):
    return _TORCH_TO_CPP.get(dtype)


def _cast_shard_for_tm(shard: torch.Tensor, tm_tensor) -> torch.Tensor:
    if tm_tensor.type == _tm.DataType.TYPE_FP32 and shard.dtype in (torch.float16, torch.bfloat16):
        return shard.float()
    if tm_tensor.type == _tm.DataType.TYPE_FP16 and shard.dtype != torch.float16:
        return shard.half()
    if tm_tensor.type == _tm.DataType.TYPE_BF16 and shard.dtype != torch.bfloat16:
        return shard.to(torch.bfloat16)
    return shard


def _infer_cpp_linear_dtype(linear: Linear):
    fmt = linear.weight_format
    if fmt is not None and fmt.cpp_dtype_name is not None:
        cpp_dtype = getattr(_tm.DataType, fmt.cpp_dtype_name, None)
        if cpp_dtype is not None:
            return cpp_dtype, fmt.block_in or 0
    weight = linear.tensors.get("weight")
    if weight is not None:
        return _TORCH_TO_CPP.get(weight.dtype), 0
    return None, 0


def _infer_compute_dtype(linear: Linear):
    w = linear.tensors.get('weight')
    if w is not None:
        d = _TORCH_TO_CPP.get(w.dtype)
        if d is not None:
            return d
        if w.dtype in _FP8_DTYPES:
            return _tm.DataType.TYPE_BF16
    for key in ('scales', 'bias'):
        t = linear.tensors.get(key)
        if t is not None:
            d = _TORCH_TO_CPP.get(t.dtype)
            if d is not None:
                return d
    return None


def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int,
                    in_dim: int, out_dim: int,
                    model_dtype=None):
    """Core tensor commit — moved from commit.py."""
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None
    packer = linear.weight_format.packer if linear.weight_format else None
    fmt = linear.weight_format
    is_quantized = fmt is not None and fmt.name != 'trivial'

    for kind, tensor in linear.tensors.items():
        if packer is not None:
            tensor = packer(tensor, kind)

        tensor_split_dim = split_dim
        if kind == "bias" and split_side == SplitSide.INPUT:
            tensor_split_dim = None

        if tensor_split_dim is not None and split_num > 1:
            split_size = tensor.shape[tensor_split_dim] // split_num
            shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        else:
            shard = tensor

        if not shard.is_cuda:
            shard = shard.cuda(0).contiguous()
        elif not shard.is_contiguous():
            shard = shard.contiguous()

        if kind == "weight" and is_quantized:
            alloc_shape = [in_dim, out_dim]
            alloc_dtype = cpp_dtype
        elif kind == "weight" and model_dtype is not None:
            alloc_shape = list(shard.shape)
            alloc_dtype = model_dtype
        else:
            alloc_shape = list(shard.shape)
            alloc_dtype = _torch_dtype_to_cpp(shard.dtype)

        dst = handle.param(kind).alloc(alloc_shape, alloc_dtype)
        shard = _cast_shard_for_tm(shard, dst)
        if dst.byte_size != shard.nbytes and dst.byte_size > shard.nbytes:
            pad_dim = tensor_split_dim if tensor_split_dim is not None else -1
            if pad_dim < 0:
                pad_dim = shard.dim() + pad_dim
            outer = shard.numel() // shard.shape[pad_dim]
            extra = (dst.byte_size - shard.nbytes) // (outer * shard.element_size())
            new_shape = list(shard.shape)
            new_shape[pad_dim] += extra
            padded = torch.zeros(new_shape, dtype=shard.dtype, device=shard.device)
            idx = [slice(None)] * shard.dim()
            idx[pad_dim] = slice(0, shard.shape[pad_dim])
            padded[tuple(idx)].copy_(shard)
            shard = padded
        dst.copy_from(shard)


# =====================================================================
# Builder base class
# =====================================================================

class Builder:
    """Base class for all module builders.

    Creates C++ module instances on each GPU via registry.
    Auto-binds children via __setattr__.
    """

    def __init__(self, config, contexts, tp=1, ranks=None):
        self.config = config
        self._contexts = contexts
        self._tp = tp
        self._ranks = ranks
        self.handles = []
        for ctx in self._contexts:
            with ctx:
                handle = _tm.create_module(config.to_cpp())
                self.handles.append(handle)

    def __setattr__(self, name, value):
        if isinstance(value, Builder):
            for parent_h, child_h in zip(self.handles, value.handles):
                parent_h.add_child_raw(name, child_h)
            return
        super().__setattr__(name, value)

    def __setitem__(self, index, value):
        if isinstance(value, Builder):
            for parent_h, child_h in zip(self.handles, value.handles):
                parent_h.add_child_raw(str(index), child_h)
            return
        raise TypeError(f"Cannot set item of type {type(value)}")

    # --- Internal commit helpers ---

    def _commit_linear(self, name, linear: Linear, split_side: SplitSide | None,
                       model_dtype=None):
        """Create LinearWeight child and commit tensor data."""
        if linear is None:
            return

        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if group_size == 0:
            group_size = 128

        from .kind_map import TRIVIAL_FORMAT
        from .linear import Linear as Lin

        # Ensure DataFormat
        if linear.data_format is None and linear.weight_format is not None:
            linear = Lin(tensors=linear.tensors,
                         weight_format=linear.weight_format,
                         data_format=linear.weight_format.to_data_format(
                             cpp_dtype if cpp_dtype else _tm.DataType.TYPE_INVALID,
                             group_size))

        tp = self._tp if split_side else 1
        for i, handle in enumerate(self.handles):
            with self._contexts[i]:
                rank = self._ranks[i] if self._ranks and tp > 1 else 0
                # Ensure LinearWeight child exists
                linear_mod = handle.child(name)
                if linear_mod is None:
                    w = linear.tensors.get('weight')
                    if w is None:
                        continue
                    in_dim = w.shape[0]
                    out_dim = w.shape[-1]
                    if split_side == SplitSide.OUTPUT:
                        out_dim = out_dim // tp
                    elif split_side == SplitSide.INPUT:
                        in_dim = in_dim // tp
                    compute_dtype = _infer_compute_dtype(linear)
                    if model_dtype is not None:
                        compute_dtype = model_dtype
                    lin_cfg = _tm.LinearConfig()
                    lin_cfg.input_dim = in_dim
                    lin_cfg.output_dim = out_dim
                    lin_cfg.data_type = compute_dtype if compute_dtype else _tm.DataType.TYPE_INVALID
                    lin_cfg.has_bias = 'bias' in linear.tensors
                    linear_mod = handle.create_child(name, lin_cfg)

                # Block-scale TP validation
                if split_side == SplitSide.OUTPUT and tp > 1:
                    wfmt = linear.weight_format
                    if wfmt is not None and wfmt.block_out:
                        for kind, tensor in linear.tensors.items():
                            if kind in ("scales", "zeros"):
                                n_blocks = tensor.size(-1)
                                assert n_blocks % tp == 0

                linear_mod.set_weight_spec(cpp_dtype, group_size)

                w = linear.tensors.get('weight')
                in_dim = w.shape[0] if w else 0
                out_dim = w.shape[-1] if w else 0
                if split_side == SplitSide.OUTPUT:
                    out_dim = out_dim // tp
                elif split_side == SplitSide.INPUT:
                    in_dim = in_dim // tp

                _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                                split_side, tp, rank, in_dim, out_dim,
                                model_dtype=model_dtype)

    def _commit_tensor(self, name, tensor: torch.Tensor | None,
                       split_side: SplitSide | None = None):
        """Commit a raw tensor to all GPU handles."""
        if tensor is None:
            return
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self.handles):
            with self._contexts[i]:
                rank = self._ranks[i] if self._ranks and tp > 1 else 0
                if split_dim is not None and tp > 1:
                    split_size = tensor.shape[split_dim] // tp
                    shard = tensor.split(split_size, dim=split_dim)[rank]
                else:
                    shard = tensor
                if not shard.is_cuda:
                    shard = shard.cuda(0).contiguous()
                elif not shard.is_contiguous():
                    shard = shard.contiguous()
                cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
                dst = handle.param(name).alloc(list(shard.shape), cpp_dtype)
                shard = _cast_shard_for_tm(shard, dst)
                dst.copy_from(shard)

    def _add_norm_child(self, name, tensor, data_type=None):
        """Create a NormConfig child and commit weight tensor."""
        from .module_configs import NormConfig
        cfg = NormConfig(dim=tensor.shape[-1], data_type=data_type or 0)
        for i, handle in enumerate(self.handles):
            with self._contexts[i]:
                child = handle.create_child(name, cfg.to_cpp())
                if tensor is not None:
                    if not tensor.is_cuda:
                        tensor = tensor.cuda(0).contiguous()
                    elif not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    cpp_dtype = _torch_dtype_to_cpp(tensor.dtype)
                    dst = child.param('weight').alloc(list(tensor.shape), cpp_dtype)
                    tensor = _cast_shard_for_tm(tensor, dst)
                    dst.copy_from(tensor)


class TextModelBuilder(Builder):
    """Wraps pre-existing C++ root ModelWeight handles."""
    def __init__(self, handles, contexts):
        # Skip Builder.__init__ — don't create new modules
        self.handles = handles
        self._contexts = contexts
        self._tp = 1
        self._ranks = None


class DecoderLayerBuilder(Builder):
    """Decoder layer — pure container, no weight logic."""
    pass


class ModuleListBuilder(Builder):
    """Module list — indexed children via __setitem__."""
    pass


class NormBuilder(Builder):
    """Norm weight builder."""
    def set_weight(self, tensor, data_type=None):
        self._commit_tensor('weight', tensor)
```

- [ ] **Step 2: Verify imports work**

```bash
PYTHONPATH=lmdeploy:build/lib python -c "
from lmdeploy.turbomind.deploy.builder import Builder, TextModelBuilder, SplitSide
print('Builder imported successfully')
print('SplitSide.OUTPUT:', SplitSide.OUTPUT)
print('SplitSide.INPUT:', SplitSide.INPUT)
"
```

Expected: No import errors.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "feat(deploy): add Builder base class with commit internals"
```

---

### Task 3: Add concrete builders — AttentionBuilder, FfnBuilder, MoeBuilder

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py`

- [ ] **Step 1: Add AttentionBuilder**

Append to `builder.py`:

```python
class AttentionBuilder(Builder):
    """Attention weight loading builder."""

    _PARAM_TP_RULES: dict[str, SplitSide] = {
        'sinks': SplitSide.OUTPUT,
    }

    def add_qkv_proj(self, q, k, v):
        from .spec import merge_qkv_linear
        merged = merge_qkv_linear(
            q, k, v,
            tp=self._tp,
            head_dim=self.config.head_dim,
            rope_dim=self.config.head_dim,  # overridden by spec if partial
            permute_qk=True,
            attn_output_gate=self.config.attn_output_gate,
            repeat_kv=0,
            kv_head_num=self.config.kv_head_num,
        )
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)

    def add_o_proj(self, o):
        self._commit_linear('wo', o, SplitSide.INPUT,
                            model_dtype=self.config.data_type)

    def add_qk_norm(self, q, k):
        if q is not None:
            self._add_norm_child('q_norm', q, data_type=self.config.data_type)
        if k is not None:
            self._add_norm_child('k_norm', k, data_type=self.config.data_type)

    def add_param(self, name, tensor):
        split_side = self._PARAM_TP_RULES.get(name)
        self._commit_tensor(name, tensor, split_side)

    def add_linear(self, name, linear):
        """Commit an attention linear with built-in TP rules."""
        from .commit import _ATTN_TP_RULES
        rule = _ATTN_TP_RULES.get(name, {})
        self._commit_linear(name, linear, model_dtype=self.config.data_type, **rule)
```

**Note:** `add_linear` is a temporary bridge method that uses `_ATTN_TP_RULES` from the old `commit.py`. This will be replaced when specs are fully migrated to call `add_qkv_proj`/`add_o_proj` directly.

- [ ] **Step 2: Add FfnBuilder**

Append to `builder.py`:

```python
class FfnBuilder(Builder):
    """FFN weight loading builder with w1+w3 fusion."""

    def add_ffn(self, w1, w2, w3):
        from .transforms import fuse_ffn_linears
        fused, fused_silu = None, False
        if w1 is not None and w3 is not None:
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self._tp,
                self.config.act_type if hasattr(self.config, 'act_type') else 'silu',
                is_moe=getattr(self.config, 'fused_moe', False))

        # Update fuse_silu on C++ config
        for handle in self.handles:
            # fuse_silu is consumed during prepare(), safe to update now
            pass  # TODO: add set_config_field binding if needed

        if fused is not None:
            self._commit_linear('w1w3', fused, SplitSide.OUTPUT,
                                model_dtype=self.config.data_type if hasattr(self.config, 'data_type') else None)
        else:
            if w1 is not None:
                self._commit_linear('w1', w1, SplitSide.OUTPUT,
                                    model_dtype=self.config.data_type if hasattr(self.config, 'data_type') else None)
            if w3 is not None:
                self._commit_linear('w3', w3, SplitSide.OUTPUT,
                                    model_dtype=self.config.data_type if hasattr(self.config, 'data_type') else None)
        if w2 is not None:
            self._commit_linear('w2', w2, SplitSide.INPUT,
                                model_dtype=self.config.data_type if hasattr(self.config, 'data_type') else None)
```

- [ ] **Step 3: Add MoeBuilder**

Append to `builder.py`:

```python
class MoeBuilder(Builder):
    """MoE weight loading builder."""

    def add_gate(self, name, linear, model_dtype=None):
        self._commit_linear(name, linear, split_side=None, model_dtype=model_dtype)

    def add_param(self, name, tensor, split_side=None):
        self._commit_tensor(name, tensor, split_side)
```

- [ ] **Step 4: Verify imports**

```bash
PYTHONPATH=lmdeploy:build/lib python -c "
from lmdeploy.turbomind.deploy.builder import (
    Builder, TextModelBuilder, DecoderLayerBuilder,
    ModuleListBuilder, NormBuilder,
    AttentionBuilder, FfnBuilder, MoeBuilder,
    SplitSide
)
print('All builders imported successfully')
"
```

Expected: No import errors.

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "feat(deploy): add AttentionBuilder, FfnBuilder, MoeBuilder"
```

---

### Task 4: Change loader to all-upfront loading

**Files:**
- Modify: `lmdeploy/turbomind/deploy/loader.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/base.py`

This changes the loading flow from per-layer batches to a single batch with all weights.

- [ ] **Step 1: Add `all_items()` method to `SafetensorsLoader`**

In `lmdeploy/turbomind/deploy/loader.py`, add a new method to `SafetensorsLoader` after `items()` (line ~105):

```python
    def all_items(self) -> dict:
        """Return ALL weights in a single dict (mmap-backed, no eager load)."""
        all_params = {}
        for shard in self.shards:
            with safe_open(shard, 'pt') as f:
                for k in f.keys():
                    if k not in self.index:
                        continue
                    all_params[self.map_key(k)] = f.get_tensor(k)
        return all_params
```

- [ ] **Step 2: Add `all_items()` method to `PytorchLoader`**

Add after `PytorchLoader.items()` (line ~145):

```python
    def all_items(self) -> dict:
        """Return ALL weights in a single dict."""
        all_params = {}
        for shard in self.shards:
            tmp = torch.load(shard, map_location='cpu', weights_only=True)
            for k, v in tmp.items():
                all_params[self.map_key(k)] = v
            del tmp
        return all_params
```

- [ ] **Step 3: Add `all_items()` to `BaseLoader`**

Add the abstract method to `BaseLoader` (line ~59):

```python
    @abstractmethod
    def all_items(self) -> dict:
        """Return ALL weights in a single dict."""
        pass
```

- [ ] **Step 4: Update `BaseInputModel.readers()` in source_model/base.py**

Change `readers()` to yield a single entry:

```python
    def readers(self) -> Iterator:
        """Yield a single (layer_id=-1, spec) pair with ALL weights."""
        import torch
        from ..loader import create_loader
        loader = create_loader(self.model_path, self._layer_pattern, self._loader_mappings)
        all_params = loader.all_items()
        yield -1, self._spec_class(all_params, self.model_config)
        torch.cuda.empty_cache()
```

- [ ] **Step 5: Verify backward compatibility**

The old `items()` method still exists and works. The new `readers()` now yields one entry instead of many. Old code that iterates `(i, reader)` pairs will still work — it just gets one pair.

```bash
PYTHONPATH=lmdeploy:build/lib python -c "
from lmdeploy.turbomind.deploy.loader import SafetensorsLoader
print('Loader imports work')
"
```

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/loader.py lmdeploy/turbomind/deploy/source_model/base.py
git commit -m "feat(deploy): add all-upfront weight loading via all_items()"
```

---

### Task 5: Update TextModelSpec base class

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

Add `model()` abstract method and context injection points while keeping existing methods for backward compatibility.

- [ ] **Step 1: Add new fields and abstract method to TextModelSpec**

In `spec.py`, update the `TextModelSpec` class. Add after the existing fields (around line 124):

```python
    # --- Builder-driven loading context (injected by TextModelLoader) ---
    _contexts: list = None  # GPU contexts
    _root_handles: list = None  # Root C++ ModelWeight handles

    @abstractmethod
    def model(self):
        """Build the full model hierarchy using builders.

        Called once by the loader with all weights available.
        Subclasses must implement this.
        """
```

**Note:** This adds a new abstract method. All existing spec subclasses will break until they implement `model()`. We'll add a default no-op implementation temporarily:

Actually, since this would break all specs immediately, add it as a non-abstract method with a default that raises NotImplementedError:

```python
    def model(self):
        """Build the full model hierarchy using builders.

        Called once by the loader with all weights available.
        Override in subclasses to use builder-driven loading.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement model(). "
            f"Use load_global/load_layer pattern instead.")
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "feat(deploy): add model() method to TextModelSpec base class"
```

---

### Task 6: Simplify TextModelLoader

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py`

This task replaces the old `_process_*` orchestration with builder-driven loading, while keeping a fallback for specs that don't implement `model()`.

- [ ] **Step 1: Update `BaseOutputModel.export()` to call `spec.model()`**

In `lmdeploy/turbomind/deploy/target_model/base.py`, modify the `export()` method (line ~107):

```python
    def export(self) -> None:
        """Export to turbomind model format."""
        from tqdm import tqdm
        pbar = tqdm(total=1, desc='Convert to turbomind format', leave=False)
        for i, reader in self.input_model.readers():
            self.model(i, reader)
            pbar.update(1)
        pbar.close()
```

The progress bar now tracks 1 step instead of num_layer steps.

- [ ] **Step 2: Update `TextModelLoader.__call__` to try builder path**

In `text_model_loader.py`, modify `__call__`:

```python
    def __call__(self, layer: int, spec: 'TextModelSpec'):
        # Try builder-driven path first
        if hasattr(spec, 'model'):
            try:
                spec._contexts = self._contexts
                spec._root_handles = [h for h in self._root._handles]
                spec.model()
                return 1
            except NotImplementedError:
                pass  # Fall through to legacy path

        # Legacy path
        if layer < 0:
            self._load_global(spec)
        elif layer >= self.model.model_config.num_layer:
            return 0
        else:
            self._load_layer(layer, spec)
        return 1
```

- [ ] **Step 3: Test backward compatibility — all existing specs still work**

Run the turbomind-tester agent on a Qwen3 model to verify existing loading still works.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py lmdeploy/turbomind/deploy/target_model/base.py
git commit -m "feat(deploy): add builder-driven loading path with legacy fallback"
```

---

### Task 7: Migrate Qwen3Spec — canonical builder-driven spec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

This is the first spec to use builder-driven loading. It serves as the reference implementation for all other specs.

- [ ] **Step 1: Read current Qwen3Spec implementation**

Read `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` to understand the current weight reading methods. These methods (`_read_attn_linears`, `ffn_linears`, `moe_ffn_linears`, etc.) stay the same — only the top-level orchestration changes.

- [ ] **Step 2: Implement `model()` on Qwen3Spec**

Add the `model()` method and supporting builder-returning methods. The weight reading helpers (`_read_attn_linears`, `_read_ffn_linears`, etc.) remain unchanged.

The implementation will follow the pattern from the design spec. Key methods to add:
- `_configure()` — set attention config fields from model config
- `norm(pfx)` — return NormBuilder
- `attn(pfx, layer)` — return AttentionBuilder
- `ffn(pfx, layer)` — return FfnBuilder
- `moe(pfx, layer)` — return MoeBuilder with expert iteration
- `layers(pfx)` — return ModuleListBuilder
- `model()` — build full hierarchy

This is the most code-heavy step. The engineer should study the existing `_read_attn_linears`, `ffn_linears`, `moe_ffn_linears`, `attn_params`, `attn_norm_children`, `moe_gate`, `moe_params`, `num_experts` methods and wire them into builder methods.

- [ ] **Step 3: Test with Qwen3 model**

Use the turbomind-tester agent with a Qwen3 model. Verify:
- Model loads without errors
- Response contains meaningful text (not gibberish)
- Response is at least 128 tokens

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "feat(deploy): migrate Qwen3Spec to builder-driven loading"
```

---

### Task 8: Migrate remaining specs

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

Each spec follows the same pattern as Task 7. Key differences:

- **GptOssSpec**: Packed expert tensors, gate/up deinterleaving, attention sinks
- **Qwen3_5Spec**: Linear attention (GDN), zero-centered RMSNorm, shared expert gate, packed MoE
- **Glm4MoeLiteSpec**: MLA attention with `kv_b_proj` folding, score correction bias

- [ ] **Step 1: Migrate GptOssSpec**
- [ ] **Step 2: Test GptOss model**
- [ ] **Step 3: Migrate Qwen3_5Spec**
- [ ] **Step 4: Test Qwen3.5 model**
- [ ] **Step 5: Migrate Glm4MoeLiteSpec**
- [ ] **Step 6: Test Glm4 model**
- [ ] **Step 7: Commit each migration separately**

---

### Task 9: Delete legacy code

**Files:**
- Delete: `lmdeploy/turbomind/deploy/distributor.py`
- Delete: `lmdeploy/turbomind/deploy/transforms.py`
- Gut: `lmdeploy/turbomind/deploy/text_model_loader.py` (remove all `_process_*` and `_load_*` methods)
- Clean: `lmdeploy/turbomind/deploy/commit.py` (remove functions now in builder.py, keep only what's still imported)
- Clean: `lmdeploy/turbomind/deploy/spec.py` (remove old abstract methods like `attn_linears`, `ffn_linears` if no longer used)

- [ ] **Step 1: Remove `_process_*` and `_load_*` from TextModelLoader**
- [ ] **Step 2: Remove legacy fallback in `__call__`**
- [ ] **Step 3: Delete `distributor.py`**
- [ ] **Step 4: Move remaining transforms into builder.py, delete `transforms.py`**
- [ ] **Step 5: Clean up `commit.py` — remove duplicated logic**
- [ ] **Step 6: Clean up `spec.py` — remove unused abstract methods**
- [ ] **Step 7: Test all models after cleanup**
- [ ] **Step 8: Commit**

```bash
git add -A lmdeploy/turbomind/deploy/
git commit -m "refactor(deploy): remove legacy loading code, builder-driven loading complete"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run turbomind-tester agent on all 13 models with TP=1**
- [ ] **Step 2: Run turbomind-tester agent on all 13 models with TP=2**
- [ ] **Step 3: Verify quantized formats still work (AWQ, GPTQ, FP8)**
- [ ] **Step 4: Commit verification results**

---

## Self-Review

**Spec coverage:** Each section of the design spec maps to tasks:
- Builder base class → Task 2
- TP split knowledge → Task 2 (SplitSide internal to builder.py)
- Concrete builders → Task 3
- Weight loading (all-upfront) → Task 4
- Spec integration → Task 5
- TextModelLoader simplification → Task 6
- C++ changes → Task 1
- File changes → Tasks 9, 10
- Verification → Task 10

**Placeholder scan:** Task 7 has a "TODO: add set_config_field binding if needed" in FfnBuilder. This needs resolution — either add a C++ binding for updating config fields, or create the FfnWeight with the correct fuse_silu from the start (requires reading weights before creating the module).

**Type consistency:** The `add_child_raw` name in C++ binding must match the Python call in `Builder.__setattr__`. Both use `add_child_raw`.
