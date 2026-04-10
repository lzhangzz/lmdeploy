# Eliminate `with_context` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `with_context` wrapper from `bind.cpp` by moving context management to the Python `Distributor` level.

**Architecture:** A new `PyContextGuard` struct in `bind.cpp` wraps `Stream` + `Allocator` copies and manages a `std::optional<ContextGuard>` lifecycle tied to Python's `__enter__`/`__exit__`. The Python `Distributor` stores one `PyContextGuard` per GPU and wraps commit operations with them.

**Tech Stack:** C++ (pybind11), Python

---

### Task 1: Expose per-device stream/allocator from TurboMind

**Files:**
- Modify: `src/turbomind/turbomind.h:16-52`
- Modify: `src/turbomind/turbomind.cc:727-730`

TurboMind needs a public method that exposes the Stream and Allocator for a given device index, so the binding can create `PyContextGuard` objects.

- [ ] **Step 1: Add `weight_context` declaration to TurboMind header**

In `src/turbomind/turbomind.h`, add a public method after the existing `root()` declaration:

```cpp
    /// Returns the Stream and Allocator for GPU `index`'s weight tree.
    std::pair<core::Stream, core::Allocator> weight_context(int index);
```

Add the necessary include for `<utility>` if not already present (it is already included via `<memory>` / `<string>`).

- [ ] **Step 2: Implement `weight_context` in turbomind.cc**

In `src/turbomind/turbomind.cc`, after the `root()` method (line 730):

```cpp
std::pair<core::Stream, core::Allocator> TurboMind::weight_context(int index)
{
    auto& mw = impl_->weights_.at(index);
    TM_CHECK(mw != nullptr);
    return {mw->stream(), mw->allocator()};
}
```

- [ ] **Step 3: Add `stream()` and `allocator()` accessors to ModelWeight**

In `src/turbomind/models/model_weight.h`, add public accessors after the `context()` method (line 34):

```cpp
    const core::Stream&    stream() const    { return stream_; }
    const core::Allocator& allocator() const { return alloca_; }
```

- [ ] **Step 4: Build and verify compilation**

Run: `cd build && ninja _turbomind 2>&1 | tail -20`
Expected: Clean build, no errors.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/turbomind.h src/turbomind/turbomind.cc src/turbomind/models/model_weight.h
git commit -m "feat: expose per-device stream/allocator from TurboMind for Python context management"
```

---

### Task 2: Add PyContextGuard and TurboMind.context() binding

**Files:**
- Modify: `src/turbomind/python/bind.cpp:566-692`

Add the `PyContextGuard` struct, bind it as a Python context manager, and add `TurboMind.context(index)`. Remove the `with_context` lambda and unwrap all Module method bindings.

- [ ] **Step 1: Add `PyContextGuard` struct and its binding**

In `bind.cpp`, replace the `with_context` lambda (lines 566-579) with:

```cpp
    // Python context manager wrapper for ContextGuard.
    // Stores copies of Stream + Allocator; constructs the real guard
    // in-place on __enter__ and destroys it on __exit__.
    struct PyContextGuard {
        ft::core::Stream    stream;
        ft::core::Allocator alloc;
        std::optional<ft::core::ContextGuard> guard;

        PyContextGuard(ft::core::Stream s, ft::core::Allocator a)
            : stream(std::move(s)), alloc(std::move(a)) {}

        void enter() { guard.emplace(stream, alloc); }
        void exit()  { guard.reset(); }
    };

    py::class_<PyContextGuard>(m, "ContextGuard")
        .def("__enter__", [](PyContextGuard& g) -> PyContextGuard& { g.enter(); return g; })
        .def("__exit__", [](PyContextGuard& g, py::object, py::object, py::object) { g.exit(); });
```

Note: add `#include <optional>` to the includes at the top of the file if not already present.

- [ ] **Step 2: Add `context(index)` to TurboMind binding**

In the TurboMind pybind class (after the `root` binding around line 673), add:

```cpp
        .def("context",
             [](ft::TurboMind* model, int index) -> std::unique_ptr<PyContextGuard> {
                 auto [stream, alloc] = model->weight_context(index);
                 return std::make_unique<PyContextGuard>(std::move(stream), std::move(alloc));
             },
             "index"_a)
```

- [ ] **Step 3: Remove `with_context` wrappers from Module bindings**

Replace the entire Module binding block (lines 581-642) with direct bindings. The old code captures `with_context` in each lambda -- remove all those captures and the `with_context` call:

```cpp
    // Module class — navigation and allocation interface
    py::class_<ft::core::Module, std::shared_ptr<ft::core::Module>>(m, "Module")
        .def("get",
             [](ft::core::Module& m, const std::string& segment) -> ft::core::Module* {
                 return m.get(segment);
             },
             py::return_value_policy::reference,
             "segment"_a)
        .def("alloc",
             [](ft::core::Module& m, const std::string& param_name, ft::DataType dtype, int group_size) {
                 return std::make_shared<Tensor>(m.alloc(param_name, ft::core::WeightSpec{dtype, group_size}));
             },
             "param_name"_a,
             "dtype"_a,
             "group_size"_a = 0)
        .def("create_param",
             [](ft::core::Module& m,
                const std::string& name,
                std::vector<size_t> shape,
                ft::DataType dtype,
                int group_size) {
                 return std::make_shared<Tensor>(
                     m.create_param(name, shape, dtype, group_size));
             },
             "name"_a,
             "shape"_a,
             "dtype"_a,
             "group_size"_a = 0)
        .def("prepare",
             [](ft::core::Module& m) { m.prepare(); })
        .def("child",
             [](ft::core::Module& m, const std::string& name) -> ft::core::Module* { return m.child(name); },
             py::return_value_policy::reference,
             "name"_a)
        // Config-based create_child: accepts any ModuleConfig subclass
        .def("create_child",
            [](ft::core::Module& m, const std::string& name,
               turbomind::core::ModuleConfig& config) -> ft::core::Module* {
                return m.create_child(name, config);
            },
            py::return_value_policy::reference,
            "name"_a, "config"_a)
        .def("type", [](ft::core::Module& m) -> const char* { return m.type(); })
        .def("full_path", [](ft::core::Module& m) -> std::string { return m.full_path(); })
        .def("__getitem__",
             [](ft::core::Module& m, const std::string& key) -> ft::core::Module* {
                 return m.get(key);
             },
             py::return_value_policy::reference)
        .def("__getitem__",
             [](ft::core::Module& m, int idx) -> ft::core::Module* {
                 return m.get(std::to_string(idx));
             },
             py::return_value_policy::reference);
```

- [ ] **Step 4: Build and verify**

Run: `cd build && ninja _turbomind 2>&1 | tail -20`
Expected: Clean build. The `with_context` lambda is fully removed.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "refactor: replace with_context with PyContextGuard context manager in binding"
```

---

### Task 3: Update Distributor to store and use context managers

**Files:**
- Modify: `lmdeploy/turbomind/deploy/distributor.py:1-66`

The Distributor gains a `contexts` parameter and wraps commit operations with context guards.

- [ ] **Step 1: Update Distributor constructor and commit methods**

Replace the entire file content with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Distributor: distributes module creation and weight commits across all GPUs."""
from __future__ import annotations

from .load_context import commit_linear, commit_tensor


class Distributor:
    """Wraps N GPU handles for a single logical module.

    Distributes create_child / commit_linear / commit_tensor
    across all GPUs with bound TP configuration.
    """

    def __init__(self, handles, contexts=None, tp=1, ranks=None):
        self._handles = handles
        self._contexts = contexts or [None] * len(handles)
        self._tp = tp
        self._ranks = ranks

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx):
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    def create_child(self, name, config, tp=None, ranks=None):
        """Create a typed module child on ALL GPUs.

        Calls ``config.for_rank(rank).to_cpp()`` per GPU.
        Returns a new Distributor scoped to the created children,
        with tp/ranks rebound if provided (otherwise inherited).
        """
        new_tp = tp if tp is not None else self._tp
        new_ranks = ranks if ranks is not None else self._ranks
        children = []
        for i, handle in enumerate(self._handles):
            with self._contexts[i] or _noop():
                rank = new_ranks[i] if new_ranks and new_tp > 1 else 0
                child = handle.create_child(name, config.for_rank(rank).to_cpp())
                children.append(child)
        return Distributor(children, self._contexts, tp=new_tp, ranks=new_ranks)

    def commit_linear(self, name, linear, split_side=None, model_dtype=None):
        """Commit a Linear bundle to all GPUs.

        If split_side is given, uses bound tp/ranks for sharding.
        If split_side is None, broadcasts (tp=1).
        """
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            with self._contexts[i] or _noop():
                rank = self._rank_for(i) if tp > 1 else 0
                commit_linear(handle, linear, name,
                              split_side=split_side, split_num=tp,
                              rank=rank, model_dtype=model_dtype)

    def commit_tensor(self, name, tensor, split_side=None):
        """Commit a raw tensor to all GPUs."""
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            with self._contexts[i] or _noop():
                rank = self._rank_for(i) if tp > 1 else 0
                commit_tensor(handle, tensor, name,
                              split_side=split_side, split_num=tp,
                              rank=rank)


class _noop:
    """No-op context manager for when no context guard is available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/distributor.py
git commit -m "refactor: Distributor stores and uses context managers for GPU operations"
```

---

### Task 4: Pass context managers into the Distributor at construction sites

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py:94-96`
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:41-48`
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:208`

All places that construct a `Distributor` must now pass context managers. Contexts are obtained from `model_comm.context(index)` via `BaseOutputModel`.

- [ ] **Step 1: Add `context()` method to BaseOutputModel**

In `lmdeploy/turbomind/deploy/target_model/base.py`, add after the `root()` method (after line 96):

```python
    def context(self, index: int):
        """Return a context manager for GPU *index*'s weight loading."""
        return self.model_comm.context(index)
```

- [ ] **Step 2: Update TextModelLoader to pass contexts**

In `lmdeploy/turbomind/deploy/text_model_loader.py`, update the constructor (lines 41-48) from:

```python
        handles = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
        self._root = Distributor(handles)
```

to:

```python
        handles = []
        contexts = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
            contexts.append(model.context(gpu))
        self._root = Distributor(handles, contexts)
```

- [ ] **Step 3: Update the MoE manual Distributor construction**

In the same file, at line 208, change from:

```python
                    parent = Distributor(children)
```

to:

```python
                    parent = Distributor(children, moe._contexts)
```

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/target_model/base.py lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor: pass context managers from TurboMind through to Distributor"
```

---

### Task 5: Update LoadContext to use context managers

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py:396-513`

`LoadContext` wraps a single handle and needs a context for `load_linear` and `load_tensor`.

- [ ] **Step 1: Update LoadContext constructor to accept context**

In `lmdeploy/turbomind/deploy/load_context.py`, update `LoadContext.__init__` (line 396) from:

```python
    def __init__(self, handle, tp_config: dict,
                 model_config: 'ModelConfig | None' = None):
        self._handle = handle
        self._tp_config = tp_config
        self._model_config = model_config
```

to:

```python
    def __init__(self, handle, tp_config: dict,
                 model_config: 'ModelConfig | None' = None,
                 context=None):
        self._handle = handle
        self._tp_config = tp_config
        self._model_config = model_config
        self._context = context
```

- [ ] **Step 2: Wrap `load_linear` with context**

In `load_linear` (around line 461), wrap the body with `with self._context or _noop():`. Define a local `_noop` class (cannot import from `distributor.py` due to circular import -- `distributor.py` already imports from `load_context.py`):

```python
# At module level in load_context.py (after existing imports)
class _noop:
    """No-op context manager for when no context guard is available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
```

Then wrap `load_linear`:

```python
    def load_linear(self, name: str, linear: Linear,
                    tp_rule: str | None = None):
        with self._context or _noop():
            # ... existing load_linear body unchanged ...
```

- [ ] **Step 3: Wrap `load_tensor` with context**

Similarly for `load_tensor`:

```python
    def load_tensor(self, name: str, tensor: torch.Tensor,
                    module_type: str = 'NormWeight',
                    module_config: dict | None = None,
                    tp_rule: str | None = None):
        with self._context or _noop():
            # ... existing load_tensor body unchanged ...
```

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor: LoadContext uses context manager for load operations"
```

---

### Task 6: Build and test end-to-end

**Files:** None (verification only)

- [ ] **Step 1: Full build**

Run: `cd build && ninja _turbomind 2>&1 | tail -20`
Expected: Clean build.

- [ ] **Step 2: Test a model with the turbomind-tester agent**

Use the turbomind-tester agent to verify a model still loads and generates correct output. This tests the full path: TurboMind → create_weights → context(index) → Distributor → commit operations.

- [ ] **Step 3: Verify multi-GPU (if applicable)**

If multiple GPUs are available, test that multi-GPU weight loading works correctly with the new context management.

- [ ] **Step 4: Final commit (if any fixes needed)**

```bash
git add -u
git commit -m "fix: address issues found during end-to-end testing"
```

---

## Dependency graph

```
Task 1 (TurboMind accessor) ──> Task 2 (bind.cpp changes)
                                      │
                                      v
Task 3 (Distributor update) ──> Task 4 (construction sites)
                                      │
                                      v
Task 5 (LoadContext update) ──> Task 6 (e2e test)
```

Tasks 3 and 5 can be done in parallel. Task 4 depends on both Task 2 and Task 3.
