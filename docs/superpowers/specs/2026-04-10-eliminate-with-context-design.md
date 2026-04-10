# Eliminate `with_context` from Python Bindings

**Date:** 2026-04-10
**Status:** Draft

## Problem

The `with_context` lambda in `src/turbomind/python/bind.cpp` (lines 568-579) wraps
7 Python-callable `Module` methods. Every call walks the parent chain to the root,
`dynamic_cast`s to `ModelWeight`, and creates a temporary `ContextGuard`. This is:

- **Coupled:** The binding layer knows about `ModelWeight` internals (the
  `dynamic_cast` and `context()` call).
- **Wasteful:** Parent-chain walk + RTTI lookup on every Python call, even for
  pure lookups like `get()` / `__getitem__`.

## Solution

Move context management to the Python `Distributor` level. The `Distributor`
stores context managers obtained from `TurboMind` and wraps commit sessions
with them. Module and ModelWeight C++ code stay unchanged. No changes to
`ContextGuard` -- it stays non-copyable, non-movable.

## Design

### 1. PyContextGuard -- Python context manager wrapper

`ContextGuard` manages a thread-local LIFO stack by count. It cannot safely be
moved: if nested guards exist, moving an outer guard would pop the wrong items
from the top of the stack. It must stay non-copyable and non-movable.

Instead, create a Python-specific wrapper that:

- Stores copies of the `Stream` and `Allocator` (obtained from TurboMind)
- Constructs the `ContextGuard` in-place on `__enter__` via `std::optional`
- Destroys it on `__exit__` via `std::optional::reset`

This gives deterministic push/pop aligned with Python's `with` block, without
needing `ContextGuard` move semantics.

```cpp
// In bind.cpp
struct PyContextGuard {
    ft::core::Stream    stream_;
    ft::core::Allocator alloc_;
    std::optional<ft::core::ContextGuard> guard_;

    PyContextGuard(ft::core::Stream s, ft::core::Allocator a)
        : stream_(std::move(s)), alloc_(std::move(a)) {}

    void enter() { guard_.emplace(stream_, alloc_); }
    void exit()  { guard_.reset(); }
};
```

Bind as a Python context manager:

```cpp
py::class_<PyContextGuard>(m, "ContextGuard")
    .def("__enter__", [](PyContextGuard& g) -> PyContextGuard& { g.enter(); return g; })
    .def("__exit__", [](PyContextGuard& g, py::object, py::object, py::object) { g.exit(); });
```

### 2. TurboMind binding exposes `context(index)`

Add `context(index)` to the TurboMind binding. It creates a `PyContextGuard`
from the per-device ModelWeight's stream and allocator:

```cpp
.def("context",
     [](ft::TurboMind* model, int index) -> std::unique_ptr<PyContextGuard> {
         auto [stream, alloc] = model->weight_context(index);
         return std::make_unique<PyContextGuard>(std::move(stream), std::move(alloc));
     },
     "index"_a)
```

This requires TurboMind to expose the per-device stream and allocator (or the
ModelWeight's context components). Add a public method like
`TurboMind::weight_context(int index)` that returns the Stream and Allocator
for the specified device.

### 3. Distributor stores and uses context managers

The `Distributor` (`lmdeploy/turbomind/deploy/distributor.py`) is given context
managers at construction time. It wraps commit operations with them:

```python
class Distributor:
    def __init__(self, handles, contexts, tp=1, ranks=None):
        self._handles = handles
        self._contexts = contexts   # one ContextGuard per device
        self._tp = tp
        self._ranks = ranks

    def commit_linear(self, name, linear, split_side=None, model_dtype=None):
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0
                commit_linear(handle, linear, name,
                              split_side=split_side, split_num=tp,
                              rank=rank, model_dtype=model_dtype)

    def commit_tensor(self, name, tensor, split_side=None):
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0
                commit_tensor(handle, tensor, name,
                              split_side=split_side, split_num=tp,
                              rank=rank)

    def create_child(self, name, config, tp=None, ranks=None):
        new_tp = tp if tp is not None else self._tp
        new_ranks = ranks if ranks is not None else self._ranks
        children = []
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = new_ranks[i] if new_ranks and new_tp > 1 else 0
                child = handle.create_child(name, config.for_rank(rank).to_cpp())
                children.append(child)
        return Distributor(children, self._contexts, tp=new_tp, ranks=new_ranks)
```

The `contexts` list is created by the caller (the model loading code):

```python
handles = [turbo.root(i) for i in range(num_devices)]
contexts = [turbo.context(i) for i in range(num_devices)]
dist = Distributor(handles, contexts, tp=tp_size, ranks=...)
```

### 4. Remove `with_context` from bind.cpp

Delete the `with_context` lambda definition (lines 568-579) and remove the
wrapper from all 7 method bindings. Methods become direct bindings:

```cpp
py::class_<ft::core::Module, std::shared_ptr<ft::core::Module>>(m, "Module")
    .def("get",
         [](ft::core::Module& m, const std::string& segment) -> ft::core::Module* {
             return m.get(segment);
         },
         py::return_value_policy::reference,
         "segment"_a)
    .def("alloc",
         [](ft::core::Module& m, const std::string& param_name,
            ft::DataType dtype, int group_size) {
             return std::make_shared<Tensor>(m.alloc(param_name, ft::core::WeightSpec{dtype, group_size}));
         },
         "param_name"_a, "dtype"_a, "group_size"_a = 0)
    .def("create_param",
         [](ft::core::Module& m, const std::string& name,
            std::vector<size_t> shape, ft::DataType dtype, int group_size) {
             return std::make_shared<Tensor>(
                 m.create_param(name, shape, dtype, group_size));
         },
         "name"_a, "shape"_a, "dtype"_a, "group_size"_a = 0)
    .def("prepare",
         [](ft::core::Module& m) { m.prepare(); })
    .def("child",
         [](ft::core::Module& m, const std::string& name) -> ft::core::Module* {
             return m.child(name);
         },
         py::return_value_policy::reference, "name"_a)
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

### 5. Other call sites using ModelWeight::context()

These existing C++ call sites remain unchanged (they already manage context
externally):

- `turbomind.cc` ProcessWeights: `auto ctx_guard = weights_[index]->context();`
- `turbomind.cc` CreateEngine: `ContextGuard guard{ctx.core_stream, ...}`
- `engine/engine.cc` per-thread: `ContextGuard ctx{stream, ...}`
- `engine/model_executor.cc` per-thread: `ContextGuard ctx{stream, ...}`

### 6. LoadContext handling

`LoadContext` in `load_context.py` wraps a single C++ Module handle. It will
need access to a context manager for its operations. The simplest approach is
for `LoadContext` to also store and use a context manager:

```python
class LoadContext:
    def __init__(self, handle, context, tp_config, model_config=None):
        self._handle = handle
        self._context = context
        self._tp_config = tp_config
        self._model_config = model_config

    def load_linear(self, name, linear, tp_rule=None):
        with self._context:
            # existing load_linear logic

    def load_tensor(self, name, tensor, ...):
        with self._context:
            # existing load_tensor logic
```

## What stays unchanged

- **`ContextGuard`**: stays non-copyable, non-movable. No changes to `context.h`.
- **Module C++ code**: no provider_, no context() method, no new members.
- **ModelWeight C++ code**: keeps stream_, alloca_, context() as-is.
- **All derived weight classes**: LinearWeight, NormWeight, AttentionWeight, etc.
- **C++ inference path**: already manages context externally.

## Files changed

| File | Change |
|------|--------|
| `src/turbomind/python/bind.cpp` | Remove `with_context`, add `PyContextGuard`, add `TurboMind::context(index)` |
| `src/turbomind/turbomind.h` | Expose accessor for per-device stream/allocator |
| `lmdeploy/turbomind/deploy/distributor.py` | Store and use context managers |
| `lmdeploy/turbomind/deploy/load_context.py` | Store and use context manager |
| Model loading code (where Distributor is constructed) | Pass context managers to Distributor |

## Multi-GPU behavior

Each device has its own `ModelWeight` with its own stream/allocator.
`TurboMind::context(index)` creates a `PyContextGuard` for device `index`. The
Distributor stores one guard per device and uses the correct one for each
handle. No cross-device interference.

CUDA device binding (`CudaDeviceGuard`) remains separate and is managed
by `TurboMind::CreateWeights`, `ProcessWeights`, and `CreateEngine` as today.
