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
with them. Module and ModelWeight C++ code stay unchanged.

## Design

### 1. ContextGuard move semantics (prerequisite)

`ContextGuard` (`src/turbomind/core/context.h`) has no move constructor. Returning
it from functions would use copy (causing double-pop on destruction). Add move
semantics:

```cpp
class ContextGuard {
public:
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

    ContextGuard(ContextGuard&& other) noexcept : n_(other.n_) { other.n_ = 0; }

    ContextGuard& operator=(ContextGuard&& other) noexcept
    {
        if (this != &other) {
            for (int i = 0; i < n_; ++i) {
                Context::pop();
            }
            n_ = other.n_;
            other.n_ = 0;
        }
        return *this;
    }

    // existing constructor and destructor unchanged
};
```

### 2. TurboMind binding exposes `context(index)`

Add a `context(index)` method to the TurboMind binding that returns a
`ContextGuard` for the specified device. The guard is wrapped as a Python
context manager (supports `with` statement).

In `bind.cpp`, bind `ContextGuard` as a Python context manager:

```cpp
py::class_<ft::core::ContextGuard>(m, "ContextGuard")
    .def("__enter__", [](ft::core::ContextGuard& g) -> ft::core::ContextGuard& { return g; })
    .def("__exit__", [](ft::core::ContextGuard& g, py::object, py::object, py::object) {});
```

Add `context(index)` to the TurboMind binding:

```cpp
.def("context",
     [](ft::TurboMind* model, int index) -> ft::core::ContextGuard {
         return model->weights_[index]->context();
     },
     "index"_a)
```

This requires a public accessor on TurboMind for the per-device
ModelWeight's context. Add a method like `model->weight_context(index)`
that returns `weights_[index]->context()`.

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

- **Module C++ code:** no provider_, no context() method, no new members
- **ModelWeight C++ code:** keeps stream_, alloca_, context() as-is
- **All derived weight classes:** LinearWeight, NormWeight, AttentionWeight, etc.
- **C++ inference path:** already manages context externally

## Files changed

| File | Change |
|------|--------|
| `src/turbomind/core/context.h` | Add move semantics to ContextGuard |
| `src/turbomind/python/bind.cpp` | Remove `with_context`, add ContextGuard binding, add `TurboMind::context(index)` |
| `src/turbomind/turbomind.h` | Expose accessor for weights_[index] context (if needed) |
| `lmdeploy/turbomind/deploy/distributor.py` | Store and use context managers |
| `lmdeploy/turbomind/deploy/load_context.py` | Store and use context manager |
| Model loading code (where Distributor is constructed) | Pass context managers to Distributor |

## Multi-GPU behavior

Each device has its own `ModelWeight` with its own stream/allocator.
`TurboMind::context(index)` returns the guard for device `index`. The
Distributor stores one guard per device and uses the correct one for each
handle. No cross-device interference.

CUDA device binding (`CudaDeviceGuard`) remains separate and is managed
by `TurboMind::CreateWeights`, `ProcessWeights`, and `CreateEngine` as today.
