# Module API Polish Design

Date: 2026-04-07

## Problem

The C++ Module API and its Python callers have two ergonomic issues:

1. `create_child()` already returns a `Module*` in C++ and via pybind, but Python callers
   consistently ignore the return value and do a separate `get(name)` call to retrieve the child.
   This is verbose and makes the code harder to follow.

2. Creating a Parameter tensor from Python is ad-hoc. Callers must understand the low-level
   `alloc()` + `copy_from()` pattern, which is designed for pre-registered parameters in LinearWeight.
   For direct Parameter creation (embeddings, norms, conv1d, etc.), there is no first-class API.

## Design

### 1. Clean up `create_child()` calling convention

**No C++ or binding changes needed.** The API already returns the child module.

Change Python callers from:
```python
module.create_child(name, 'LinearWeight', config)
child = module.get(name)
```

To:
```python
child = module.create_child(name, 'LinearWeight', config)
```

### 2. Add `create_param()` to C++ Module

New C++ method:
```cpp
Tensor Module::create_param(const std::string& name,
                            const std::vector<size_t>& shape,
                            DataType dtype,
                            int group_size = 0);
```

This method:
- Creates a Parameter tensor with the given shape and dtype
- Registers it as a named parameter on the module (via `add_param()`)
- Returns the allocated Tensor for the caller to fill via `copy_from()`

Python binding:
```python
tensor = module.create_param(name, shape, dtype, group_size=0)
tensor.copy_from(data)
```

**Why two steps (create + copy) instead of one:** The caller needs to prepare data between
allocation and copy — TP sharding, dtype casting, and padding all happen in Python before
the copy. A single-call API would either skip these transforms or push C++ to understand
TP sharding, which is the wrong layer.

### 3. Migration strategy

Gradual, three steps:

**Step 1 — Add `create_param()` to C++ + pybind** (pure addition, nothing breaks)
- Add method to `Module` class in `module.h`/`module.cc`
- Add pybind binding in `bind.cpp`

**Step 2 — Migrate raw parameter callers to `create_param()`**
- For each call site using `alloc()` for non-LinearWeight parameters, switch to `create_param()`
- Each migration is independent and can be done per-file

**Step 3 — Clean up `create_child()` + `get()` patterns**
- Remove separate `get()` calls across all callers
- Each cleanup is a one-line change per call site

### What we do NOT change

- `alloc()` — still needed for LinearWeight's multi-kind allocation (weight/scales/zeros/bias)
- `copy_from()` — the actual data transfer mechanism
- `create_child()` API signature — callers just use the return value
- LoadContext, commit_linear, commit_tensor — structural changes out of scope
- Python loading code structure — no restructuring of module.py, text_model_loader.py, etc.

## Files affected

### C++ side
- `src/turbomind/core/module.h` — add `create_param()` declaration
- `src/turbomind/core/module.cc` — add `create_param()` implementation
- `src/turbomind/python/bind.cpp` — add pybind binding for `create_param()`

### Python side (caller cleanup)
- `lmdeploy/turbomind/deploy/load_context.py` — use `create_child()` return value
- `lmdeploy/turbomind/deploy/module.py` — use `create_child()` return value, migrate raw param sites to `create_param()`
- `lmdeploy/turbomind/deploy/text_model_loader.py` — use `create_child()` return value
- Any source model spec files that call `create_child()` + `get()`
