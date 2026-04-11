# alloc-to-param Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `alloc` from the Module interface, replace with a lightweight `Param` handle and LinearWeight-specific `set_weight_spec`.

**Architecture:** Introduce `Param` as a minimal Tensor-slot handle returned by `Module::param()`. Allocation is per-param: `param(name).alloc(shape, dtype)`. LinearWeight gets a non-virtual `set_weight_spec(dtype, group_size)` for quantization metadata. All `alloc` overrides on derived classes are removed.

**Tech Stack:** C++ (core module system), pybind11 (Python binding), Python (load pipeline).

---

### Task 1: Add `Param` class and update Module interface

**Files:**
- Modify: `src/turbomind/core/module.h`
- Modify: `src/turbomind/core/module.cc`

- [ ] **Step 1: Add `Param` class to `module.h`**

Add before the `Module` class definition, after `WeightSpec` (which will be removed):

```cpp
/// Lightweight handle to a Tensor slot within a Module.
/// Returned by Module::param(name). Used for per-param allocation.
class Param {
    Tensor* slot_;

public:
    explicit Param(Tensor* slot = nullptr) : slot_(slot) {}

    /// Allocate the tensor with explicit shape/dtype. Returns the tensor for data copy.
    Tensor alloc(const std::vector<size_t>& shape, DataType dtype)
    {
        TM_CHECK(slot_ != nullptr);
        auto layout = Layout{std::vector<ssize_t>(shape.begin(), shape.end())};
        *slot_ = Tensor{std::move(layout), dtype, kDEVICE};
        return *slot_;
    }

    /// Get current tensor (empty if not yet allocated).
    Tensor get() const { return slot_ ? *slot_ : Tensor{}; }

    explicit operator bool() const { return slot_ && static_cast<bool>(*slot_); }
};
```

- [ ] **Step 2: Update `TM_PARAM_CASE` macro in `module.h`**

Change from returning `Tensor*` to returning `Param`:

```cpp
/// Fragment for param() override body: matches name and returns Param handle.
#define TM_PARAM_CASE(name)          \
    if (name_str == #name) {         \
        return Param{&name};        \
    }
```

- [ ] **Step 3: Remove `WeightSpec`, `alloc`, and `create_param` from `module.h`**

Remove:
- The entire `WeightSpec` struct (lines 136-139)
- `virtual Tensor alloc(...)` declaration (line 200)
- `Tensor create_param(...)` declaration (lines 204-207)
- The doc comment referencing `alloc(param_name, spec)` (line 149)

Update `param()` return type from `Tensor*` to `Param` in:
- The virtual declaration inside Module class (line 190)
- The `TM_MODULE_DECLARE` macro (line 99)

- [ ] **Step 4: Remove `alloc` and `create_param` implementations from `module.cc`**

Delete:
- `Module::alloc(...)` method body (lines 58-65)
- `Module::create_param(...)` method body (lines 67-77)

- [ ] **Step 5: Build to verify**

```bash
cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80
```

Expect: compile errors in derived classes (alloc overrides reference removed base). This is expected — Task 2 fixes them.

---

### Task 2: Clean up derived weight classes

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`
- Modify: `src/turbomind/models/norm_weight.h`
- Modify: `src/turbomind/models/norm_weight.cc`
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/models/delta_net_weight.cc`

- [ ] **Step 1: Remove `alloc` override from NormWeight**

In `norm_weight.h`: remove `Tensor alloc(...) override;` declaration (line 30).

In `norm_weight.cc`: delete the entire `NormWeight::alloc(...)` method body (lines 43-57).

- [ ] **Step 2: Remove `alloc` override from AttentionWeight**

In `attention_weight.h`: remove `Tensor alloc(...) override;` declaration (line 22).

In `attention_weight.cc`: delete the entire `AttentionWeight::alloc(...)` method body (lines 32-41).

- [ ] **Step 3: Remove `alloc` override from MoeWeight**

In `moe_weight.h`: remove `Tensor alloc(...) override;` declaration (line 20).

In `moe_weight.cc`: delete the entire `MoeWeight::alloc(...)` method body (lines 39-48).

- [ ] **Step 4: Remove `alloc` override from DeltaNetWeight**

In `delta_net_weight.h`: remove `Tensor alloc(...) override;` declaration (line 23).

In `delta_net_weight.cc`: delete the entire `DeltaNetWeight::alloc(...)` method body (lines 28-47).

- [ ] **Step 5: Add `set_weight_spec` to LinearWeight, remove `alloc`/`do_allocate`/`allocate`**

In `linear_weight.h`:
- Remove `Tensor alloc(...) override;` (line 41)
- Remove `void allocate(...)` (line 39)
- Remove `void do_allocate(...)` (line 86)
- Add `void set_weight_spec(DataType weight_dtype, int group_size);` in the public section

In `linear_weight.cc`:
- Delete `do_allocate()` method body (lines 95-141)
- Delete `allocate()` method body (lines 147-150)
- Delete `alloc()` method body (lines 156-191)
- Add `set_weight_spec()` implementation:

```cpp
void LinearWeight::set_weight_spec(DataType weight_dtype, int group_size)
{
    // For dense float weights, coerce to model compute dtype
    if (weight_dtype != data_type && IsDenseFloatType(weight_dtype) && IsDenseFloatType(data_type)) {
        weight_dtype = data_type;
    }
    weight_format = weight_dtype;
    this->group_size = group_size;
    format_ = MakeLinearWeightFormat(data_type, weight_format, group_size);
    policy_ = ResolveLinearPolicy(format_, data_type, getSMVersion());
}
```

- [ ] **Step 6: Update LinearWeight::prepare() to set up k_desc**

At the top of `LinearWeight::prepare()`, before the existing format conversion logic, add the k_desc initialization that was previously in `do_allocate()`:

```cpp
void LinearWeight::prepare()
{
    if (!weight) {
        return;
    }

    auto stream = core::Context::stream().handle();

    // Set up GEMM descriptor (was previously in do_allocate)
    k_desc.type  = weight.dtype();
    k_desc.order = gemm::kRowMajor;
    k_desc.rows  = input_dim;
    k_desc.cols  = output_dim;
    k_desc.ld    = output_dim;

    // ... rest of existing prepare() unchanged ...
```

- [ ] **Step 7: Build to verify**

```bash
cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80
```

Expect: clean build (no errors). The Python binding will have errors referencing `alloc`/`create_param` — that's Task 3.

---

### Task 3: Update Python binding

**Files:**
- Modify: `src/turbomind/python/bind.cpp`

- [ ] **Step 1: Add `Param` binding**

Add after the `Tensor` binding section, before the Module binding:

```cpp
    // Param — lightweight handle to a Module parameter slot
    py::class_<ft::core::Param>(m, "Param")
        .def("alloc",
             [](ft::core::Param& p, std::vector<size_t> shape, ft::DataType dtype) {
                 return std::make_shared<Tensor>(p.alloc(shape, dtype));
             },
             "shape"_a,
             "dtype"_a)
        .def("get",
             [](ft::core::Param& p) { return std::make_shared<Tensor>(p.get()); })
        .def("__bool__", [](ft::core::Param& p) { return static_cast<bool>(p); });
```

- [ ] **Step 2: Add `param()` to Module binding, remove `alloc` and `create_param`**

In the Module binding block, remove:
- The `.def("alloc", ...)` block (lines 595-601)
- The `.def("create_param", ...)` block (lines 602-614)

Add:
```cpp
        .def("param",
             [](ft::core::Module& m, const std::string& name) -> ft::core::Param {
                 return m.param(name);
             },
             "name"_a)
```

- [ ] **Step 3: Add LinearWeight binding with `set_weight_spec`**

After the Module binding block, add:

```cpp
    // LinearWeight — specific interface for weight loading
    py::class_<turbomind::LinearWeight, ft::core::Module>(m, "LinearWeight")
        .def("set_weight_spec",
             [](turbomind::LinearWeight& lw, ft::DataType dtype, int group_size) {
                 lw.set_weight_spec(dtype, group_size);
             },
             "dtype"_a,
             "group_size"_a);
```

This requires including the LinearWeight header at the top of bind.cpp:

```cpp
#include "src/turbomind/models/linear_weight.h"
```

- [ ] **Step 4: Build to verify**

```bash
cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80
```

Expect: clean build.

---

### Task 4: Update Python load code

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1: Update `_commit_tensors` to use `param().alloc()`**

Replace the `handle.alloc(kind, cpp_dtype, group_size)` call (line 183) with `handle.param(kind).alloc(shard.shape, dst_dtype)`. The dtype coercion for dense-float weights moves here from C++:

```python
def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int,
                    model_dtype=None):
    """Commit tensor data from a ``Linear`` to a pre-created C++ LinearWeight handle.

    Handles packing, TP sharding, allocation, dtype casting, and padding.
    This is the shared tensor-commit loop used by both ``commit_linear`` and
    ``LoadContext.load_linear``.
    """
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    packer = linear.weight_format.packer if linear.weight_format else None

    def _kind_order(item):
        k, _ = item
        if k in ("weight", "qweight"):
            return (0, k)
        return (1, k)

    for kind, tensor in sorted(linear.tensors.items(), key=_kind_order):
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

        # Resolve allocation dtype: for dense float weight/qweight, use model
        # compute dtype to match the coercion previously done in C++ alloc.
        dst_dtype = cpp_dtype
        if kind in ("weight", "qweight") and model_dtype is not None:
            fmt = linear.weight_format
            if fmt is None or fmt.name == 'dense':
                dst_dtype = model_dtype if isinstance(model_dtype, int) else model_dtype.value

        dst = handle.param(kind).alloc(list(shard.shape), dst_dtype)
        if dst:
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
```

- [ ] **Step 2: Update `commit_linear` to call `set_weight_spec` and pass `model_dtype`**

After creating/finding the LinearWeight child, call `set_weight_spec`:

```python
    # ... existing code to create/find linear_mod ...

    # Set weight spec for quantization metadata
    import _turbomind as _tm
    resolved_cpp, resolved_gs = cpp_dtype, group_size
    if isinstance(linear_mod, _tm.LinearWeight):
        linear_mod.set_weight_spec(cpp_dtype if cpp_dtype else 0, group_size)

    _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                    split_side, split_num, rank, model_dtype=model_dtype)
```

Note: `cpp_dtype` from `_infer_cpp_linear_dtype` may be `None` for dense weights without a matching dtype. In that case, use the `compute_dtype` that was already resolved in `commit_linear`.

- [ ] **Step 3: Update `commit_tensor` to use `param().alloc()`**

Replace `module.alloc(name, cpp_dtype, 0)` (line 325):

```python
def commit_tensor(module, tensor: torch.Tensor | None, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0):
    if tensor is None:
        return

    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    if split_dim is not None and split_num > 1:
        split_size = tensor.shape[split_dim] // split_num
        shard = tensor.split(split_size, dim=split_dim)[rank]
    else:
        shard = tensor

    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()
    cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
    if cpp_dtype is None:
        return
    dst = module.param(name).alloc(list(shard.shape), cpp_dtype)
    if dst:
        shard = _cast_shard_for_tm(shard, dst)
        dst.copy_from(shard)
```

- [ ] **Step 4: Update `LoadContext.load_linear` similarly**

In `load_linear`, after creating the child and before calling `_commit_tensors`, call `set_weight_spec`:

```python
    def load_linear(self, name: str, linear: Linear,
                    tp_rule: str | None = None):
        with self._context or _noop():
            # ... existing code to compute dims, create child ...

            child_handle.set_weight_spec(cpp_dtype if cpp_dtype else 0, group_size)

            _commit_tensors(child_handle, linear, cpp_dtype, group_size,
                            tp_side, split_num, self.rank,
                            model_dtype=self.cpp_dtype)
```

---

### Task 5: Update GEMM test code

**Files:**
- Modify: `src/turbomind/kernels/gemm/test/testbed_v3.h`

- [ ] **Step 1: Replace `alloc` calls with `set_weight_spec` + `param().alloc()`**

Replace lines 299-306:

```cpp
        original.configure(input_dim, output_dim, data_type, false);
        original.set_weight_spec(data_type, group_size);
        original.param("weight").alloc({(size_t)input_dim, (size_t)output_dim}, data_type);
        rng_.NormalFloat(original.weight(), 1., .1);

        quant.configure(input_dim, output_dim, data_type, false);
        quant.set_weight_spec(weight_type, group_size);
        quant.param("weight").alloc({(size_t)input_dim, (size_t)output_dim}, weight_type);
        dequant.configure(input_dim, output_dim, data_type, false);
        dequant.set_weight_spec(data_type, group_size);
        dequant.param("weight").alloc({(size_t)input_dim, (size_t)output_dim}, data_type);
```

Note: The testbed's `DenseWeight` is `using DenseWeight = LinearWeight;` (line 24), so these methods are directly available.

- [ ] **Step 2: Build to verify**

```bash
cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80
```

---

### Task 6: End-to-end verification

- [ ] **Step 1: Check GPU availability**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Ensure at least one GPU has < 1000 MiB used.

- [ ] **Step 2: Run a model test**

Use the turbomind-tester agent or run `scripts/test_turbomind_model.py` with a locally cached model. Verify:
- Model loads without errors
- Response contains meaningful text (not gibberish)
- Response length is at least 128 tokens

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: replace Module::alloc with Param handle and LinearWeight::set_weight_spec

Remove alloc/create_param from Module interface. Introduce Param as a
lightweight Tensor-slot handle. LinearWeight gains set_weight_spec() for
quantization metadata. All derived class alloc overrides removed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
