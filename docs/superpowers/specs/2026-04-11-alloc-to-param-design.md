# alloc-to-param: Remove `alloc` from Module, Introduce `Param` Handle

Date: 2026-04-11

## Problem

`Module::alloc(param_name, WeightSpec)` is a virtual method on the generic Module
interface that serves only the weight-loading pipeline. Problems:

1. **Wrong abstraction level** -- `WeightSpec` (dtype + group_size) is quantization
   metadata meaningful only for `LinearWeight`. Other classes ignore it.
2. **Conflates allocation and retrieval** -- `alloc` creates tensors AND returns them
   for data copy, mixing two responsibilities.
3. **LinearWeight cascading** -- calling `alloc("weight")` also silently allocates
   scales, zeros, bias. Hidden side effects make the flow hard to reason about.
4. **Boilerplate** -- every new weight class needs a bespoke `alloc` override with
   string-matching logic, even though Python already knows the shapes from the
   checkpoint.

## Design

### Param -- lightweight Tensor slot handle

```cpp
class Param {
    Tensor* slot_;
public:
    explicit Param(Tensor* slot = nullptr) : slot_(slot) {}

    Tensor alloc(const std::vector<size_t>& shape, DataType dtype) {
        *slot_ = Tensor{Layout{shape}, dtype, kDEVICE};
        return *slot_;
    }

    Tensor get() const { return slot_ ? *slot_ : Tensor{}; }
    explicit operator bool() const { return slot_ && static_cast<bool>(*slot_); }
};
```

Param is stack-allocated, copyable, returned by value. No owner pointer, no name
string -- it is purely a handle to a Tensor slot.

### Module changes

**Remove from Module:**
- `virtual Tensor alloc(const std::string&, const WeightSpec&)` -- deleted
- `Tensor create_param(const std::string&, vector<size_t>, DataType, int)` -- deleted
- `struct WeightSpec` -- deleted from module.h

**Change on Module:**
- `param(name)` return type changes from `Tensor*` to `Param`

**X-macro update:**
```cpp
#define TM_PARAM_CASE(name)       \
    if (name_str == #name) {      \
        return Param{&name};     \
    }
```

`for_each_param` and `verify` are unchanged -- they work with `Tensor&` directly
and don't go through `param()`.

### LinearWeight-specific interface

LinearWeight gains a non-virtual method for setting up quantization metadata:

```cpp
class LinearWeight: public core::Module {
public:
    // Called before individual param allocations. Sets weight_format, group_size,
    // format_, policy_ -- all the metadata that prepare() and format conversion need.
    void set_weight_spec(DataType weight_dtype, int group_size);
    // ...
};
```

This is NOT on Module. It is exposed in the Python binding only on LinearWeight:
```cpp
py::class_<turbomind::LinearWeight, ft::core::Module>(m, "LinearWeight")
    .def("set_weight_spec", &turbomind::LinearWeight::set_weight_spec,
         "dtype"_a, "group_size"_a);
```

LinearWeight's `prepare()` uses the stored metadata (set by `set_weight_spec`) for
format conversion. The `k_desc`/`q_desc` setup moves from `do_allocate()` to
`prepare()`.

Both `do_allocate()` and `allocate()` are removed. All allocation goes through
`set_weight_spec()` (metadata) + per-param `param().alloc()` (tensors).

### Derived class cleanup

All `alloc` overrides are removed:

| Class           | What happens instead                                          |
|-----------------|---------------------------------------------------------------|
| NormWeight      | Python calls `param("weight").alloc(shape, dtype)` directly  |
| AttentionWeight | `param("sinks").alloc(shape, dtype)` -- no override needed   |
| DeltaNetWeight  | `param("A_log").alloc(...)`, etc. -- no override needed      |
| MoeWeight       | `param("score_correction_bias").alloc(...)` -- no override   |
| LinearWeight    | `set_weight_spec()` + per-param `param().alloc()`            |

These classes no longer need any allocation-related virtual overrides.

### Python flow

**LinearWeight (`_commit_tensors` / `commit_linear`):**
```python
linear_mod.set_weight_spec(cpp_dtype, group_size)
for kind, tensor in sorted(...):
    shard = ...  # TP split, pack, cast
    dst = linear_mod.param(kind).alloc(shard.shape, dst_dtype)
    dst.copy_from(shard)
```

**Generic modules (`commit_tensor`):**
```python
dst = module.param(name).alloc(shard.shape, cpp_dtype)
dst.copy_from(shard)
```

### Dtype coercion

Currently `LinearWeight::alloc` coerces dense-float checkpoint dtypes to the model's
compute dtype. In the new design, Python handles this -- it already has `model_dtype`
and `_cast_shard_for_tm`. Python passes the model compute dtype for dense-float
weights, matching the current behavior.

### Test code

`testbed_v3.h` currently calls `alloc("weight", WeightSpec{...})`. Changes to:
```cpp
original.configure(input_dim, output_dim, data_type, false);
original.set_weight_spec(data_type, group_size);
original.param("weight").alloc({input_dim, output_dim}, data_type);
```

## Files changed

- `src/turbomind/core/module.h` -- remove alloc, WeightSpec, create_param; add Param
- `src/turbomind/core/module.cc` -- remove alloc, create_param implementations
- `src/turbomind/models/linear_weight.h` -- add set_weight_spec, remove alloc override
- `src/turbomind/models/linear_weight.cc` -- remove alloc/do_allocate, update prepare
- `src/turbomind/models/attention_weight.{h,cc}` -- remove alloc override
- `src/turbomind/models/norm_weight.{h,cc}` -- remove alloc override
- `src/turbomind/models/moe_weight.{h,cc}` -- remove alloc override
- `src/turbomind/models/delta_net_weight.{h,cc}` -- remove alloc override
- `src/turbomind/python/bind.cpp` -- expose Param, LinearWeight; remove alloc binding
- `lmdeploy/turbomind/deploy/load_context.py` -- use param().alloc() / set_weight_spec
- `src/turbomind/kernels/gemm/test/testbed_v3.h` -- use new API
