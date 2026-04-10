# Module Implementation Overhaul

Date: 2026-04-10

## Problem

Module classes in TurboMind have four issues:

1. **Boilerplate**: Each module class repeats the same 3-5 virtual method bodies (add_child, child, param, for_each_child, for_each_param) in `.cc` files — mechanical X-macro expansions that add no information.
2. **Naming bug**: `TM_PARAM_MEMBER` uses names with trailing underscores (e.g., `sinks_`), so `TM_PARAM_CASE` stringifies to `"sinks_"` — but `alloc()` and consumers expect `"sinks"`. Param lookups silently fail.
3. **`mutable` overuse**: All `TM_PARAM_MEMBER` params are mutable because `param()` and `for_each_param()` are const methods. This undermines const-correctness for weight tensors that should be immutable after loading.
4. **Dead code**: `FfnWeight::set_fused_moe()` has zero callers; `set_fused_silu()` has a Python binding that can be replaced by config-driven initialization.

## Design

### 1. New macro system in module.h

**TM_PARAM_MEMBER** — drop `mutable`, drop underscore appending:

```cpp
#define TM_PARAM_MEMBER(name) Tensor name{};
```

Param X-macro lists drop trailing underscores:

```cpp
// Before
#define ATTENTION_WEIGHT_PARAMS(X) X(sinks_)
// After
#define ATTENTION_WEIGHT_PARAMS(X) X(sinks)
```

Member names become `sinks`, `conv1d`, `A_log`, `dt_bias`, `score_correction_bias` (no trailing `_`). Wire names (`"sinks"`, `"conv1d"`, etc.) match what `alloc()` and consumers already expect.

**TM_MODULE_DECLARE** — generates data members + virtual method declarations for the public section:

```cpp
#define TM_MODULE_DECLARE(Class, ChildrenX, ParamsX)  \
    ChildrenX(TM_CHILD_MEMBER)                         \
    ParamsX(TM_PARAM_MEMBER)                           \
    core::Module* add_child(std::string, std::unique_ptr<core::Module>) override; \
    core::Module* child(const std::string&) const override; \
    Tensor* param(const std::string&) override; \
    void for_each_child(ForChildFn) const override; \
    void for_each_param(ForParamFn) override;
```

Data members are public — execution layers already access them directly, and the `child()`/`param()` virtual dispatch is for the module framework, not encapsulation.

Each class always defines its own X-macro lists. Empty lists are per-class (e.g., `#define FFN_WEIGHT_PARAMS(X)`), not shared.

**TM_MODULE_METHODS** — generates all method definitions for `.cc` files:

```cpp
#define TM_MODULE_METHODS(Class, ChildrenX, ParamsX)  \
    core::Module* Class::add_child(...) { ChildrenX(TM_ADD_CHILD_CASE) return nullptr; } \
    core::Module* Class::child(...) const { ChildrenX(TM_CHILD_CASE) return nullptr; } \
    Tensor* Class::param(...) { ParamsX(TM_PARAM_CASE) return nullptr; } \
    void Class::for_each_child(...) const { ChildrenX(TM_VISIT_CHILD) } \
    void Class::for_each_param(...) { ParamsX(TM_VISIT_PARAM) }
```

Existing CASE macros (`TM_ADD_CHILD_CASE`, `TM_CHILD_CASE`, `TM_PARAM_CASE`, `TM_VISIT_CHILD`, `TM_VISIT_PARAM`) remain unchanged as internal helpers.

**Header usage:**

```cpp
class AttentionWeight : public core::Module {
public:
    AttentionWeight(Config cfg);
    // alloc(), etc.
    TM_MODULE_DECLARE(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)
};
```

**Source usage:**

```cpp
TM_MODULE_METHODS(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)
```

### 2. Mutable fix

- `param()` and `for_each_param()` become non-const in the base `Module` class.
- Call sites that call these through `const Module&` need updating to non-const references.

### 3. Dead code & cleanup

- Remove `FfnWeight::set_fused_moe()` — zero callers.
- Keep `FfnWeight::set_fused_silu()` for C++ MoE block view propagation (moe_weight.cc).
- Remove the Python pybind binding for `set_fused_silu`.
- Fix `load_context.py` to pass `fuse_silu` through config at creation time instead of calling `set_fused_silu()` post-hoc.
- Remove per-class accessor methods now redundant with public members (e.g., `MoeWeight::score_correction_bias()`).

### 4. Call site updates

All direct member accesses that used trailing-underscore names update to the new names:

| Old | New |
|-----|-----|
| `weights.sinks_` | `weights.sinks` |
| `weights.A_log_` | `weights.A_log` |
| `weights.dt_bias_` | `weights.dt_bias` |
| `weights.conv1d_` | `weights.conv1d` |
| `weights.score_correction_bias_` | `weights.score_correction_bias` |

## Affected classes

| Class | File | Children | Params |
|-------|------|----------|--------|
| LinearWeight | linear_weight.h/.cc | 0 | 4 |
| NormWeight | norm_weight.h/.cc | 0 | 1 |
| AttentionWeight | attention_weight.h/.cc | 10 | 1 |
| FfnWeight | ffn_weight.h/.cc | 4 | 0 |
| MoeWeight | moe_weight.h/.cc | 3 | 1 |
| DeltaNetWeight | delta_net_weight.h/.cc | 3 | 3 |
| DecoderLayerWeight | decoder_layer_weight.h/.cc | 6 | 0 |
| ModelWeight | model_weight.h/.cc | 4 | 0 |

## Implementation order

1. Redesign macros in module.h (TM_MODULE_DECLARE, TM_MODULE_METHODS, fix TM_PARAM_MEMBER, drop mutable)
2. Make param()/for_each_param() non-const in base Module class
3. Update all 8 module classes (rename param X-macro entries to drop `_`, use TM_MODULE_DECLARE in .h, TM_MODULE_METHODS in .cc)
4. Update all call sites (member name changes, const fixups)
5. Remove dead code (set_fused_moe, pybind for set_fused_silu, redundant accessors)
6. Fix load_context.py to pass fuse_silu through config
