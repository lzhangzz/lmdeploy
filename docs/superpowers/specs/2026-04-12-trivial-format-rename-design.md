# Rename "dense" Weight Format to "trivial"

## Problem

The TurboMind engine uses "dense" in two incompatible senses:

1. **Weight format** — meaning non-quantized / floating-point (FP16, BF16, FP32).
2. **Model architecture** — meaning non-MoE (dense FFN vs sparse MoE experts).

The weight-format usage contradicts the established ML meaning of "dense vs sparse"
(model architecture). This creates confusion when reading code that touches both
concepts, e.g. "dense FFN with dense weights" vs "MoE with dense weights."

## Solution

Rename every occurrence of "dense" that refers to the weight-format concept to
**"trivial"** — borrowed from mathematics where "trivial" denotes the simplest /
identity case. A trivial weight format applies no quantization.

The model-architecture sense (dense FFN vs MoE) is unaffected.

## Rename Mapping

### Python

| File | Current | Proposed |
|------|---------|----------|
| `kind_map.py:123` | `DENSE_SUFFIXES` | `TRIVIAL_SUFFIXES` |
| `kind_map.py:203` | `_normalize_dense()` | `_normalize_trivial()` |
| `kind_map.py:328` | `_accepts_dense()` | `_accepts_trivial()` |
| `kind_map.py:419` | `DENSE_FORMAT` (name=`"dense"`) | `TRIVIAL_FORMAT` (name=`"trivial"`) |
| `kind_map.py:275-276` | `None: _normalize_dense` maps | `None: _normalize_trivial` |
| `kind_map.py:497-498` | `None: DENSE_FORMAT` maps | `None: TRIVIAL_FORMAT` |
| `spec.py:6` | `from .kind_map import DENSE_FORMAT` | `from .kind_map import TRIVIAL_FORMAT` |
| `spec.py:334` | `weight_format=DENSE_FORMAT` | `weight_format=TRIVIAL_FORMAT` |
| `commit.py:121` | `fmt.name != 'dense'` | `fmt.name != 'trivial'` |
| `gpt_oss_spec.py:62` | `lin.weight_format.name == "dense"` | `lin.weight_format.name == "trivial"` |

Comments and docstrings in these files that use "dense" in the weight-format sense
are updated to "trivial." Comments using "dense" in the MoE-architecture sense are
left unchanged.

### C++

#### `IsDenseFloatType` → `IsTrivialFloatType`

The predicate `t == kFloat || t == kHalf || t == kBfloat16` is currently duplicated
in three locations. All are renamed to `IsTrivialFloatType` and deduplicated into a
single shared definition.

| File | Current | Proposed |
|------|---------|----------|
| `core/data_format.cc:21` | `static bool IsDenseFloatType(DataType t)` | Remove; use shared header |
| `core/data_format.cc:31` | `if (IsDenseFloatType(...))` | `if (IsTrivialFloatType(...))` |
| `utils/memory_utils.cu:115` | `bool IsDenseFloatType(DataType t)` | Remove; use shared header |
| `utils/memory_utils.cu:126` | `if (!IsDenseFloatType(...))` | `if (!IsTrivialFloatType(...))` |
| `models/linear_weight.cc:98` | `auto is_dense_float = [...]` | Use `IsTrivialFloatType` from header |

The shared definition goes in a header already included by consumers (e.g.
`memory_utils.h` or a new small header), as an inline function:

```cpp
inline bool IsTrivialFloatType(DataType t) {
    return t == kFloat || t == kHalf || t == kBfloat16;
}
```

#### Comments

| File | Line | Update |
|------|------|--------|
| `models/linear_weight.cc:69` | `dense (non-quantized)` | `trivial (non-quantized)` |
| `models/linear_weight.cc:97` | `dense float weights` | `trivial float weights` |
| `models/linear_weight.cc:134` | `dense weights` | `trivial weights` |
| `utils/memory_utils.h:32` | `dense float type` | `trivial float type` |
| `models/linear_weight.h:38` | `dense float weights` | `trivial float weights` |
| `kernels/gemm/convert_v3.cu:115` | `dense floating point` | `trivial floating point` |
| `models/moe_weight.cc:157` | `dense bf16/fp16` | `trivial bf16/fp16` |
| `models/moe_weight.cc:39` | `LlamaDenseWeight.cc` | Update historical reference |

#### Test names

| File | Current | Proposed |
|------|---------|----------|
| `core/test_data_format.cc:18` | `"DataFormat dense is not quantized"` | `"DataFormat trivial is not quantized"` |
| `core/test_data_format.cc:71` | `"DataFormat dense BF16"` | `"DataFormat trivial BF16"` |

### Legacy naming

#### `DenseWeight` type alias (test code)

`src/turbomind/kernels/gemm/test/testbed_v3.h:24` defines `using DenseWeight =
LinearWeight;`. This alias is used ~30 times in the test infrastructure. It is
misleading because `LinearWeight` handles all formats, not just trivial ones.

Replace all `DenseWeight` usage with `LinearWeight` directly and remove the alias.

#### `dense` parameter name in `LlamaLinear.cu`

The functions `GetOperandB`, `GetOperandA`, and `Forward` take `const
LinearWeight& dense` parameters. Rename to `weight` — the natural name for a
`LinearWeight` parameter.

#### Deleted `LlamaDenseWeight.cc` in CMake

`src/turbomind/models/llama/CMakeLists.txt:16` and
`src/turbomind/kernels/gemm/CMakeLists.txt:57` reference `LlamaDenseWeight.cc`,
which no longer exists on disk. Remove these dead references.

## What is NOT changed

- **MoE architecture sense** — `first_k_dense_replace`, "dense FFN", `dense_h_to_4h`,
  etc. in model specs and PyTorch model code remain unchanged.
- **`DenseWeight` in test function `LinkExperts`** — the parameter named `dense` in
  `testbed_v3.h:81`'s `LinkExperts` is renamed as part of the `DenseWeight` →
  `LinearWeight` alias removal.
- **External API** — `DENSE_FORMAT` is not part of any public API outside the
  `lmdeploy.turbomind.deploy` package, so no deprecation period is needed.
