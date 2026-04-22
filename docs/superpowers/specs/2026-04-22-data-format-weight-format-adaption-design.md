# `DataFormat` / `WeightFormat` adaption: single-phase linear construction

Date: 2026-04-22

Scope: TurboMind's linear-weight pipeline, both Python and C++.

- `src/turbomind/core/data_format.{h,cc}` + `test_data_format.cc`
- `src/turbomind/models/linear_weight.{h,cc}`, `moe_weight.cc`
- `src/turbomind/python/bind.cpp`
- `lmdeploy/turbomind/deploy/kind_map.py`, `linear.py`, `converter.py`, `spec.py`
- `lmdeploy/turbomind/deploy/builder/_base.py`, `attention.py`, `deltanet.py`, `mla.py`
- `lmdeploy/turbomind/deploy/source_model/utils.py` + spec subclasses (qwen3, qwen3_5, gpt_oss, glm4_moe_lite)
- `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`

Out of scope: `src/turbomind/kernels/gemm/test/testbed_v3.h` (currently stale: references a deleted field `LinearWeight::resolved_`; treated as if it does not exist).

## Motivation

The `LinearWeight` pipeline today is four phases bolted together:

1. **C++ construction.** `handle.create_child(name, LinearConfig{input_dim, output_dim, data_type, has_bias})` — at this point the module is incompletely specified for any quantized format. `weight_format`, `group_size`, and the derived `format: DataFormat` are all defaulted.
2. **Python→C++ bolt-on.** `linear_mod.set_weight_spec(cpp_dtype, block_in)` recomputes `format = MakeLinearWeightFormat(...)` and `policy = ResolveLinearPolicy(...)`.
3. **Per-param allocation.** Python calls `handle.param(name).alloc(alloc_shape, alloc_dtype)` per kind, sometimes relabeling shapes/dtypes (quantized weight stored as int32 but logically UINT4).
4. **C++ `prepare()`.** Reads `weight_format`, `input_dtype()`, `data_type`, `is_grouped_`, dispatches.

This design has three symptoms:

- **`LinearConfig` is too narrow.** It looks like it fully configures a linear but doesn't carry weight-storage format. That forces phase 2 (`set_weight_spec`) to exist as a post-hoc mutation.
- **Parallel information channels across the seam.** `(compute_dtype, weight_dtype, block_sizes)` flows through three separate mechanisms (`LinearConfig.data_type`, `set_weight_spec`, `param().alloc(shape, dtype)`) that C++ re-stitches into a single `DataFormat`. The single-arg `MakeLinearWeightFormat(data_type, weight_format, group_size)` relies on overloaded magic per weight_format to fill in `block_sizes`.
- **`group_size` is a redundant thread.** After 2026-04-21-commit-simplification, `WeightFormat.block_in` / `block_out` are authoritative — but every `Spec` still carries `self._group_size`, every subclass `__init__` forwards a `group_size` kwarg, and `build_linear` still does per-call `dataclasses.replace` to resolve sentinels.

The Python-side carries two vestigial artifacts: `Linear.data_format` is a field that is written (as `None`) but never read live, and `WeightFormat.to_data_format()` is a broken dead method (passes weight_dtype as compute_dtype).

This spec collapses the lifecycle into **single-phase construction from a `LinearConfig` that carries the full weight `DataFormat`**. The three parallel channels become one. `set_weight_spec`, `configure`, `LinearPolicy`, and `MakeLinearWeightFormat`'s overloaded single-arg signature are all removed. `Linear.data_format` is revived as the authoritative carrier of the C++ `DataFormat` across the seam.

The user's original framing was "`group_size` in Python is superseded by `WeightFormat.block_in/out`". That purge is Section 5; it's enabled by the structural work in Sections 2–4.

## 1. Architecture

Three `DataFormat`s per linear (weight/input/output) fully describe the GEMM. One bridge across the seam: `WeightFormat.make_data_format(data_type) -> _tm.DataFormat`.

```
┌─────── Python ───────┐              ┌─────── C++ ───────┐
checkpoint
    │
    ▼
build_linear(data_type):
    fmt = WeightFormat (descriptor)
    data_format = fmt.make_data_format(data_type)   ── bridge
    │                                       ▼
    ▼                                  _tm.DataFormat
Linear {
    tensors,
    weight_format: WeightFormat,
    data_format: _tm.DataFormat
}
    │
    ▼
_commit_linear():
    lin_cfg = LinearConfig {
        input_dim, output_dim,
        data_type,           ─── model activation dtype (scalar)
        format = linear.data_format,
        has_bias,
    }
    handle.create_child(name, lin_cfg)  ───►  LinearWeight ctor:
                                                 weight_format = cfg.format
                                                 (input_format,
                                                  output_format)
                                                  = DeriveActivationFormats(
                                                      weight_format, data_type, SM)

    for kind, tensor in linear.tensors:
        _copy_shard_to_param(...)     ──►  byte-copy into param slots
```

### Ownership

- **Checkpoint-side** (suffix map, normalizer, packer, accepts, zeros_factory, dequant) → Python `WeightFormat` singletons. Unchanged.
- **Runtime block sizes per linear** → `converter.py` resolves once from `quant_config + user input`, stores on the resolved `WeightFormat` instance. No per-linear `dataclasses.replace`.
- **GEMM-support authority** → C++ `ResolveLinearWeightFormat` validates `(weight_dtype, block_in, block_out)` combinations. Today: hardcoded per-weight_dtype table. Future: kernel-registry lookup. Python never duplicates this knowledge.
- **Compute-side format** (three `DataFormat`s, allocations, GEMM descriptors, format conversion) → C++ `LinearWeight` / `LinearConfig` / `DeriveActivationFormats` / `prepare()`.

### Naming clarification

`data_type` on `LinearConfig` / `LinearWeight` is the **model activation dtype** (what activations flow as between layers). It is not a "compute dtype"; each tensor has its own dtype in its own `DataFormat`. `data_type` survives as a scalar on `LinearConfig` because — specifically for FP8 / FP4 weights — `weight_format` alone does not uniquely determine the output activation dtype.

### Block-sizes ordering

`DataFormat.block_sizes[i]` is the block size along the described tensor's `shape[i]`. For a weight tensor in TM `[in, out]` layout, that means:

- `block_sizes[0]` = `block_in` (K-axis)
- `block_sizes[1]` = `block_out` (N-axis)

Today's C++ readers treat `block_sizes[1]` as the K-axis group size (kernel-native `[N, K]` order). Under this refactor, readers flip to `block_sizes[0]` for K-axis grouping. Python's `WeightFormat.block_in` / `block_out` maps directly to `block_sizes[0]` / `block_sizes[1]`. One convention, everywhere.

## 2. `DataFormat` pybind + factory

### Pybind bindings

`DataFormat` sub-fields stay `def_readonly` — Python never mutates a `DataFormat` in place. Python constructs whole ones via the factory and assigns them to `cfg.format`. Sufficient because `def_readwrite` on `LinearConfig.format` (via the X-macro `bind_config` helper) accepts an already-constructed `DataFormat` by copy.

No change to the `DataFormat` / `QuantParamDesc` binding blocks.

### Factory: `ResolveLinearWeightFormat`

Renamed from `MakeLinearWeightFormat`; new signature with explicit block sizes:

```cpp
/// Construct the DataFormat for a linear weight tensor in TM [in, out] layout.
/// block_sizes stored in tensor-shape order: {block_in, block_out}.
/// Scales / zeros dtypes are derived from (data_type, weight_dtype) per the
/// format's GEMM convention. Validates that the combination is supported.
DataFormat ResolveLinearWeightFormat(DataType data_type,
                                     DataType weight_dtype,
                                     int      block_in,
                                     int      block_out);
```

Rules (today's behavior, restated for the new signature + tensor-shape ordering):

| weight_dtype | validation | block_sizes | scales.dtype | zeros.dtype |
|---|---|---|---|---|
| trivial float (BF16/FP16/FP32) | `block_in == 1 && block_out == 1` | `{1, 1}` | `kNull` | `kNull` |
| `kFloat8_e4m3` | `block_in == 128 && block_out == 128` | `{128, 128}` | `kFloat` | `kNull` |
| `kFloat4_e2m1` | `block_in > 0 && block_out == 1` | `{block_in, 1}` | `kUint8` | `kNull` |
| `kUint4` / `kUint8` | `0 < block_in ≤ 256 && block_out == 1` | `{block_in, 1}` | `data_type` | `data_type` |

`TM_CHECK` on violations. Future: the hardcoded `if`-chain becomes a kernel-registry query; signature unchanged.

Pybind binding:

```cpp
m.def("ResolveLinearWeightFormat", &ResolveLinearWeightFormat,
      py::arg("data_type"),
      py::arg("weight_dtype"),
      py::arg("block_in"),
      py::arg("block_out"));
```

The old `MakeLinearWeightFormat` binding is removed.

### Python bridge: `WeightFormat.make_data_format`

Replaces the dead `to_data_format(cpp_dtype, group_size=0)`:

```python
def make_data_format(self, data_type: _tm.DataType) -> _tm.DataFormat:
    """Construct the C++ DataFormat describing this checkpoint format's
    weight storage, resolved for the given model activation dtype.

    Delegates format-validity rules to the C++ factory
    (``ResolveLinearWeightFormat``), which is the source of truth on which
    (weight_dtype, block_sizes) combinations are supported on this GPU.
    """
    if self.cpp_dtype_name is None:
        # trivial: weight dtype equals activation dtype, no blocking
        return _tm.ResolveLinearWeightFormat(data_type, data_type, 1, 1)
    weight_dtype = getattr(_tm.DataType, self.cpp_dtype_name)
    return _tm.ResolveLinearWeightFormat(
        data_type,
        weight_dtype,
        self.block_in  or 1,   # None → 1 (no blocking on this axis)
        self.block_out or 1,
    )
```

Precondition: `block_in` / `block_out` are fully resolved (no `0` sentinel) by the time this is called. Violation surfaces as a C++ factory validation error.

No new fields on `WeightFormat` singletons — existing `cpp_dtype_name` / `block_in` / `block_out` are sufficient.

## 3. C++ `LinearConfig` and `LinearWeight`

### `LinearConfig`

```cpp
struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

    #define LINEAR_FIELDS(X) \
        X(int,        input_dim)  \
        X(int,        output_dim) \
        X(DataType,   data_type)  \
        X(DataFormat, format)     \
        X(bool,       has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

    #undef LINEAR_FIELDS
};
```

Added: `format: DataFormat`. The X-macro `bind_config` helper picks it up automatically — no change to `bind.cpp`'s `bind_config<LinearConfig>` call.

### `LinearWeight`

```cpp
class LinearWeight: public core::Module {
public:
    const char* type() const override { return "LinearWeight"; }

    LinearWeight() = default;                     // for default-constructed block views
    LinearWeight(const core::LinearConfig& cfg);  // single-phase construction

    void prepare() override;
    void copy_metadata_to(LinearWeight& dst) const;

    void set_grouped(bool grouped) { is_grouped_ = grouped; }

    explicit operator bool() const noexcept { return static_cast<bool>(weight); }

    // --- three DataFormats fully describe the GEMM ---
    DataFormat weight_format{};  // from cfg.format
    DataFormat input_format{};   // derived in ctor
    DataFormat output_format{};  // derived in ctor

    DataType input_dtype()  const { return input_format.dtype;  }
    DataType output_dtype() const { return output_format.dtype; }

    // --- dimensions + model activation dtype ---
    int      input_dim  = 0;
    int      output_dim = 0;
    DataType data_type{};   // model activation dtype, copied from cfg.data_type

    // --- GEMM knobs ---
    Epilogue     epilogue{};
    MatrixLayout k_desc{};
    MatrixLayout q_desc{};

#define LINEAR_WEIGHT_CHILDREN(X)
#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight) \
    X(bias)   \
    X(scales) \
    X(zeros)

    TM_MODULE_DECLARE(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

private:
    bool has_bias_   = false;
    bool is_grouped_ = false;
};
```

**Deleted fields** (relative to today):
- `int group_size` — redundant with `weight_format.block_sizes[0]`.
- scalar `DataType weight_format` — renamed/subsumed by `DataFormat weight_format`.
- `LinearPolicy policy` — replaced by `input_format` / `output_format`.
- `struct LinearPolicy` and `ResolveLinearPolicy` — deleted entirely.

**Deleted methods**:
- `configure(int, int, DataType, bool)` — body moves into ctor.
- `set_weight_spec(DataType, int)` — the format arrives via `cfg.format`.
- `preprocess()` — was a no-op.

### Ctor

```cpp
LinearWeight::LinearWeight(const core::LinearConfig& cfg)
    : input_dim(cfg.input_dim)
    , output_dim(cfg.output_dim)
    , data_type(cfg.data_type)
    , weight_format(cfg.format)
    , has_bias_(cfg.has_bias)
{
    std::tie(input_format, output_format) =
        DeriveActivationFormats(weight_format, data_type, getSMVersion());
}
```

No allocation in the ctor — param slots are still allocated by the Python-side `_copy_shard_to_param` on first commit. (Pre-allocation is a separate refactor.)

### `DeriveActivationFormats` replaces `ResolveLinearPolicy`

```cpp
/// Derive (input_format, output_format) for a GEMM whose weight uses
/// `weight_format`, given the model's activation dtype and hardware SM.
std::pair<DataFormat, DataFormat>
DeriveActivationFormats(const DataFormat& weight_format,
                        DataType          data_type,
                        int               sm);
```

Behavior (identical to today's `ResolveLinearPolicy`, reshaped into two DataFormats, with `block_sizes` in tensor-shape order):

| Weight format | input_format | output_format |
|---|---|---|
| trivial (`{bf16/fp16/fp32, {1,1}}`) | `{data_type, {1,1}}` | `{data_type, {1,1}}` |
| FP8 on SM90 | `{fp8_e4m3, {gs, 1}, scales=f32}` where `gs = weight_format.block_sizes[0]` | `{data_type, {1,1}}` |
| FP8 elsewhere | `{data_type, {1,1}}` | `{data_type, {1,1}}` |
| U4 / U8 / FP4 | `{data_type, {1,1}}` | `{data_type, {1,1}}` |

The empty-format case (`weight_format.dtype == kNull`) — used for `tok_embeddings` / `lm_head` loaded via `LinearBuilder.set_weight()` — maps to trivial `input_format` / `output_format` via `data_type`. The existing short-circuit in `prepare()` remains on this path.

### `prepare()` adjustments

- `weight_format.dtype` replaces the scalar `weight_format`.
- K-axis group size reads `weight_format.block_sizes[0]` (was `block_sizes[1]` via the old `[N, K]` kernel-native convention).
- `input_dtype()` reads `input_format.dtype`; `output_dtype()` reads `output_format.dtype`. Same values as today.
- `is_grouped_` unchanged (set via existing `set_grouped()` for MoE).

### `copy_metadata_to`

Copies the three DataFormats + remaining scalars:

```cpp
void LinearWeight::copy_metadata_to(LinearWeight& dst) const {
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.data_type     = data_type;
    dst.weight_format = weight_format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.is_grouped_   = is_grouped_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}
```

### `moe_weight.cc::LinkExperts`

Today reads `d.weight_format == kFloat8_e4m3` (scalar). Updated to `d.weight_format.dtype == kFloat8_e4m3`. Same logic, accessing the renamed field through its new struct wrapper.

### pybind

- `LinearConfig.format` gets `def_readwrite` automatically via the X-macro.
- The `LinearWeight::set_weight_spec` binding (lines 669-675 of `bind.cpp`) is removed.

## 4. Python data flow

### `Linear` dataclass tightens

```python
@dataclass
class Linear:
    tensors: dict[str, Tensor]
    weight_format: WeightFormat        # required, non-optional
    data_format:   _tm.DataFormat      # required, non-optional
```

`| None` and defaults go away. This catches the former `wfmt=None`/`dfmt=None` fallback paths at construction time.

### `build_linear` populates `Linear.data_format`

New required kwargs `data_type` and `weight_format` (the pre-resolved active format — see §5):

```python
def build_linear(params, prefix, *, data_type: _tm.DataType,
                 weight_format: WeightFormat,
                 index: int | None = None) -> Linear | None:
    available = {s: params[prefix + s] for s in ALL_SUFFIXES if (prefix + s) in params}
    if index is not None:
        available = {s: t[index] for s, t in available.items()}

    # Active format first; fall back to TRIVIAL only (other quantized formats
    # cannot match a checkpoint of a different format).
    fmt = (weight_format if weight_format.accepts(available)
           else (TRIVIAL_FORMAT if TRIVIAL_FORMAT.accepts(available)
                 else None))
    if fmt is None:
        return None

    tensors = {kind: fmt.normalizer(available[s], kind)
               for s, kind in fmt.suffix_map.items() if s in available}
    if not tensors:
        return None
    fmt.complete_tensors(tensors)

    return Linear(tensors=tensors,
                  weight_format=fmt,
                  data_format=fmt.make_data_format(data_type))
```

Deleted:
- `block_in` / `block_out` kwargs.
- The in-body `dataclasses.replace` sentinel resolution.
- Runtime iteration over the full `FORMAT_PRIORITY` list.

`FORMAT_PRIORITY` is no longer iterated at runtime in `build_linear` (the loop collapses to "active format → trivial fallback"). It survives as the source list for `ALL_SUFFIXES` (the checkpoint-suffix union used at the top of `build_linear`) — same role as today, just no longer driving dispatch.

### `_commit_linear` becomes a pass-through

```python
def _commit_linear(self, name, linear, split_side=None, model_dtype=None):
    self._ensure_handles()
    w = linear.tensors.get('weight')
    if w is None:
        return

    assert linear.data_format is not None, (
        f"{name}: Linear.data_format must be populated by build_linear or "
        f"by a fusion helper with explicit data_type.")

    tp = self._tp if split_side else 1
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    in_dim, out_dim = w.shape[0], w.shape[-1]
    if split_side == SplitSide.OUTPUT:   out_dim //= tp
    elif split_side == SplitSide.INPUT:  in_dim  //= tp

    data_type = model_dtype or _infer_compute_dtype(linear)

    lin_cfg = _tm.LinearConfig()
    lin_cfg.input_dim  = in_dim
    lin_cfg.output_dim = out_dim
    lin_cfg.data_type  = data_type or _tm.DataType.TYPE_INVALID
    lin_cfg.format     = linear.data_format
    lin_cfg.has_bias   = 'bias' in linear.tensors

    fmt = linear.weight_format
    packer = fmt.packer
    tensors = {k: packer(t, k) for k, t in linear.tensors.items()} if packer else linear.tensors
    is_quantized = linear.data_format.is_quantized()
    weight_cpp_dtype = linear.data_format.dtype

    # ... TP validation + per-GPU commit loop unchanged ...

    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            rank = self._rank_for(i) if tp > 1 else 0
            linear_mod = handle.child(name) or handle.create_child(name, lin_cfg)
            # NO set_weight_spec call.

            for kind, tensor in tensors.items():
                shard = _shard(tensor, kind_split_dims[kind], tp, rank)
                if kind == 'weight' and is_quantized:
                    alloc_shape, alloc_dtype = [in_dim, out_dim], weight_cpp_dtype
                elif kind == 'weight' and model_dtype is not None:
                    alloc_shape, alloc_dtype = None, model_dtype
                else:
                    alloc_shape, alloc_dtype = None, None
                _copy_shard_to_param(linear_mod, kind, shard,
                                     alloc_shape=alloc_shape,
                                     alloc_dtype=alloc_dtype)
```

Deleted:
- `cpp_dtype = _infer_cpp_linear_dtype(linear)` — reads `linear.data_format.dtype` directly.
- `block_in = (fmt.block_in or 0) if fmt is not None else 0` line.
- `linear_mod.set_weight_spec(cpp_dtype, block_in)` call.
- The `_infer_cpp_linear_dtype` function itself (no callers remain).

### `_dequant_linear` with explicit `data_type`

```python
def _dequant_linear(linear: Linear, *, data_type: _tm.DataType) -> Linear:
    if linear.weight_format.dequant is None:
        return linear
    new_tensors = linear.weight_format.dequant(linear.tensors, data_type=data_type)
    return Linear(
        tensors=new_tensors,
        weight_format=TRIVIAL_FORMAT,
        data_format=TRIVIAL_FORMAT.make_data_format(data_type),
    )
```

`data_type` is keyword-only, required. The former `if fmt is None` branch disappears (see invariants below).

### Propagating `data_type` to callers

| Caller | Signature change | Source of `data_type` |
|---|---|---|
| `attention.py::dequant_mixed(*linears)` | `dequant_mixed(*linears, data_type)` | `self.config.data_type` at call site |
| `_base.py::_ensure_compatible_formats(linears)` | `_ensure_compatible_formats(linears, *, data_type)` | `self.config.data_type` at call site |
| `deltanet.py` call site | pass `data_type=self.config.data_type` | available on Builder |
| `source_model/utils.py::_fold_head_dim` | `_fold_head_dim(lin, …, *, data_type)` | `self._cpp_dtype()` at spec call site |

`source_model/utils.py::_dequant_linear` is an exact duplicate of `_base.py::_dequant_linear`. Deleted; imports from `_base`.

### `WeightFormat.dequant` uniform signature

All dequant callables take `data_type` uniformly, even where unused:

```python
@dataclass(frozen=True)
class WeightFormat:
    ...
    dequant: Callable[[dict[str, Tensor], _tm.DataType],
                      dict[str, Tensor]] | None
```

`_dequant_awq` accepts an unused `data_type` param (AWQ's `dequantize_gemm` infers dtype from scales). `_dequant_fp8` uses it to replace the hardcoded BF16:

```python
_CPP_TO_TORCH: dict[_tm.DataType, torch.dtype] = {v: k for k, v in _TORCH_TO_CPP.items()}

def _dequant_fp8(tensors: dict[str, Tensor],
                 data_type: _tm.DataType) -> dict[str, Tensor]:
    weight = tensors["weight"]
    scales = tensors["scales"]
    block_size = 128
    fp8_weight = weight.view(torch.float8_e4m3fn).float()
    scale = scales.float()
    scale = scale.repeat_interleave(block_size, dim=0)
    scale = scale.repeat_interleave(block_size, dim=1)
    scale = scale[: fp8_weight.shape[0], : fp8_weight.shape[1]]
    result = {"weight": (fp8_weight * scale).to(_CPP_TO_TORCH[data_type])}
    if "bias" in tensors:
        result["bias"] = tensors["bias"]
    return result
```

This is a behavior change for FP8-activation-FP16 models (today silently coerced to BF16 — a latent bug).

### Ad-hoc `Linear` constructions normalized

Every `Linear(...)` call site explicitly supplies `weight_format` and `data_format`:

| Site | Change |
|---|---|
| `gpt_oss_spec.py:200` MoE gate: `Linear(tensors)` | `Linear(tensors, weight_format=TRIVIAL_FORMAT, data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype()))` |
| `glm4_moe_lite_spec.py:239` MoE gate: same | same |
| `mla.py:57, 60, 74` intermediate `Linear(tensors={...})` | inherit from input linear (`weight_format=q_b.weight_format, data_format=q_b.data_format`) |

### `Linear.concat_*_dim` asserts uniform formats

The silent `wfmt = next(iter(fmts)) if len(fmts) == 1 else None` fallback becomes:

```python
@classmethod
def concat_out_dim(cls, xs):
    first = xs[0]
    result = {kind: torch.cat([x.tensors[kind] for x in xs], dim=first.tensors[kind].dim() - 1)
              for kind in first.tensors}
    wfmts = {x.weight_format for x in xs}
    dfmts = {x.data_format  for x in xs}
    assert len(wfmts) == 1 and len(dfmts) == 1, (
        "concat requires uniform weight_format and data_format; "
        "call dequant_mixed first")
    return Linear(tensors=result,
                  weight_format=next(iter(wfmts)),
                  data_format=next(iter(dfmts)))
```

Same for `concat_in_dim`. `split_out_dim` / `split_in_dim` inherit from `self` as today — correct because `self` is now guaranteed non-None.

## 5. Spec-level `_group_size` purge

### Converter resolves the active `WeightFormat` once

```python
# converter.py (replaces the `group_size = _validate_quant_group_size(...)` flow tail)
group_size = _validate_quant_group_size(engine_config.model_format, group_size)

fmt = get_weight_format(engine_config.model_format)
if fmt.block_in == 0:                              # AWQ / GPTQ / CT
    fmt = replace(fmt,
                  block_in=group_size,
                  block_out=fmt.block_out or group_size)
# FP8 (block_in=128) / MXFP4 (block_in=32) / TRIVIAL (block_in=None) pass through unchanged.

spec = spec_cls(hf_cfg, engine_config, weight_format=fmt)
```

### `TextModelSpec`

```python
def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format: WeightFormat):
    self.hf_cfg = hf_cfg
    self.engine_cfg = engine_cfg
    self._weight_format = weight_format
    self._parse_base(hf_cfg)

def _linear(self, pfx: str):
    return build_linear(self.params, pfx,
                        weight_format=self._weight_format,
                        data_type=self._cpp_dtype())
```

Deleted: `self._group_size`, `group_size` kwarg on `__init__`.

### Subclass specs

Every subclass `__init__` loses its `group_size` forwarding boilerplate:

```python
# Before
def __init__(self, hf_cfg, engine_cfg, *, group_size: int = 0):
    super().__init__(hf_cfg, engine_cfg, group_size=group_size)
    ...

# After
def __init__(self, hf_cfg, engine_cfg, *, weight_format):
    super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
    ...
```

Direct `build_linear` calls in `qwen3_5_spec.py` and `gpt_oss_spec.py`:

```python
# Before
lin = build_linear(self.params, pfx, index=expert_idx,
                   block_in=self._group_size,
                   block_out=self._group_size)
# After
lin = build_linear(self.params, pfx, index=expert_idx,
                   weight_format=self._weight_format,
                   data_type=self._cpp_dtype())
```

### Why store on the spec (not engine_config)

`WeightFormat` is an internal type from `kind_map.py`. `TurbomindEngineConfig` is a public, user-facing dataclass. Leaking `WeightFormat` onto the public config couples the API to an internal type. A private `self._weight_format` on the spec is the right scope.

### `FORMAT_PRIORITY`

After this change, `FORMAT_PRIORITY` is no longer iterated at runtime — `build_linear`'s loop collapses to `active-format → TRIVIAL_FORMAT fallback`. The list survives solely as the source of `ALL_SUFFIXES` (the checkpoint-suffix union). It is not deleted.

## 6. Invariants, validation, testing

### Four post-refactor invariants

1. Every `Linear` has `weight_format: WeightFormat` and `data_format: _tm.DataFormat`, both non-None. `None` as a legal value is eliminated.
2. `linear.data_format == linear.weight_format.make_data_format(data_type)` for the `data_type` in scope at construction. Not runtime-enforced (no cheap check), but every construction site obeys it by calling `make_data_format` explicitly.
3. `WeightFormat.block_in` / `block_out` on a `Linear`'s `weight_format` are fully resolved (no `0` sentinels). Resolution happens exactly once, in `converter.py`.
4. `LinearConfig.format` fully describes weight storage. The `LinearWeight` ctor is single-phase: all three `DataFormat`s populated from a valid config. No `set_weight_spec`.

### Validation boundaries

| Boundary | Check | Failure mode |
|---|---|---|
| Python `WeightFormat.make_data_format(data_type)` | None — pure forward call | — |
| C++ `ResolveLinearWeightFormat` | Validates `(weight_dtype, block_in, block_out)` combination; `TM_CHECK` on violations | `RuntimeError` via pybind |
| C++ `LinearWeight` ctor → `DeriveActivationFormats` | `TM_CHECK` on unreachable cases only | `RuntimeError` via pybind |
| Python `_commit_linear` | `assert linear.data_format is not None` | `AssertionError` with linear name |
| Python `Linear.concat_*_dim` | `assert` uniform `weight_format` and `data_format` across inputs | `AssertionError` pointing to `dequant_mixed` |

No `if fmt is None` defensive branches remain. Violations surface as precise assertions.

### Tests

**C++ `test_data_format.cc`** — update REQUIRE tuples for the new signature + ordering:

| Case | Today | After |
|---|---|---|
| trivial half | `{1, 1}` | `{1, 1}` |
| FP8 | `{128, 128}` | `{128, 128}` |
| FP4 gs=128 | `{1, 128}` | `{128, 1}` |
| AWQ U4 gs=128 | `{1, 128}` | `{128, 1}` |
| U8 gs=64 | `{1, 64}` | `{64, 1}` |
| trivial BF16 | (no block_sizes check) | — |

All `MakeLinearWeightFormat(...)` call sites become `ResolveLinearWeightFormat(...)` with the new 4-arg signature.

**Python `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`** — `_make_linear` helper updated to construct `Linear`s with `TRIVIAL_FORMAT` and a real `make_data_format(...)` result instead of leaving `weight_format` / `data_format` defaulted. Existing `test_format_propagation` which forces `'fake_fmt'` / `'fake_data'` via `object.__setattr__` continues to work (decorators don't interpret the format).

**Python `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py`** — expected to pass unchanged. Regression signal: if it fails, the `group_size=32` commit path is broken.

**Integration (per AGENTS.md, via `scripts/test_turbomind_model.py`)**:

| Model axis | Purpose | TP |
|---|---|---|
| Trivial BF16 dense | control; exercises single-phase ctor with empty DataFormat path | 1, 2 |
| AWQ U4 | exercises resolved block_in threading; `_commit_linear` pass-through; quantized alloc_shape relabel | 1, 2 |
| FP8 (blocked) | exercises fixed block sizes, SM90 native FP8 input derivation | 1, 2 |
| Compressed-tensors int4 g=32 | regression guard for the commit-simplification fix | 1 |
| Mixed-format fusion (e.g. attn with trivial + AWQ proj) | exercises `dequant_mixed` + explicit `data_type` plumbing | 1 |
| MoE (gpt_oss or qwen3_5) | exercises packed-expert `build_linear(index=...)` | 1 |

Acceptance gate per run: ≥128 tokens of coherent, prompt-relevant output. Gibberish = silent bug.

## Behavior changes summary

1. **`set_weight_spec`, `configure`, `LinearPolicy`, `ResolveLinearPolicy`, `MakeLinearWeightFormat` are gone.** Single-phase construction from `LinearConfig.format`.
2. **`DataFormat.block_sizes` ordering flips** to tensor-shape order. Readers update from `block_sizes[1]` to `block_sizes[0]` for K-axis grouping. Invisible to callers.
3. **`_dequant_fp8` honors `data_type`**: fixes a latent bug where FP8-activation-FP16 models were silently coerced to BF16.
4. **`Linear.weight_format` and `Linear.data_format` are required.** `None` is no longer legal; ad-hoc construction sites updated to supply both explicitly.
5. **`concat_*_dim` asserts uniform formats.** Previously returned a silent `None` weight_format on mixed input, deferring the error; now fails at the assert with a pointer to `dequant_mixed`.
6. **Spec subclasses drop `group_size` kwarg.** `converter.py` resolves the active `WeightFormat` once; specs carry `self._weight_format` instead of `self._group_size`.

## Files changed

| File | Change |
|---|---|
| `src/turbomind/core/data_format.h` | `MakeLinearWeightFormat` → `ResolveLinearWeightFormat` with new signature `(data_type, weight_dtype, block_in, block_out)`. |
| `src/turbomind/core/data_format.cc` | Same; emit `block_sizes` in tensor-shape order. |
| `src/turbomind/core/test_data_format.cc` | Update REQUIREs for new signature + ordering. |
| `src/turbomind/models/linear_weight.h` | `LinearConfig.format` field added; three DataFormats on LinearWeight; delete `configure`/`set_weight_spec`/`preprocess`/`LinearPolicy`/`group_size`/scalar `weight_format`. |
| `src/turbomind/models/linear_weight.cc` | Single-phase ctor; `DeriveActivationFormats` replaces `ResolveLinearPolicy`; `prepare()` reads `block_sizes[0]` for K-group size; `copy_metadata_to` updated. |
| `src/turbomind/models/moe_weight.cc` | `d.weight_format == kFloat8_e4m3` → `d.weight_format.dtype == kFloat8_e4m3` (field is now a struct). |
| `src/turbomind/python/bind.cpp` | Rename factory binding; remove `set_weight_spec` binding. `DataFormat` bindings unchanged. |
| `lmdeploy/turbomind/deploy/kind_map.py` | `WeightFormat.make_data_format(data_type)` replaces dead `to_data_format`; `build_linear(*, data_type, weight_format, index)`; drop `block_in`/`block_out` kwargs; `WeightFormat.dequant` signature takes `data_type`; `_dequant_fp8` uses `_CPP_TO_TORCH[data_type]`; `_dequant_awq` accepts unused `data_type`. |
| `lmdeploy/turbomind/deploy/linear.py` | `Linear.weight_format` / `data_format` required (drop `| None`, drop defaults); `concat_*_dim` asserts uniform formats. |
| `lmdeploy/turbomind/deploy/converter.py` | Resolve active `WeightFormat` via `replace`; pass `weight_format=fmt` to `spec_cls(...)` (was `group_size=...`). |
| `lmdeploy/turbomind/deploy/spec.py` | `TextModelSpec.__init__(*, weight_format)`; store `self._weight_format`; `_linear()` passes `weight_format` + `data_type`. Delete `group_size` kwarg and `self._group_size`. |
| `lmdeploy/turbomind/deploy/builder/_base.py` | `_commit_linear` pass-through (`lin_cfg.format = linear.data_format`); delete `set_weight_spec` call + `_infer_cpp_linear_dtype`; `_dequant_linear(*, data_type)`; `_ensure_compatible_formats(*, data_type)`. |
| `lmdeploy/turbomind/deploy/builder/attention.py` | `dequant_mixed(*linears, data_type)`; callers pass `self.config.data_type`. |
| `lmdeploy/turbomind/deploy/builder/deltanet.py` | `_ensure_compatible_formats` call passes `data_type=self.config.data_type`. |
| `lmdeploy/turbomind/deploy/builder/mla.py` | Three `Linear(tensors={...})` constructions inherit `weight_format` / `data_format` from input. |
| `lmdeploy/turbomind/deploy/source_model/utils.py` | Delete duplicate `_dequant_linear`; import from `_base`. `_fold_head_dim(*, data_type)`; caller passes `self._cpp_dtype()`. |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | `__init__` drops `group_size` kwarg; takes `weight_format` through super. |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Same. Direct `build_linear` calls switch to `weight_format=self._weight_format, data_type=self._cpp_dtype()`. |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Same. Plus MoE gate Linear construction supplies `weight_format=TRIVIAL_FORMAT, data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype())`. |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | `__init__` drops `group_size` kwarg. MoE gate Linear construction supplies formats explicitly. |
| `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py` | `_make_linear` helper builds real `TRIVIAL_FORMAT` + `data_format`. |

## Commit plan (sketch)

Five atomic commits in order. The full implementation plan goes in a separate plan doc; this sketch confirms the seams are natural and each commit is testable in isolation.

1. **`refactor(core): ResolveLinearWeightFormat factory with explicit block sizes; tensor-shape ordered block_sizes`** — C++-only. Rename + new signature + ordering + update `test_data_format.cc`, `ResolveLinearPolicy`/`prepare()` readers, pybind binding name. `set_weight_spec` still bolted on — Python still calls it with `block_in = block_sizes[0]`. Verifies on any model.

2. **`refactor(deploy): WeightFormat.make_data_format; populate Linear.data_format; build_linear takes data_type`** — Python-only. `build_linear` gains required `data_type` kwarg; still iterates `FORMAT_PRIORITY` and runs `dataclasses.replace(fmt, block_in=..., block_out=...)` from `block_in`/`block_out` kwargs as today; calls `fmt.make_data_format(data_type)` to populate `Linear.data_format`. `_commit_linear` still calls `set_weight_spec` but reads `linear.data_format.dtype` directly (deletes `_infer_cpp_linear_dtype`). No behavior change; verifies on any model.

3. **`refactor(deploy): explicit data_type in dequant pipelines; fix _dequant_fp8 hardcoded BF16`** — Python-only. Plumb `data_type` through dequant helpers (`_dequant_linear`, `_ensure_compatible_formats`, `dequant_mixed`, `_fold_head_dim`); change `WeightFormat.dequant` signature uniformly; rewrite `_dequant_fp8` to honor `data_type`; dedupe `_dequant_linear`; tighten `Linear.weight_format`/`data_format` to required; update ad-hoc `Linear(...)` constructions (MoE gates, MLA intermediates); `concat_*_dim` asserts uniform formats. Verifies on mixed-format models, AWQ+bias attention fusion, and FP8-activation-FP16 (latent bug fix).

4. **`refactor(linear_weight): single-phase LinearConfig.format; delete set_weight_spec / configure / LinearPolicy`** — C++ + Python together. Add `LinearConfig.format`; single-phase ctor with `DeriveActivationFormats` replacing `ResolveLinearPolicy`; delete `set_weight_spec`, `configure`, `preprocess`, `LinearPolicy`, `policy`/`group_size`/scalar `weight_format` fields. Python `_commit_linear` writes `lin_cfg.format = linear.data_format` instead of calling `set_weight_spec`. Largest atomic; verifies on the full matrix.

5. **`refactor(deploy): resolve WeightFormat at converter; purge Spec._group_size`** — Python-only. `converter.py` resolves active `WeightFormat` via `replace(fmt, block_in=group_size, ...)`; `TextModelSpec.__init__` takes `weight_format` instead of `group_size`; subclass specs drop their `group_size` kwarg boilerplate; direct `build_linear` call sites in subclass specs pass `self._weight_format` and `self._cpp_dtype()`. `build_linear` gains `weight_format` kwarg and drops `block_in`/`block_out`; loop collapses to active-format-vs-trivial. Verifies on any model.

Seams 1 and 2 are safe prep. Seam 3 is the explicit-plumbing + latent-bug pass. Seam 4 is the structural collapse. Seam 5 is the wrap-up cleanup.
