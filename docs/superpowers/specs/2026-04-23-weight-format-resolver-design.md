# Weight Format Resolver: Centralized Multi-Format Resolution

## Problem

Weight-format handling is threaded through every spec and several utils, and
the `WeightFormat` type is shaped awkwardly:

- The single active quantized format is passed explicitly everywhere
  (`spec_cls(..., weight_format=weight_format)` → `self._weight_format` →
  every `build_linear` call → every `read_packed_moe_expert` call).
- The model's compute dtype is passed in parallel
  (`self._cpp_dtype()` → `build_linear(data_type=...)`, `read_packed_moe_expert(data_type=...)`,
  `reorder_rotary_emb(data_type=...)`, `TextModelBuilder(data_type=...)`).
- `WeightFormat` is a `@dataclass(frozen=True)` that holds callable fields
  (`normalizer`, `packer`, `accepts`, `zeros_factory`, `dequant`) — so it is
  effectively a policy object wearing a dataclass's clothes.
- Because it is frozen, the converter must use `dataclasses.replace(..., block_in=group_size)`
  to inject runtime block sizes. The `block_in=0` "read at commit time" sentinel
  exists only to work around this.
- `cpp_dtype_name: str` is a string attribute looked up against `_turbomind.DataType`
  at each `make_data_format` call instead of storing the enum value directly.
- `build_linear` returns `None` silently in two unrelated situations: "no
  tensors at prefix" (legitimate absence — e.g. `q_b_proj` vs `q_proj`,
  packed vs unpacked MoE experts) and "tensors exist but no format accepts
  them" (broken checkpoint). Consumers cannot distinguish these, so every
  caller handles `None` defensively.
- `build_linear` probes the union of every known format's suffixes
  (`ALL_SUFFIXES`), a list that grows every time a new format is added,
  even when only two formats are actually in play for a given model.
- Today a model can only carry one active quantized format (plus implicit
  trivial fallback). Mixed-format models (router = trivial alongside
  quantized experts) work only because `build_linear` hardcodes the fallback
  to `TRIVIAL_FORMAT`.

## Goals

1. Centralize all "which format for which checkpoint tensors" logic in a
   single `WeightFormatResolver` owned by the spec.
2. Centralize the model compute dtype on the resolver. It becomes a property
   of the single `resolver=` handle passed through `build_linear`,
   `read_packed_moe_expert`, and (the Linear path of) `reorder_rotary_emb`,
   replacing the paired `data_type=, weight_format=` kwargs. The dtype
   still flows through — but as one field of one object, not two parallel
   parameters, and the spec no longer holds a `_weight_format` alongside
   `_cpp_dtype()`.
3. Replace the dataclass-with-callables `WeightFormat` with an abstract base
   class + one concrete subclass per format. Each subclass stores its block
   sizes and `_tm.DataType` as ordinary instance / class state.
4. Explicitly support a list of candidate formats per model so mixed-format
   resolution is first-class rather than implicit. Resolution is priority-based
   (quantized candidates first, trivial last).
5. Make failure modes loud and distinct: "no tensors found and caller did not
   opt into optional" and "tensors present but no candidate accepts" both
   raise, with diagnostic context. Only "no tensors AND caller passed
   `optional=True`" returns `None`.

## Design

### `WeightFormat` class hierarchy

Replace `kind_map.py::WeightFormat` (frozen dataclass) with an abstract base
class in a new file, `lmdeploy/turbomind/deploy/weight_format.py`.

```python
class WeightFormat(ABC):
    # Class-level, immutable per format
    name:           ClassVar[str]
    suffix_map:     ClassVar[dict[str, str]]            # {.qweight: weight, ...}
    weight_dtype:   ClassVar["_tm.DataType | None"]     # None for trivial
    has_zero_point: ClassVar[bool]                      # gates synthesize_zeros in the resolver

    # Instance state — only what the format actually varies on
    block_in:  int | None
    block_out: int | None

    @abstractmethod
    def accepts(self, available: dict[str, Tensor]) -> bool: ...
    @abstractmethod
    def normalize(self, tensor: Tensor, kind: str) -> Tensor: ...

    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        return tensor                                   # identity default, actually used

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}.synthesize_zeros not implemented")

    def dequant(self, tensors, data_type) -> dict[str, Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__}.dequant not implemented")

    def make_data_format(self, data_type) -> "_tm.DataFormat":
        if self.weight_dtype is None:
            return _tm.ResolveLinearWeightFormat(data_type, data_type, 1, 1)
        return _tm.ResolveLinearWeightFormat(
            data_type, self.weight_dtype,
            self.block_in or 1, self.block_out or 1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WeightFormat):
            return NotImplemented
        return (type(self) is type(other)
                and self.block_in  == other.block_in
                and self.block_out == other.block_out)

    def __hash__(self) -> int:
        return hash((type(self), self.block_in, self.block_out))
```

`pack` has a valid identity default used at runtime by `TrivialFormat` and
`FP8Format`. `synthesize_zeros` and `dequant` have no valid general
default; the base implementations raise, so calling them on a format that
doesn't support the operation is a loud bug. Formats that do support the
operation simply override.

`__eq__` / `__hash__` are defined explicitly so that two format instances
compare equal when they share class and block sizes. This matters for the
set-based uniformity checks in `Linear.concat_out_dim` /
`Linear.concat_in_dim` (which today work because
`WeightFormat` is a frozen dataclass with auto-generated equality).
Consumers like `_dequant_linear` can therefore construct fresh
`TrivialFormat()` instances locally without needing access to a shared
singleton — the freshly built `Linear` compares equal to resolver-built
trivial linears during fusion.

Concrete subclasses:

| Class                       | `weight_dtype`                      | `has_zero_point` | Overrides                               |
| --------------------------- | ----------------------------------- | ---------------- | --------------------------------------- |
| `TrivialFormat`             | `None`                              | `False`          | `dequant` (identity)                    |
| `AWQFormat`                 | `_tm.DataType.TYPE_UINT4`           | `True`           | `pack`, `dequant`                       |
| `GPTQFormat`                | `_tm.DataType.TYPE_UINT4`           | `True`           | `pack`, `synthesize_zeros`              |
| `CompressedTensorFormat`    | `_tm.DataType.TYPE_UINT4`           | `True`           | `pack`, `synthesize_zeros`              |
| `FP8Format`                 | `_tm.DataType.TYPE_FP8_E4M3`        | `False`          | `dequant`                               |
| `MXFP4Format`               | `_tm.DataType.TYPE_FP4_E2M1`        | `False`          | `pack`                                  |

Notes:

- `TrivialFormat.__init__()` takes no args; block sizes stay `None`.
- `TrivialFormat.dequant` returns the tensors unchanged (nothing to undo —
  already trivial). This makes mixed fusion groups containing trivial
  linears work without a caller-side special case.
- The int4 formats (`AWQFormat`, `GPTQFormat`, `CompressedTensorFormat`)
  take `*, block_in: int` required (the converter supplies the group_size
  — no more `block_in=0` sentinel).
- `FP8Format.__init__()` hard-codes `block_in=block_out=128`; `MXFP4Format.__init__()` hard-codes `block_in=32`.
- `AWQFormat` sets `has_zero_point=True` but doesn't override `synthesize_zeros`
  because the AWQ checkpoint always ships `.qzeros`. A broken AWQ
  checkpoint without zeros would trigger the base-class `NotImplementedError`
  at the resolver.
- `GPTQFormat`, `CompressedTensorFormat`, and `MXFP4Format` don't override
  `dequant`: they have no Python-side dequantization path and must stay
  in quantized form through fusion. A fusion group that somehow required
  dequantizing one of them would hit the base-class `NotImplementedError`
  — the correct, loud signal for that misconfiguration.

`synthesize_zeros` is consulted only when `has_zero_point` is `True` and
`zeros` is absent from the normalized tensors — replacing the old
`zeros_factory is not None` idiom plus `WeightFormat.complete_tensors`.

### `WeightFormatResolver`

New class in the same file. Owns the model compute dtype and the ordered
candidate list.

```python
class WeightFormatResolver:
    def __init__(self, *, data_type: "_tm.DataType",
                 formats: list[WeightFormat]):
        self._data_type = data_type
        self._formats   = formats
        self._suffixes  = frozenset(s for f in formats for s in f.suffix_map)

    @property
    def data_type(self) -> "_tm.DataType":
        return self._data_type

    def resolve(self, params: dict[str, Tensor], prefix: str, *,
                index: int | None = None,
                optional: bool = False) -> Linear | None:
        available = {s: params[prefix + s]
                     for s in self._suffixes if (prefix + s) in params}
        if index is not None:
            available = {s: t[index] for s, t in available.items()}

        if not available:
            if optional:
                return None
            raise KeyError(
                f"no checkpoint tensors found at prefix {prefix!r} "
                f"(candidate suffixes: {sorted(self._suffixes)})")

        for fmt in self._formats:
            if fmt.accepts(available):
                return self._build_linear(fmt, available)

        raise ValueError(
            f"no weight format accepts tensors at {prefix!r}: "
            f"got {sorted(available)}, "
            f"tried {[f.name for f in self._formats]}")

    def _build_linear(self, fmt, available) -> Linear:
        tensors = {
            kind: fmt.normalize(available[s], kind)
            for s, kind in fmt.suffix_map.items()
            if s in available
        }
        if fmt.has_zero_point and "zeros" not in tensors:
            # synthesize_zeros raises NotImplementedError by default; formats
            # that declare has_zero_point=True must override when the
            # checkpoint can legitimately omit zeros (GPTQ/CT symmetric int4).
            tensors["zeros"] = fmt.synthesize_zeros(tensors["scales"])
        return Linear(tensors=tensors,
                      weight_format=fmt,
                      data_format=fmt.make_data_format(self._data_type))
```

Key properties:

- The suffix probe is scoped to the candidates' union, not a global
  `ALL_SUFFIXES`. Adding a new format anywhere else in the module does not
  widen the probe at unrelated call sites.
- Priority is encoded by list order. The converter puts quantized candidates
  first and `TrivialFormat()` last, so a prefix whose tensors satisfy both
  a quant format and trivial (never happens in practice — the key sets
  disjoin) deterministically resolves to the quant one, and a prefix that
  only matches trivial (router, norm-like linears in a quantized model)
  falls through.
- Two distinct failure paths, both loud. `optional=False` + zero tensors →
  `KeyError`. Any tensors present + no `accepts` match → `ValueError`. The
  only `None` return is legitimate "optional module absent".

### Converter

`converter.py` stops importing `get_weight_format` and `replace`; replaces
them with a local factory:

```python
def _build_resolver(model_format, group_size, data_type) -> WeightFormatResolver:
    formats: list[WeightFormat] = []
    match model_format:
        case None | "hf":                 pass
        case "awq":                       formats.append(AWQFormat(block_in=group_size))
        case "gptq":                      formats.append(GPTQFormat(block_in=group_size))
        case "compressed-tensors":        formats.append(CompressedTensorFormat(block_in=group_size))
        case "fp8":                       formats.append(FP8Format())
        case "mxfp4":                     formats.append(MXFP4Format())
        case _:
            raise ValueError(f"unknown model_format: {model_format!r}")
    formats.append(TrivialFormat())
    return WeightFormatResolver(data_type=data_type, formats=formats)
```

`get_tm_config` gains a resolver-build step with a specific ordering
constraint: it must run **after** the int4 fp16 force (line 170-171
today) and **before** the `compressed-tensors → awq` rename (line 172-173).
This preserves the existing comment's intent ("resolve the active
WeightFormat before the CT→AWQ rename so compressed-tensors models still
get CompressedTensorFormat").

```python
# ... quant_config processing, group_size validation,
#     model_format defaulting to 'hf' ...

# Resolve dtype and force fp16 for int4 formats.
dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
    dtype = 'float16'
engine_config.dtype = dtype

# Build resolver now — model_format is still 'compressed-tensors', dtype
# is finalized.
data_type = _cpp_dtype(dtype)                       # from .builder import _cpp_dtype
resolver  = _build_resolver(engine_config.model_format,
                            group_size, data_type)

# C++-side label rename (no effect on the resolver).
if engine_config.model_format == 'compressed-tensors':
    engine_config.model_format = 'awq'

# ... session_len, tp defaults, hf_overrides, spec build ...
spec = spec_cls(hf_cfg, engine_config, resolver=resolver)
```

`_cpp_dtype` is re-imported from `.builder` in `converter.py` (it already
lives in `builder/_base.py` and is exported via `builder/__init__.py`).

### Spec refactor

`TextModelSpec` drops `_weight_format`; `_linear` and `_cpp_dtype` delegate
to the resolver:

```python
class TextModelSpec(ABC):
    def __init__(self, hf_cfg, engine_cfg, *, resolver):
        self.hf_cfg     = hf_cfg
        self.engine_cfg = engine_cfg
        self._resolver  = resolver
        self._parse_base(hf_cfg)

    def _linear(self, pfx: str, *, optional: bool = False) -> Linear | None:
        return self._resolver.resolve(self.params, pfx, optional=optional)

    def _cpp_dtype(self):
        return self._resolver.data_type
```

Each of the four spec subclasses changes its one constructor parameter name
(`weight_format` → `resolver`) and the `super().__init__` call.

Call-site changes to accommodate the loud failure contract:

- **`glm4_moe_lite_spec.py::attn`** — `q_b_proj` is optional in MLA
  (some checkpoints use `q_proj` instead):

  ```python
  q_b = (self._linear(f'{pfx}.q_b_proj', optional=True)
         or self._linear(f'{pfx}.q_proj'))
  ```

  The second call is non-optional: if both are missing, the checkpoint is
  broken and we raise.

- **`qwen3_5_spec.py::_moe_expert_ffn`** — per-expert FFN may be entirely
  absent when the checkpoint packs experts. `ffn()` gains an `optional=False`
  kwarg that threads through to its three `_linear` calls with explicit
  all-or-nothing semantics:

  ```python
  def ffn(self, pfx, layer, inter_size=None, fused_moe=False, *, optional=False):
      w1 = self._linear(f'{pfx}.gate_proj', optional=optional)
      w3 = self._linear(f'{pfx}.up_proj',   optional=optional)
      w2 = self._linear(f'{pfx}.down_proj', optional=optional)
      present = [t is not None for t in (w1, w3, w2)]
      if not any(present):
          return None                                # all absent → legitimate fallback signal
      if not all(present):
          raise ValueError(
              f'{pfx}: partial FFN checkpoint '
              f'(gate_proj={present[0]}, up_proj={present[1]}, down_proj={present[2]})')
      ...
  ```

  `_moe_expert_ffn` then calls `self.ffn(..., optional=True)` and falls back
  to `_packed_moe_ffn` when it gets `None`. Partial-absence checkpoints
  (one of three missing) are always an error, not a fallback signal.

### Utils refactor

`read_packed_moe_expert` takes the resolver instead of the `data_type` +
`weight_format` pair:

```python
def read_packed_moe_expert(params, gate_up_pfx, down_pfx, expert_idx,
                           *, resolver, interleaved=False, trans=False):
    gate_up = resolver.resolve(params, gate_up_pfx, index=expert_idx)
    down    = resolver.resolve(params, down_pfx,    index=expert_idx)
    ...
```

Callers in `qwen3_5_spec.py` and `gpt_oss_spec.py` change
`data_type=self._cpp_dtype(), weight_format=self._weight_format` to
`resolver=self._resolver`.

`reorder_rotary_emb` has two input paths today. The `Tensor` path doesn't
need the dtype at all and is unchanged. The `Linear` path needs the dtype
for its `_dequant_linear` fallback (when `block_out % head_dim != 0`) — we
switch that kwarg from `data_type` to `resolver`:

```python
def reorder_rotary_emb(x, head_dim: int, rope_dim: int, *, resolver=None):
    if isinstance(x, Linear):
        if resolver is None:
            raise TypeError("resolver required when passing a Linear")
        # ... internally calls _dequant_linear(x, data_type=resolver.data_type)
    # Tensor path: resolver ignored; unchanged behavior
    return _reorder_rotary_emb(x, head_dim, rope_dim)
```

Only the three Linear-input call sites change:

- `qwen3_spec.py::attn` — Q and K projections.
- `qwen3_5_spec.py::attn` — Q and K projections.
- `gpt_oss_spec.py::attn` — Q and K projections.

Each swaps `data_type=self._cpp_dtype()` for `resolver=self._resolver`.

`TextModelSpec.qk_norm` in `spec.py` uses `reorder_rotary_emb` on a raw
`Tensor` (the norm weight), which takes the Tensor path. **It does not
need the resolver** and its signature is unchanged.

The dtype threading for the `reorder_rotary_emb` Linear path is not
eliminated — `_dequant_linear` genuinely needs the compute dtype
(`DataFormat.dtype` stores the weight storage dtype, not compute) — but it
now flows through the resolver object for consistency with the rest of
the refactor.

### Builder-side changes

Two small adjustments in `builder/_base.py`:

**`_dequant_linear`** drops the `fmt.dequant is None` guard and calls
`fmt.dequant(...)` directly:

```python
def _dequant_linear(linear, *, data_type):
    fmt = linear.weight_format
    new_tensors = fmt.dequant(linear.tensors, data_type)   # identity for trivial; real for AWQ/FP8; raises for others
    trivial = TrivialFormat()
    return Linear(
        tensors=new_tensors,
        weight_format=trivial,
        data_format=trivial.make_data_format(data_type))
```

`TrivialFormat.dequant` is identity, `AWQFormat.dequant` and
`FP8Format.dequant` do real work, GPTQ / CompressedTensor / MXFP4 inherit
the base-class raise. A mixed fusion group that would require dequantizing
one of those three is a broken configuration — the raise makes it visible
at the call site. The fresh `TrivialFormat()` instance is equal-by-value
to the resolver's own trivial instance (see `__eq__`/`__hash__` above) so
downstream set-uniformity checks still work.

**`_commit_linear`**'s packer hoist simplifies. Today:

```python
packer = fmt.packer if fmt else None
if packer is not None:
    tensors = {k: packer(t, k) for k, t in linear.tensors.items()}
else:
    tensors = linear.tensors
```

`fmt.packer` (a nullable callable field) becomes `fmt.pack` (a method with
a valid identity default). The hoist collapses to one line:

```python
tensors = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
```

Identity `pack` on `TrivialFormat` / `FP8Format` matches today's "packer is
None, use tensors unchanged" path without the None-check branch.

### What stays unchanged

- Builder code paths that don't touch format capability: `_commit_linear`,
  `_copy_shard_to_param`, TP-splitting, packer dispatch, fusion-time
  layout helpers.
- The dtype channel inside builders (`self.config.data_type`), which is
  separate from the spec-side threading this refactor targets.
- `TextModelBuilder(..., data_type=self._cpp_dtype())` at the root. The
  call site now sources `data_type` from `self._resolver.data_type` via
  `_cpp_dtype` but the builder signature does not change.
- `Linear` itself (the `weight_format` and `data_format` fields).
- The C++ `_tm.ResolveLinearWeightFormat` path and `DataFormat`.

### Deletions

From `kind_map.py` (to be renamed `weight_format.py`):

- `ALL_SUFFIXES` frozenset
- `FORMAT_PRIORITY` list
- `_WEIGHT_FORMAT_MAP` dict, `get_weight_format()` function
- `_FORMAT_MAP` dict, `get_suffix_map()` function
- `_NORMALIZER_MAP` dict, `get_normalizer()` function
- The six module-level `*_FORMAT` singleton instances
- `WeightFormat.complete_tensors` method and `zeros_factory` field (subsumed by
  `has_zero_point` + `synthesize_zeros` on the ABC)
- `cpp_dtype_name: str` attribute (replaced by `weight_dtype: _tm.DataType`
  class attribute)
- `build_linear` top-level function (replaced by `resolver.resolve`)

From `converter.py`:

- `from .kind_map import get_weight_format` import
- `from dataclasses import replace` import and the `replace(weight_format,
  block_in=group_size)` block
- (adds `from .builder import _cpp_dtype`)

From `spec.py`:

- `from .builder import _cpp_dtype as _cd` (the spec's `_cpp_dtype()` now
  delegates to `self._resolver.data_type`)
- `self._weight_format` field

### Test file updates

- `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py` loads
  `kind_map.py` by absolute file path (lines 85–87). The path and the
  registered module name both update to `weight_format.py` /
  `lmdeploy.turbomind.deploy.weight_format`. The test itself uses
  `weight_format='placeholder'` as a string sentinel (line 129) and does
  not touch the class hierarchy — no other changes needed there.
- `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py` imports
  `pack_u4_row` from `lmdeploy.turbomind.deploy.parameter`, a module that
  no longer exists in the current tree (the test is pre-broken, unrelated
  to this refactor). We leave it alone.

## Verification

No unit tests exist for this subsystem. Per `AGENTS.md`, verification is
end-to-end via `scripts/test_turbomind_model.py`, requiring ≥128 tokens of
meaningful response per run.

Matrix (one model per candidate path):

| Scenario              | Exercises                                                                         |
| --------------------- | --------------------------------------------------------------------------------- |
| trivial (hf)          | Base resolver path; no quant format in candidate list                             |
| awq                   | `AWQFormat` + `TrivialFormat` (router stays trivial); group_size via ctor         |
| gptq                  | Same, plus `synthesize_zeros` path for symmetric int4                             |
| compressed-tensors    | `CompressedTensorFormat`; pre-rename resolver instantiation                       |
| fp8                   | `FP8Format.dequant` during fusion (`_ensure_compatible_formats`)                  |
| mxfp4 (gpt-oss)       | Mixed-format resolution in one model: experts=mxfp4, router=trivial — canonical   |

Manual smoke tests for the loud-failure paths (not automated):

- Wrong `--model-format` on a checkpoint → `ValueError` with candidate names
  and available suffixes.
- Empty prefix passed to `resolve()` without `optional=True` → `KeyError`
  with candidate suffix list.

## Migration

Single atomic PR. The resolver, `WeightFormat` class hierarchy, converter
factory, spec base refactor, subclass constructor renames, and util
signature changes all depend on each other and cannot be landed in halves
without keeping two resolution paths live. Rollout order within the PR:

1. Rename `kind_map.py` → `weight_format.py`. Overhaul its contents: drop
   the frozen dataclass, the six module-level singletons, and all
   `get_*_map` / `get_weight_format` lookup functions; add the `WeightFormat`
   ABC, six concrete subclasses (with `__eq__`/`__hash__`), and
   `WeightFormatResolver`. The private u4 helpers (`_get_u4_slices`,
   `_unpack_awq_gemm`) and the public `pack_u4_row` stay as module-level
   utilities.
2. Update `converter.py`: drop `get_weight_format` / `dataclasses.replace`
   imports, add `_cpp_dtype` import, insert the `_build_resolver` factory
   and resolver-build step (after int4 fp16 force, before CT→AWQ rename).
3. Update `spec.py` base: `__init__(*, resolver)`, `_linear(pfx, *,
   optional=False)`, `_cpp_dtype` delegates to `resolver.data_type`.
4. Update each of the four spec subclasses: constructor kwarg rename,
   `optional=True` at legitimate-absence sites, swap `data_type=, weight_format=`
   pairs for `resolver=` at util call sites.
5. Update `source_model/utils.py::read_packed_moe_expert` signature
   (`resolver=`) and `reorder_rotary_emb` Linear-path kwarg (`resolver=`).
6. Update `builder/_base.py`: drop the `fmt.dequant is None` guard in
   `_dequant_linear`; collapse the packer hoist in `_commit_linear` to
   always call `fmt.pack(...)`; replace `from ..kind_map import TRIVIAL_FORMAT`
   with `from ..weight_format import TrivialFormat` (used inline).
7. Update `linear.py` TYPE_CHECKING import (`from .weight_format import
   WeightFormat`).
8. Update `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`'s
   hard-coded `kind_map.py` path and module registration name to
   `weight_format.py` / `lmdeploy.turbomind.deploy.weight_format`.
