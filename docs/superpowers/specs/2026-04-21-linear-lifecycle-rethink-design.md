# Linear module lifecycle: rethink the construct / configure / allocate / commit pipeline

Date: 2026-04-21

Scope: TurboMind's linear-weight handling, both Python and C++.

- `src/turbomind/core/data_format.{h,cc}` + `test_data_format.cc`
- `src/turbomind/models/linear_weight.{h,cc}`, `model_weight.{h,cc}`, `language_model.cc`, `ffn_weight.cc`, `moe_weight.cc`
- `src/turbomind/kernels/gemm/test/testbed_v3.h`
- `src/turbomind/python/bind.cpp`
- `src/turbomind/models/llama/LlamaLinear.{h,cu}`
- `lmdeploy/turbomind/deploy/kind_map.py`, `linear.py`, `converter.py`, `spec.py`
- `lmdeploy/turbomind/deploy/builder/_base.py`, `builder/linear.py` (deleted), `builder/attention.py`, `builder/ffn.py`, `builder/mla.py`, `builder/deltanet.py`, `builder/__init__.py`
- `lmdeploy/turbomind/deploy/source_model/*_spec.py`

## Motivation

The current `LinearWeight` lifecycle is a four-phase bolted-together sequence that leaks responsibilities across the Python/C++ seam and duplicates the same information through three parallel channels:

**Phase 1 (C++)** — `handle.create_child(name, LinearConfig{input_dim, output_dim, data_type, has_bias})` constructs a `LinearWeight`. At this point the module is incompletely specified for any quantized format: `weight_format`, `group_size`, and the derived `format: DataFormat` are all defaulted.

**Phase 2 (Python→C++)** — `linear_mod.set_weight_spec(cpp_dtype, block_in)` bolts the storage format onto an already-constructed module. C++ silently coerces trivial-float mismatches (BF16↔FP16), recomputes `format = MakeLinearWeightFormat(...)` and `policy = ResolveLinearPolicy(...)`. The same coercion also lives in Python as `_cast_shard_for_tm`. Two implementations of one rule.

**Phase 3 (Python)** — `handle.param(name).alloc(alloc_shape, alloc_dtype)` per kind. Python has to compute the right shape/dtype and for quantized weights **lies** about them:
- Trivial weight → `alloc(None, model_dtype)` so C++ coerces to compute dtype.
- Quantized weight → `alloc([in, out], UINT4)` even though the tensor is physically int32 `[in, out/8]`. Same byte size, different label.
- Bias/scales/zeros → `alloc(None, None)`.

Nested `if/elif/else` in Python picks the relabel per kind. The module should know its own param shapes from its format; Python shouldn't have to dictate them.

**Phase 4 (C++)** — `prepare()` reads `weight_format`, `input_dtype()`, `data_type`, `is_grouped_`, dispatches. Works correctly, but off state that phases 2–3 set up.

Plus two post-hoc mutations outside the lifecycle:
- `set_grouped(bool)` — separate setter called after `create_child` for MoE expert weights.
- `preprocess()` — now a no-op but still declared.

And two path-level asymmetries on the Python side:
- `LinearBuilder.set_weight(tensor)` (for `token_embeddings` / `lm_head`) bypasses phases 2 and 3, relying on a special branch at the top of `LinearWeight::prepare()` to handle the defaulted state.
- `tok_embeddings` is declared as a `LinearWeight` child in C++ even though it is an embedding lookup parameter, not a GEMM operand; only `->weight` is ever read.

Information flow today, illustrated:

```
Checkpoint ──► build_linear ──► Linear {tensors, weight_format, data_format=DEAD}
                                              │
                                              ▼ at _commit_linear time
     LinearConfig{input_dim, output_dim, data_type, has_bias}       (channel 1)
     linear_mod.set_weight_spec(cpp_dtype, block_in)                 (channel 2)
     handle.param(k).alloc(shape, dtype)   per kind                  (channel 3 — shape/dtype lie for quant)
                                              │
                                              ▼ C++
     MakeLinearWeightFormat(dt, wf, gs) ──► DataFormat
     ResolveLinearPolicy(format, dt, sm) ──► LinearPolicy
     prepare() reads weight_format / input_dtype() / data_type / is_grouped_
```

Three parallel channels carry `(compute_dtype, weight_dtype, block_sizes)` which the C++ side then re-stitches back into a single `DataFormat`. `LinearConfig` "looks" like it fully configures a linear but in fact doesn't carry storage format. `Linear.data_format` is written by `build_linear` but nothing live reads it. `WeightFormat` singletons get cloned via `dataclasses.replace` to splice in runtime block sizes — they are not truly constant.

This spec redesigns the lifecycle around a single authoritative representation: **three `DataFormat`s fully describe the GEMM (weight/input/output); `LinearConfig` carries the weight `DataFormat`; construction is single-phase; param slots are pre-allocated by C++ from the format.**

## Architecture

```
Checkpoint ──► build_linear ──► Linear {tensors, weight_format, data_format}
                                              │
                                              ▼ at _commit_linear time
     LinearConfig{input_dim, output_dim, format=linear.data_format,
                  has_bias, is_grouped, epilogue}
                                              │
                                              ▼ C++
     LinearWeight ctor:
         format = cfg.format                                 # weight storage descriptor
         (input_format, output_format) = DeriveActivationFormats(format, ...)
         for each declared kind in format: alloc param slot  # pre-alloc
     Python commits tensors via slot.copy_from() only        # no alloc at commit time
```

**Ownership after the rethink:**

- Checkpoint-side concerns (suffix map, normalizer, packer, accepts, zeros_factory, dequant) → Python `WeightFormat` — one constant singleton per `FormatKind`.
- Runtime block sizes per linear → Python `BlockSize` from `converter.py` quant-config parse; stored on `Spec`.
- Concrete `DataFormat` per linear (compute dtype + weight storage + block sizes) → produced by `WeightFormat.make_data_format(compute_dtype, block_in, block_out)`; carried on `Linear.data_format`; passed into `LinearConfig.format`.
- Compute-side format (three `DataFormat`s, allocations, GEMM descriptors, format conversion) → C++ `LinearWeight` / `LinearConfig` / `DeriveActivationFormats` / `prepare()`.

One bridge crosses the seam: `WeightFormat.make_data_format(compute_dtype, block_in, block_out) -> _tm.DataFormat`.

## 1. C++ `LinearWeight` lifecycle

### `LinearConfig` carries everything structural

```cpp
struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

    int        input_dim;
    int        output_dim;
    DataFormat format;        // weight storage: compute dtype, weight dtype, block sizes, scales/zeros descriptors
    bool       has_bias;
    bool       is_grouped;    // for MoE grouped-GEMM experts
    Epilogue   epilogue;      // kNone / kGatedSilu
};
```

`LinearConfig.data_type` is removed. Callers read `cfg.format.dtype` — the compute dtype is a property of the weight's `DataFormat`, not a separate field. `is_grouped` and `epilogue` subsume the post-hoc `set_grouped` and `FfnWeight::prepare`'s epilogue mutation.

### `LinearWeight` simplifies

```cpp
class LinearWeight: public core::Module {
public:
    const char* type() const override { return "LinearWeight"; }

    LinearWeight() = default;                         // for MoeWeight::block_ (copy_metadata_to path)
    LinearWeight(const core::LinearConfig& cfg);      // single-phase construction, pre-allocates slots

    void configure(int input_dim, int output_dim,
                   const DataFormat& format, bool has_bias);  // for testbed use

    void prepare() override;                          // entry asserts format.dtype != DataType{}

    void copy_metadata_to(LinearWeight& dst) const;

    explicit operator bool() const noexcept { return static_cast<bool>(weight); }

    // --- three DataFormats fully describe the GEMM input @ weight -> output ---
    DataFormat format;          // alias for weight_format (the weight tensor's storage)
    DataFormat input_format;    // activation input format  (derived at construction)
    DataFormat output_format;   // activation output format (derived at construction)

    DataType input_dtype()  const { return input_format.dtype;  }
    DataType output_dtype() const { return output_format.dtype; }

    // --- dimensions ---
    int input_dim  = 0;
    int output_dim = 0;

    // --- GEMM config knobs ---
    Epilogue    epilogue{};
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

Deleted relative to today:
- `set_weight_spec(DataType, int)` — folded into `LinearConfig.format`.
- `set_grouped(bool)` — folded into `LinearConfig.is_grouped`.
- `preprocess()` — was a no-op.
- `configure(int,int,DataType,bool)` overload — replaced by `configure(int,int,DataFormat,bool)`.
- Public fields `weight_format: DataType`, `group_size: int`, `data_type: DataType` — all redundant with `format`.
- The `LinearPolicy` struct — redundant with the three `DataFormat`s.

### `MakeLinearWeightFormat` signature

```cpp
/// Construct the DataFormat describing a weight tensor in TM [in, out] layout.
/// block_sizes is stored in that tensor's own shape order: {block_in, block_out}.
DataFormat MakeLinearWeightFormat(DataType compute_dtype,
                                  DataType weight_dtype,
                                  int      block_in,
                                  int      block_out);
```

Today's signature is `(compute_dtype, weight_dtype, group_size)` where the single `group_size` is overloaded to populate three different `block_sizes` patterns depending on `weight_dtype` (`{1, gs}` for U4/U8/FP4, `{128, 128}` for FP8, `{1, 1}` for trivial). Making block sizes explicit kills the overloaded magic and enforces the invariant "`block_sizes[i]` corresponds to the described tensor's `shape[i]`".

The function validates that the `(weight_dtype, block_in, block_out)` combo is consistent with known formats. Scales / zeros dtype derivation stays format-specific and stays inside this function.

### `DeriveActivationFormats` replaces `ResolveLinearPolicy`

```cpp
/// Derive (input_format, output_format) for a GEMM whose weight uses `weight_format`.
/// compute_dtype is the model's activation dtype.
std::pair<DataFormat, DataFormat>
DeriveActivationFormats(const DataFormat& weight_format, DataType compute_dtype, int sm);
```

Behavior table (identical to today's `ResolveLinearPolicy`, reshaped):

| Weight format | input_format | output_format |
| --- | --- | --- |
| trivial (`{dtype=bf16/fp16/fp32, block_sizes={1,1}}`) | `{compute_dtype, {1,1}}` | `{compute_dtype, {1,1}}` |
| FP8 on SM90 | `{fp8_e4m3, {1, gs}, scales.dtype=f32}` | `{compute_dtype, {1,1}}` |
| FP8 elsewhere | `{compute_dtype, {1,1}}` | `{compute_dtype, {1,1}}` |
| U4 / U8 / FP4 | `{compute_dtype, {1,1}}` | `{compute_dtype, {1,1}}` |

GEMM-level `QuantDesc` objects are synthesized on-demand from the three `DataFormat`s at `LlamaLinear::Forward` dispatch time, not cached as separate state.

### Pre-allocation at construction

`LinearWeight(const LinearConfig& cfg)`:

1. Stores `input_dim`, `output_dim`, `has_bias_`, `is_grouped_`, `epilogue` from cfg.
2. `format = cfg.format`.
3. `(input_format, output_format) = DeriveActivationFormats(format, format.dtype, getSMVersion())`.
4. Allocates every param slot declared by `format`:
   - `weight`: shape `[input_dim, output_dim]`, dtype `format.dtype`.
   - `scales` if `format.scales.present()`: shape derived from `[input_dim, output_dim]` divided by `format.block_sizes`, dtype `format.scales.dtype`.
   - `zeros` if `format.zeros.present()`: shape = scales' shape, dtype `format.zeros.dtype`.
   - `bias` if `has_bias_`: shape `[output_dim]`, dtype `format.dtype`.

After construction the module has every declared param slot allocated (zeroed) with the correct shape/dtype. Python's commit code no longer computes or passes alloc shape/dtype.

### `prepare()` assertion

Entry:
```cpp
TM_CHECK(format.dtype != DataType{}) << "LinearWeight::prepare: format was never set";
```

Permanent guard: once `_commit_tensor`-on-LinearWeight bypass is gone (Section 3), the trivial-default branch (`if (weight_format == DataType{}) { ... }`) has no reachable case. The assert catches regressions.

### `prepare()` block-size invariant under transposes

When the FP8-native path transposes weight from `[in, out]` to `[out, in]`, it also updates `format.block_sizes` in place to preserve the invariant `block_sizes[i] == weight.shape[i]`:

```cpp
std::swap(format.block_sizes[0], format.block_sizes[1]);
```

For symmetric FP8 (`{128, 128}`) this is numerically invariant, but the descriptor stays consistent with the described tensor even under asymmetric future shapes.

### MoE `block_` remains untouched

`MoeWeight::prepare()` creates `block_` (a batched-pointer view over all experts' prepared LinearWeights) via `add_child(name, std::make_unique<LinearWeight>())` — **default-constructed** — then `LinkLinearExperts` fills it via `copy_metadata_to` plus synthetic `MakeBlockedPtrs` / `MakeStridedPtrs` tensors. This path is outside the `LinearConfig` flow by design: the tensors aren't real GPU buffers. `copy_metadata_to` now copies `format`, `input_format`, `output_format`, `epilogue`, `is_grouped_`, `has_bias_`, `k_desc`, `q_desc` — structurally unchanged, field names shifted to the new scheme.

## 2. Python `WeightFormat` as a constant `DataFormat` factory

### `FormatKind` enum + constant singletons

```python
class FormatKind(enum.Enum):
    TRIVIAL            = "trivial"
    AWQ                = "awq"
    GPTQ               = "gptq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FP8                = "fp8"
    MXFP4              = "mxfp4"


@dataclass(frozen=True)
class WeightFormat:
    """Constant per-format-kind descriptor. A factory of DataFormats.

    Owns checkpoint-side concerns (suffix map, normalizer, packer, detection,
    optional dequant / zeros synthesis) plus a factory method that produces a
    concrete ``_tm.DataFormat`` for a particular linear given compute dtype
    and block sizes. Holds no block-size state itself.
    """
    kind:          FormatKind
    suffix_map:    dict[str, str]
    normalizer:    Callable[[Tensor, str], Tensor]
    packer:        Callable[[Tensor, str], Tensor] | None
    weight_dtype:  _tm.DataType | None   # None iff kind == TRIVIAL (uses compute dtype)
    accepts:       Callable[[dict[str, Tensor]], bool]
    zeros_factory: Callable[[Tensor], Tensor] | None
    dequant:       Callable[[dict[str, Tensor]], dict[str, Tensor]] | None

    def make_data_format(self, *,
                         compute_dtype: _tm.DataType,
                         block_in: int,
                         block_out: int) -> _tm.DataFormat:
        if self.kind is FormatKind.TRIVIAL:
            assert block_in == 1 and block_out == 1, (
                f"TRIVIAL requires block_in == block_out == 1, "
                f"got ({block_in}, {block_out})")
            return _tm.MakeLinearWeightFormat(compute_dtype, compute_dtype,
                                              block_in, block_out)
        assert block_in >= 1 and block_out >= 1, (
            f"{self.kind.value} requires block_in >= 1 and block_out >= 1, "
            f"got ({block_in}, {block_out})")
        return _tm.MakeLinearWeightFormat(compute_dtype, self.weight_dtype,
                                          block_in, block_out)
```

Deleted fields vs today: `name: str`, `block_in: int | None`, `block_out: int | None`, `cpp_dtype_name: str | None`. Replaced by `kind`, `weight_dtype`, and a factory.

Singletons become truly constant — no `dataclasses.replace` runtime cloning.

### `BlockSize` NamedTuple

```python
class BlockSize(NamedTuple):
    block_in: int
    block_out: int
```

Used as the post-parse representation of quantization block sizes. `group_size` does not appear anywhere in Python code after `converter.py`.

### `Linear.data_format` revived, mandatory

```python
@dataclass
class Linear:
    tensors:       dict[str, Tensor]        # keys ⊆ {"weight","bias","scales","zeros"}
    weight_format: WeightFormat             # constant per format-kind
    data_format:   _tm.DataFormat           # per-linear, mandatory, never None

    def __post_init__(self):
        assert 'weight' in self.tensors, "Linear must carry a 'weight' tensor"
        assert self.data_format is not None, "Linear.data_format is mandatory"
```

The `data_format` field carries per-linear block sizes (previously on the mutable `WeightFormat`). Every Linear-producer (`build_linear`, `interleave_linears`, `chunk_linears`, `fold_kv_b`, `pad_wo_input`, `split_output_gate`, `_repeat_kv_heads`, `fuse_qkv`, `split_qkv`, `fuse_gdn`, `dequant_mixed`) threads it through.

Transforms that produce trivial-format Linears (`fold_kv_b` / `pad_wo_input` / `_dequant_linear`) construct a fresh trivial `data_format = TRIVIAL_FORMAT.make_data_format(compute_dtype, 1, 1)`.

`preprocess_linear` is deleted — unreachable today.

### `build_linear` signature

```python
def build_linear(params, prefix, *,
                 compute_dtype: _tm.DataType,
                 blocks: BlockSize,
                 index: int | None = None) -> Linear | None:
    available = {s: params[prefix + s] for s in ALL_SUFFIXES if (prefix + s) in params}
    if index is not None:
        available = {s: t[index] for s, t in available.items()}

    fmt = next((f for f in FORMAT_PRIORITY if f.accepts(available)), None)
    if fmt is None:
        return None

    # Trivial linears in an otherwise-quantized model use (1, 1) regardless
    # of caller-provided blocks.
    if fmt.kind is FormatKind.TRIVIAL:
        data_format = fmt.make_data_format(compute_dtype=compute_dtype,
                                           block_in=1, block_out=1)
    else:
        data_format = fmt.make_data_format(compute_dtype=compute_dtype,
                                           block_in=blocks.block_in,
                                           block_out=blocks.block_out)

    tensors = {kind: fmt.normalizer(available[s], kind)
               for s, kind in fmt.suffix_map.items() if s in available}
    if not tensors:
        return None
    if fmt.zeros_factory and 'scales' in tensors and 'zeros' not in tensors:
        tensors['zeros'] = fmt.zeros_factory(tensors['scales'])

    return Linear(tensors=tensors, weight_format=fmt, data_format=data_format)
```

No `dataclasses.replace(fmt, ...)`. No dead `to_data_format(0, 0)` call. `complete_tensors` inlined.

### `converter.py` — one translation point

```python
# Deleted: _DEFAULT_GROUP_SIZES, _SUPPORTED_GROUP_SIZES, _validate_quant_group_size.

def _parse_quant_blocks(model_format, quant_config) -> BlockSize:
    """Translate quant-config into (block_in, block_out). Single translation point.

    block_in = block size along input (K) dim
    block_out = block size along output (N) dim
    Values >= 1. No silent defaults.
    """
    if model_format in (None, 'hf'):
        return BlockSize(block_in=1, block_out=1)
    if model_format == 'mxfp4':
        return BlockSize(block_in=32, block_out=1)          # fixed by OCP MX standard
    gs = quant_config and quant_config.get('group_size')
    if gs is None or gs < 1:
        raise ValueError(
            f"Format {model_format!r} requires group_size >= 1 in quant_config; got {gs!r}.")
    if model_format in ('awq', 'gptq', 'compressed-tensors'):
        return BlockSize(block_in=gs, block_out=1)          # grouped along K
    if model_format == 'fp8':
        if gs != 128:
            raise ValueError(f"FP8 requires group_size == 128, got {gs}")
        return BlockSize(block_in=128, block_out=128)
    raise ValueError(f"Unsupported model_format: {model_format}")
```

No default fallback. Compressed-tensors with `group_size=32` now flows through correctly; previously silently clobbered to 128.

### `Spec` stores `BlockSize`

`TextModelSpec.__init__(..., blocks: BlockSize)`. Stored as `self._blocks`. `_linear(pfx)` passes it:

```python
def _linear(self, pfx):
    return build_linear(self.params, pfx,
                        compute_dtype=self._cpp_dtype(),
                        blocks=self._blocks)
```

## 3. Unified commit path

### `tok_embeddings` is a parameter, not a `LinearWeight`

`ModelWeight` reshape:

```cpp
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)           \
    X(tok_embeddings)
```

`tok_embeddings` was always used as a lookup table (`weights_.tok_embeddings->weight` in `language_model.cc`); no GEMM is performed with it. Wrapping it in `LinearWeight` was scaffolding. `verify()` and `vocab_size` derivation in `model_weight.cc` read the Tensor directly.

### `_commit_linear` / `_commit_tensor` accept kwargs for TP and config overrides

```python
def _commit_linear(self, name, linear, split_side=None, *,
                   tp=None, ranks=None,
                   is_grouped: bool = False,
                   epilogue=_tm.Epilogue.kNone):
    """Commit a Linear to a named LinearWeight child.

    tp / ranks default to the builder's own values; override for top-level
    linears whose TP differs from their parent (e.g. output committed onto
    the root text-model handle).

    is_grouped / epilogue are LinearConfig knobs the caller supplies based
    on the semantics of this particular commit. This primitive stays
    format-agnostic and name-agnostic.
    """

def _commit_tensor(self, name, tensor, split_side=None, *, tp=None, ranks=None):
    """Commit a raw tensor to a named parameter slot on the builder's handle.
    Used for non-linear params (norm, conv1d, scalars, sinks) and for the root
    text-model's tok_embeddings param."""
```

Inside `_commit_linear` all GPU-invariant work is hoisted above the per-GPU loop (packer, `LinearConfig` build, TP-split validation). Post-Section-1 no `alloc(shape, dtype)` call is made from Python — param slots are pre-allocated in the C++ constructor; commit just copies bytes.

`_make_linear_config_for` is name-agnostic and knows nothing about which builder or layer this linear belongs to:

```python
def _make_linear_config_for(self, linear, split_side, tp, *,
                            is_grouped: bool,
                            epilogue) -> _tm.LinearConfig:
    w = linear.tensors['weight']
    in_dim, out_dim = w.shape[0], w.shape[-1]
    if split_side is SplitSide.OUTPUT:
        out_dim //= tp
    elif split_side is SplitSide.INPUT:
        in_dim  //= tp
    cfg = _tm.LinearConfig()
    cfg.input_dim  = in_dim
    cfg.output_dim = out_dim
    cfg.format     = linear.data_format
    cfg.has_bias   = 'bias' in linear.tensors
    cfg.is_grouped = is_grouped
    cfg.epilogue   = epilogue
    return cfg
```

**Callers supply `is_grouped` / `epilogue` explicitly** based on what the commit means in their context. Example from `FfnBuilder.add_ffn`:

```python
def add_ffn(self, w1, w2, w3):
    is_grouped = getattr(self.config, 'fused_moe', False)
    act_type   = _act_type_name(getattr(self.config, 'act_type', 0))

    fused = None
    fused_silu = False
    if w1 is not None and w3 is not None:
        fused, fused_silu = fuse_ffn_linears(w1, w3, self._tp, act_type,
                                             is_moe=is_grouped)

    if fused is not None:
        epilogue = _tm.Epilogue.kGatedSilu if fused_silu else _tm.Epilogue.kNone
        self._commit_linear('w1w3', fused, SplitSide.OUTPUT,
                            is_grouped=is_grouped, epilogue=epilogue)
    else:
        if w1 is not None:
            self._commit_linear('w1', w1, SplitSide.OUTPUT,
                                is_grouped=is_grouped)
        if w3 is not None:
            self._commit_linear('w3', w3, SplitSide.OUTPUT,
                                is_grouped=is_grouped)
    if w2 is not None:
        self._commit_linear('w2', w2, SplitSide.INPUT,
                            is_grouped=is_grouped)
```

Benefits:
- `_commit_linear` and `_make_linear_config_for` are general primitives with no knowledge of the FFN-vs-attention-vs-MLA distinction, no knowledge of name-based dispatch.
- The one place that knows "w1w3 + fuse_silu means kGatedSilu" is `FfnBuilder.add_ffn` — exactly where that semantic lives.
- `FfnConfig.fuse_silu` on the C++ side becomes redundant (the epilogue now lives directly on the `LinearConfig` of `w1w3`). Decide in implementation whether to keep / drop the `FfnConfig.fuse_silu` field.

### `LinearBuilder` and `set_weight` deleted

`lmdeploy/turbomind/deploy/builder/linear.py` is deleted entirely. `make_linear_config` goes with it. `LinearBuilder` exists only as scaffolding for the bypass — in the unified world it serves no purpose.

`TextModelSpec.token_embeds` and `lm_head` become direct commits on the root `TextModelBuilder`:

```python
def token_embeds(self, root, key):
    emb = self._get(key)                              # [vocab, hidden], unpadded
    tp  = self.engine_cfg.attn_tp_size * self.engine_cfg.attn_cp_size
    root._commit_tensor('tok_embeddings', emb,
                        split_side=SplitSide.OUTPUT,  # shard along hidden dim
                        tp=tp, ranks=self._attn_ranks)

def lm_head(self, root, key):
    lin = self._linear(key)                           # Linear in TM [hidden, vocab] layout
    tp  = self.engine_cfg.attn_tp_size * self.engine_cfg.attn_cp_size
    root._commit_linear('output', lin,
                        split_side=SplitSide.OUTPUT,
                        tp=tp, ranks=self._attn_ranks)
```

Key properties:
- `tok_embeddings` is **not** padded along vocab — the embedding lookup never indexes past `vocab_size - 1`, and the only dim that must be TP-divisible is hidden (which it already is, checked by builder machinery). Sharded along hidden via `SplitSide.OUTPUT`.
- `lm_head` **is** padded along vocab — the sampling / penalty / logprob kernels stride on `vocab_size_padded = round_up(vocab_size, tp_size)`. The pad happens on the `Linear` bundle after `self._linear(key)` returns; since lm_head is almost always trivial-format in practice, this is a simple `pad_out_dim(weight, padded_vocab, dim=-1)` over each tensor in the bundle. A guard in `lm_head` asserts trivial format (or falls back to an explicit error) so quantized-lm_head (rare) surfaces rather than silently corrupts.
- `lm_head` uses `self._linear(key)` — the same factory every other linear uses. Quantization, bias, dtype all flow through automatically.
- Tied-embeddings case: caller passes `'model.embed_tokens'` as the prefix. The same tensor is read twice (once as a raw param for `tok_embeddings` — unpadded, once via the trivial normalizer for `output` — padded), committed to independent slots with their respective layouts. No special code path.

### Implication for `ModelWeight::vocab_size` / `vocab_size_padded`

Today `model_weight.cc` derives:
```cpp
vocab_size        = tok_embeddings->weight.shape(0);   // today: padded_vocab (= real_vocab rounded up to tp)
vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);
```

With `tok_embeddings` unpadded, the derivation shifts its source to `output`'s weight (padded LinearWeight in `[hidden, vocab_padded]` layout) and takes the vocab dim from there:

```cpp
vocab_size_padded = output->output_dim * tp_size;       // output_dim is the TP-sharded dim
vocab_size        = /* real vocab from hf_cfg, or retained as-is if tok_embeddings stored unpadded */;
```

Alternative (simpler): keep computing `vocab_size_padded = round_up(vocab_size, tp_size)` and have `vocab_size = hf_cfg.vocab_size` pushed in at construction. Either way, `vocab_size_padded` retains its value so downstream kernels are unaffected. The commit picks one derivation path and documents it.

Per-arch `model()` simplifies:

```python
def model(self):
    root = TextModelBuilder(self._root_handles, self._contexts)
    self.token_embeds(root, 'model.embed_tokens')
    root.norm = self.output_norm(self._norm_key)
    lm_key = 'model.embed_tokens' if self._tie_embeddings else 'lm_head'
    self.lm_head(root, lm_key)
    root.layers = self.layers(self._layer_prefix)
```

Normalize `_embed_key` / `_norm_key` in the base class to prefixes (not full keys with `.weight` suffix).

### `FfnWeight::prepare` post-hoc mutations gone

Both `set_grouped(true)` (for fused MoE experts) and `epilogue = kGatedSilu` (for `w1w3` when fused SiLU) are passed directly on `LinearConfig` at construction. Python callers in `FfnBuilder.add_ffn` supply these via `_commit_linear` kwargs, based on the fusion decision they just made. The `FfnWeight::prepare()` loops that mutated children post-construction are deleted.

## 4. `Linear` dataclass finalization

Covered inline in Section 2. Summary:

- `data_format` mandatory; `__post_init__` guards.
- `preprocess_linear` deleted.
- Fusion helpers (`_can_fuse_w1w3`, `split_qkv`) switch from reading `fmt.block_out` (gone) to `linear.data_format.block_sizes[<out-dim index>]`. For weight tensors in TM `[in, out]` layout, that's `block_sizes[1]`; for scales in their own layout, same index.
- `_ensure_compatible_formats` and `_dequant_linear` continue reading `WeightFormat.dequant`; unchanged.

## 5. Commit plan

Six commits, each independently testable per `AGENTS.md`:

### C1 — `refactor(core): MakeLinearWeightFormat takes (compute_dtype, weight_dtype, block_in, block_out); block_sizes tensor-shape-ordered`

- `src/turbomind/core/data_format.{h,cc}` — signature change, body rewrites `fmt.block_sizes = {block_in, block_out}` (tensor-shape ordered — `block_sizes[0]` always corresponds to the described tensor's `shape[0]`).
- `src/turbomind/core/test_data_format.cc` — six call sites updated.
- `src/turbomind/python/bind.cpp` — binding signature.
- `src/turbomind/kernels/gemm/test/testbed_v3.h` — three call sites, still using old `set_weight_spec` for now.
- `src/turbomind/models/linear_weight.cc` — `set_weight_spec` internally calls new signature. **All `block_sizes[...]` readers updated**: `ResolveLinearPolicy` (and its successor in C2) reads `block_sizes[0]` for the K-dim block size on weight tensors in `[in, out]` layout (previously `block_sizes[1]`). The FP8-native transpose path either reads the correct index post-transpose or swaps the array (see §1 "block-size invariant under transposes") — the commit picks one and documents it.
- `src/turbomind/models/moe_weight.cc` — `LinkLinearExperts` reads `block_sizes` by the current tensor layout (weight's `shape[i]`).
- `lmdeploy/turbomind/deploy/kind_map.py` — the one Python call site (`WeightFormat.to_data_format`, still present from today's dead path — or update `_DEAD` attach at build_linear if it's still there).

Verify: trivial dense + AWQ + FP8 at tp=1. The FP8 case is the critical regression guard for the index flip.

### C2 — `refactor(linear_weight): drop LinearPolicy, carry input/output DataFormats`

- `src/turbomind/models/linear_weight.h` — delete `LinearPolicy`; add `DataFormat input_format, output_format` fields; `DeriveActivationFormats` declaration.
- `src/turbomind/models/linear_weight.cc` — `set_weight_spec` calls `DeriveActivationFormats`; `prepare()` reads `input_format.dtype`.
- `src/turbomind/models/llama/LlamaLinear.cu` — synthesize `QuantDesc` from the three `DataFormat`s at dispatch.
- `src/turbomind/models/moe_weight.cc` — reads `input_format.dtype`.

Verify: trivial + AWQ + FP8 at tp=1.

### C3 — `refactor(linear_weight): LinearConfig carries DataFormat/is_grouped/epilogue; pre-alloc at construction; kill set_weight_spec/set_grouped/preprocess` *(largest atomic commit — C++ + Python together)*

- `src/turbomind/models/linear_weight.{h,cc}` — `LinearConfig` with `format`/`is_grouped`/`epilogue`; `configure(in, out, DataFormat, has_bias)`; param slot pre-allocation; `prepare()` entry assert; `prepare()` swap `block_sizes` on FP8-native transpose; delete `set_weight_spec`, `set_grouped`, `preprocess`, old `configure` overload; delete `weight_format`/`group_size`/`data_type` fields (use `format`).
- `src/turbomind/models/ffn_weight.cc` — delete `set_grouped` loop, `epilogue` assignment.
- `src/turbomind/kernels/gemm/test/testbed_v3.h` — single-phase `configure(in, out, DataFormat, false)`.
- `src/turbomind/python/bind.cpp` — LinearConfig binding updated.
- `lmdeploy/turbomind/deploy/kind_map.py` — full §2 reshape: `FormatKind`, `WeightFormat` with `weight_dtype: _tm.DataType`, factory, `build_linear(..., blocks)`, trivial override.
- `lmdeploy/turbomind/deploy/converter.py` — purge defaults; add `_parse_quant_blocks`; pass `BlockSize` to Spec.
- `lmdeploy/turbomind/deploy/spec.py` — `__init__(..., blocks)`; `_linear` uses it.
- `lmdeploy/turbomind/deploy/source_model/*_spec.py` — direct `build_linear` kwarg updates.
- `lmdeploy/turbomind/deploy/builder/_base.py` — `_commit_linear` populates new cfg fields; `_copy_linear_to_handle` helper; `_make_linear_config_for` helper; no alloc at commit time.

Verify: full matrix — trivial / AWQ / FP8 / MXFP4 / MLA / DeltaNet at tp=1 and tp=2.

### C4 — `refactor(model_weight): tok_embeddings as Tensor parameter`

- `src/turbomind/models/model_weight.{h,cc}` — child → param; `verify()`, `vocab_size` derivation updated.
- `src/turbomind/models/language_model.cc` — drop `->weight` indirection.
- `lmdeploy/turbomind/deploy/spec.py` — `token_embeds(root, key)` uses `_commit_tensor`. No padding.
- `lmdeploy/turbomind/deploy/builder/_base.py` — `_commit_tensor` gains `tp`/`ranks` kwargs.
- `lmdeploy/turbomind/deploy/source_model/*_spec.py` — `model()` one-line change.

Verify: trivial dense at tp=1 and tp=2. Specific assertions:
- `weights_.tok_embeddings` is now `[vocab, hidden / tp]` (unpadded along vocab). Token-id lookups never exceed `vocab - 1`, so padding rows were dead storage; outputs identical.
- `weights_.output->weight` retains `[hidden, vocab_padded / tp]` shape — lm_head still pads, so downstream sampling / penalty / logprob kernels (which stride on `vocab_size_padded`) see no change.
- `ModelWeight::vocab_size_padded` continues to equal `round_up(real_vocab_size, tp_size)` under the new derivation; confirm numerically on a model where `real_vocab_size % tp_size != 0` to catch silent breakage. Check: sampling kernel produces same logits distribution for a fixed prompt.
- Tied-embeddings model: confirm `tok_embeddings` (unpadded) and `output` (padded) produce coherent output — both reads come from the same checkpoint tensor, committed to independent slots.

### C5 — `refactor(deploy): delete LinearBuilder, make_linear_config, set_weight; unify lm_head via self._linear`

- `lmdeploy/turbomind/deploy/builder/linear.py` — deleted.
- `lmdeploy/turbomind/deploy/builder/__init__.py` — exports dropped.
- `lmdeploy/turbomind/deploy/spec.py` — `lm_head(root, key)` via `self._linear(key)` + `_commit_linear`.
- `lmdeploy/turbomind/deploy/source_model/*_spec.py` — `model()` reorg; `_embed_key`/`_norm_key` normalized to prefixes.

Verify: trivial + AWQ at tp=1 and tp=2 (including one tied-embeddings model).

### C6 — `refactor(deploy): Linear.data_format mandatory; delete preprocess_linear; thread data_format through fusion helpers`

- `lmdeploy/turbomind/deploy/linear.py` — `data_format` mandatory; `__post_init__` guards; `preprocess_linear` removed.
- `lmdeploy/turbomind/deploy/builder/attention.py`, `ffn.py`, `mla.py`, `deltanet.py` — every `Linear(...)` constructor threads a real `data_format`; `fmt.block_out` readers switch to `linear.data_format.block_sizes[1]`.

Verify: full matrix at tp=1.

## 6. Verification plan

Per `AGENTS.md`:

| Model class | TP | Primary commit |
| --- | --- | --- |
| Qwen3-small trivial | 1, 2 | C1, C3, C4, C5, C6 |
| AWQ quantized | 1, 2 | C1, C3, C6 |
| FP8 (GLM-4.7-Flash FP8 if available, else other FP8 model in local cache) | 1, 2 | C1, C2, C3 |
| MXFP4 (GPT-OSS) | 1 | C3 |
| MLA (GLM-4.7-Flash via `glm4_moe_lite_spec`) | 2 | C3 |
| DeltaNet (Qwen3.5) | 1 | C3, C6 |
| Compressed-tensors `group_size=32`, if available | 1 | C3 — latent-bug regression guard |

Each run: `scripts/test_turbomind_model.py`, ≥ 128 tokens of coherent output. Gibberish at any commit → halt and bisect. Check `get_gpu_usage` first.

## 7. Summary of deletions and additions

### Deleted

**C++:**
- `struct LinearPolicy`
- `LinearWeight::set_weight_spec`, `set_grouped`, `preprocess`, `configure(int,int,DataType,bool)`
- `LinearWeight::weight_format`, `group_size` public fields; `data_type` public field (read `format.dtype`)
- `LinearConfig::data_type` field
- `FfnWeight::prepare` post-hoc `set_grouped` loop
- `FfnWeight::prepare` post-hoc `epilogue = kGatedSilu` assignment
- `tok_embeddings` as `LinearWeight` child (becomes Tensor param)
- `ResolveLinearPolicy` function (replaced by `DeriveActivationFormats`)
- `LinearWeight::prepare()`'s trivial-default branch (asserted impossible)

**Python:**
- `LinearBuilder` class
- `LinearBuilder.set_weight` method
- `make_linear_config` helper
- `lmdeploy/turbomind/deploy/builder/linear.py` file
- `_DEFAULT_GROUP_SIZES`, `_SUPPORTED_GROUP_SIZES`, `_validate_quant_group_size`
- `Spec._group_size` (replaced by `self._blocks: BlockSize`)
- `WeightFormat.name: str`, `block_in`, `block_out`, `cpp_dtype_name` fields
- `dataclasses.replace(fmt, ...)` runtime format mutation
- `Linear.data_format = None` path (now mandatory)
- `preprocess_linear` function
- `_infer_cpp_linear_dtype` helper

### Added

**C++:**
- `LinearConfig::format: DataFormat`, `is_grouped: bool`, `epilogue: Epilogue` fields
- `LinearWeight::input_format`, `output_format` fields
- `DeriveActivationFormats(const DataFormat&, DataType, int) -> pair<DataFormat, DataFormat>`
- `MakeLinearWeightFormat(compute_dtype, weight_dtype, block_in, block_out)` signature
- Pre-allocation of `LinearWeight` param slots at construction
- `TM_CHECK(format.dtype != DataType{})` at `prepare()` entry
- `block_sizes` swap on FP8-native transpose in `prepare()`

**Python:**
- `FormatKind` enum
- `BlockSize` NamedTuple
- `WeightFormat.make_data_format(compute_dtype, block_in, block_out)` factory
- `WeightFormat.weight_dtype: _tm.DataType | None` field
- `_parse_quant_blocks` in `converter.py`
- `TextModelSpec._blocks: BlockSize` field
- `tp` / `ranks` kwargs on `_commit_linear` and `_commit_tensor`
- `is_grouped` / `epilogue` kwargs on `_commit_linear` (caller-supplied, no name-based dispatch)
- `_copy_linear_to_handle`, `_make_linear_config_for(..., *, is_grouped, epilogue)` helpers in `_base.py`
- `Linear.__post_init__` guards

### Invariants strengthened

1. `DataFormat.block_sizes[i]` always corresponds to the described tensor's `shape[i]`. No hidden convention flip anywhere. Transposes update `block_sizes` accordingly.
2. `LinearConfig` fully describes a `LinearWeight`. No second phase.
3. `Linear.data_format` is the authoritative per-linear storage descriptor. Always present.
4. `group_size` appears only inside `converter.py` during quant-config parse; translated to `BlockSize` at parse time.
5. `LinearWeight` param slots are pre-allocated at construction from the format. Python commit = byte copy only, no shape/dtype declaration.
6. `tok_embeddings` is a parameter (it's a lookup table), not a `LinearWeight`. Unpadded along vocab.
7. `lm_head` / `output` retains vocab padding (required by sampling / penalty / logprob kernels). Padding is applied in Python on the `Linear` bundle returned by `self._linear(key)`, before commit.
8. Every linear — `tok_embeddings` excepted — flows through a single commit primitive (`_commit_linear`).
9. Every LM head, tied or untied, is loaded via `self._linear(prefix)` identically to all other linears.

### Behavior changes visible in tests

1. **Compressed-tensors with `group_size=32` works correctly.** Previously silently clobbered to 128.
2. **Missing `group_size` on a quantized format raises** at `_parse_quant_blocks` instead of silently defaulting to 128.
3. **`byte_size != shard.nbytes` raises** inside `_copy_shard_to_param` (already landed in the earlier commit-simplification spec; re-affirmed here).
4. **`LinearWeight::prepare()` aborts** if called with an unset format (`format.dtype == DataType{}`). Shouldn't fire for any live path.
