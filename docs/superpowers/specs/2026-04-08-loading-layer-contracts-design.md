# Loading Pipeline Layer Contracts Design

Date: 2026-04-08
Scope: Full stack — Python configs, LayerWriter, component processing, commit layer, C++ constructors
Goal: Define inter-layer contracts so the loading pipeline is correct by construction, not by bug-patching.

## Problem Statement

The loading pipeline cleanup (April 8) and TP loading refactor (April 8) introduced a layered architecture with typed configs, a LayerWriter abstraction, and component-major processing. Six bugfix commits (10 distinct issues) revealed that the layer boundaries were under-specified: configs missing protocol methods, TP info lost during navigation, data-dependent fields hardcoded before they were known, fused tensors silently mis-sharded, and C++ constructors silently dropping config fields.

This spec defines contracts between each layer so that violations are caught at the boundary, not discovered in all-model testing.

## Layer Architecture

The loading pipeline has 5 layers, each with one job:

```
┌─────────────────────────────────────────────┐
│ 1. Config Protocol (configs.py)             │  for_rank + to_cpp
├─────────────────────────────────────────────┤
│ 2. LayerWriter (text_model_loader.py)       │  GPU distribution
├─────────────────────────────────────────────┤
│ 3. Component Processing (_process_* methods)│  READ→TRANSFORM→CREATE→COMMIT
├─────────────────────────────────────────────┤
│ 4. Commit Layer (load_context.py)           │  Tensor sharding + C++ handoff
├─────────────────────────────────────────────┤
│ 5. C++ Constructors (weight classes)        │  Config struct → weight init
└─────────────────────────────────────────────┘
```

## Layer 1: Config Protocol (configs.py, module_config.h)

Every config dataclass that flows through `LayerWriter.create_child` must implement two methods:

### `for_rank(rank: int) -> Self`

| Config type | Behavior | Rationale |
|---|---|---|
| TP-split (AttentionConfig, FfnConfig, MoeConfig, DeltaNetConfig) | `return replace(self, tp_rank=rank)` | Module needs per-GPU rank |
| Broadcast (LinearConfig, ModuleListConfig, NormConfig, DecoderLayerConfig) | `return self` | No rank-dependent fields |

Every config class must have `for_rank`. Missing it is a protocol violation that crashes at `LayerWriter.create_child` time.

### `to_cpp() -> _tm.XConfig`

**DataType wrapping.** Never assign raw `int` to a C++ `DataType` field:
```python
cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
```
The pybind11 boundary requires enum objects, not plain ints.

**Type fidelity.** Python field types must match C++ struct field types exactly:
- `str` ↔ `std::string` (e.g., `topk_method`, `scoring_func`)
- `int` ↔ `int`
- `bool` ↔ `bool`
- `float` ↔ `double`

**Completeness.** Every field in the C++ struct must have a corresponding field in the Python dataclass. `to_cpp()` must copy all fields. A missing field means the C++ constructor silently gets a zero/empty default.

## Layer 2: LayerWriter (text_model_loader.py)

### Creation: `create_child(name, config, tp=None, ranks=None)`

For each GPU handle `i`, calls `config.for_rank(ranks[i]).to_cpp()` and creates a C++ child module. Returns a new `LayerWriter` scoped to the created children with rebound `tp`/`ranks`.

TP info is bound at creation time. A child writer inherits parent's tp/ranks unless overridden:
```python
new_tp = tp if tp is not None else self._tp
new_ranks = ranks if ranks is not None else self._ranks
```

### Commit: `commit_linear` / `commit_tensor`

If `split_side` is given, use bound `tp`/`ranks` for sharding. If `split_side` is `None`, broadcast (tp=1).

### No navigation, no TP recovery

The previous design had `_process_raw_tensors` navigating into existing TP-split children (e.g., traversing `"attention.q_norm.weight"` by splitting on `.`). This required TP recovery via `_MOD_TP_ATTR`, which was brittle and non-extensible.

**New design:** Each `_process_*` method owns ALL parameters for its component — both linears and leaf parameters. `_process_raw_tensors` is eliminated. No module tree navigation. No `_MOD_TP_ATTR`.

Parameters live on leaf modules. Each `_process_*` method creates the leaf module and commits the parameter directly, using its own writer which already has the correct TP.

### Initialization timing

`model.gpu_count` may be 0 at `TextModelLoader.__init__` time (GPUs not yet initialized). Per-GPU rank lists must be computed lazily:

```python
def __init__(self, model):
    self._attn_ranks = None
    self._mlp_ranks = None

def _ensure_ranks(self):
    if self._attn_ranks is None:
        self._attn_ranks = [self.model.tp_ranks(gpu)[0]
                            for gpu in range(self.model.gpu_count)]
        self._mlp_ranks = [self.model.tp_ranks(gpu)[1]
                           for gpu in range(self.model.gpu_count)]
```

`_ensure_ranks()` is called at the start of `_load_layer`.

## Layer 3: Component Processing (_process_* methods)

### Lifecycle: READ → TRANSFORM → CREATE → COMMIT

Each `_process_*` method owns the full lifecycle for one component. CREATE must come after TRANSFORM when config has data-dependent fields (`fuse_silu` is determined by `fuse_ffn_linears` which inspects actual weight data).

| Component | Data-dependent field | Lifecycle |
|---|---|---|
| Attention | none | READ → CREATE → COMMIT |
| Dense FFN | `fuse_silu` | READ → TRANSFORM → CREATE → COMMIT |
| MoE experts | `fuse_silu` | READ → TRANSFORM → CREATE → COMMIT |
| Linear attention | none | READ → CREATE → COMMIT |
| Norms | none | CREATE → COMMIT |

### Parameter ownership

Each `_process_*` method owns ALL tensors for its component. `raw_layer_tensors` is eliminated.

**`_process_attention`** owns:
- Linears: from `spec.attn_linears(layer)` — QKV merge, wo
- Parameters: q_norm, k_norm, sinks, q_a_layernorm, kv_a_layernorm — from new `spec.attn_params(layer)`

**`_process_moe`** owns:
- Gate linear, shared_gate, score_correction_bias — from new `spec.moe_params(layer)`
- Expert linears: from `spec.moe_ffn_linears(layer, e)`

**`_process_linear_attn`** owns:
- Linears: from `spec.linear_attn_linears(layer)`
- Parameters: A_log, dt_bias — from new `spec.linear_attn_params(layer)`

### `fuse_ffn_linears` signature

The TP-loading-refactor spec said "drop tp/rank params." In practice, `tp` is kept because `_can_fuse_w1w3` needs it for block-scale alignment checks, and `chunk_linears` needs it for sharding-aware fusion. Only `rank` was dropped (no internal per-rank sharding).

```python
def fuse_ffn_linears(w1, w3, tp, act_type, is_moe) -> (Linear | None, bool)
```

## Layer 4: Commit Layer (load_context.py, transforms.py)

### Sharding-aware fusion (all fusion paths)

All fusion operations produce results that are naively shardable along the output dim. This is the same pattern already used by attention (`merge_qkv_v2`) and linear attention (`fuse_gdn_in_proj`), extended to FFN `chunk_linears`.

**Attention QKV** (`merge_qkv_v2`): reshapes each of Q, K, V as `[K, tp, per_tp]`, concatenates on the inner dim, flattens. Naive output-dim split gives each rank `[Q_ri | K_ri | V_ri]`. Already sharding-aware.

**Linear attention GDN** (`fuse_gdn_in_proj`): same `_tp_interleave_tensor` pattern — reshapes as `[tp, per_tp]`, concatenates, flattens. Already sharding-aware.

**FFN interleave** (`interleave_linears`): alternating elements (`w1[0], w3[0], w1[1], w3[1], ...`) distribute evenly across ranks with a naive split. Already sharding-aware.

**FFN chunk** (`chunk_linears`): currently just `cat([w1, w3], dim=-1)`, NOT sharding-aware. A naive TP=2 split gives rank 0 all of w1 and rank 1 all of w3.

**Fix:** `chunk_linears` takes `tp` and uses the same `_tp_interleave_tensor` pattern:
```python
def chunk_linears(w1, w3, tp=1):
    if tp <= 1:
        # No TP: simple concatenation
        return Linear(tensors={k: cat([w1[k], w3[k]], dim=-1) ...})
    # TP-aware: same _tp_interleave_tensor pattern as merge_qkv_v2
    fused = {}
    for kind in w1.tensors:
        t1, t3 = w1.tensors[kind], w3.tensors[kind]
        d = t1.dim() - 1  # output dim (last)
        # [K, N] -> [K, tp, N/tp]
        r1 = _tp_interleave_tensor(t1, tp, d)
        r3 = _tp_interleave_tensor(t3, tp, d)
        # [K, tp, N/tp] cat on last dim -> [K, tp, 2*N/tp]
        combined = cat([r1, r3], dim=d + 1)
        # [K, tp, 2*N/tp] -> [K, 2*N]
        shape = list(combined.shape)
        fused[kind] = combined.reshape(shape[:d] + [shape[d] * shape[d+1]])
    return Linear(tensors=fused, ...)
```

Each rank gets `[w1_shard_i | w3_shard_i]` — exactly what the C++ chunked layout expects.

### Consequences

With sharding-aware fusion on all paths:
- `fused_count` field on `Linear` is eliminated
- `fused_count` preservation in `commit_linear` is eliminated
- `fused_count`-aware reshaping in `_commit_tensors` is eliminated
- Negative dim comparison bug is eliminated (the special sharding path is gone)
- `commit_linear` and `_commit_tensors` become straightforward: shard along `split_side` dim, no special cases

## Layer 5: C++ Constructors (weight classes)

### Delegation completeness

When a typed config constructor delegates to a positional constructor, config fields NOT accepted by the delegate must be set explicitly after delegation:

```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : FfnWeight(cfg.hidden_dim, cfg.inter_size, cfg.has_bias,
                cfg.tp_size, cfg.tp_rank, cfg.data_type,
                static_cast<ActivationType>(cfg.act_type), cfg.fuse_silu)
{
    if (cfg.fused_moe) {
        set_fused_moe(true);  // not in positional constructor
    }
}
```

After writing a typed config constructor, verify every config struct field is consumed. An unconsumed field means silent default behavior.

### Direct assignment over vector indexing

When the config carries a scalar value that the legacy code stored as a vector element, assign the scalar directly:

```cpp
// WRONG: vector is size 1, indexing by layer_id > 0 is OOB
expert_num_ = moe_param_.expert_num[layer_id_];

// CORRECT
expert_num_ = cfg.expert_num;
```

### Type fidelity (C++ side)

Mirrors the Python-side rule:
- `std::string` for `topk_method`, `scoring_func` — assign directly from `cfg.topk_method`, do not call `std::to_string()`
- `DataType` for `data_type` — pybind11 passes enum objects, no conversion needed on C++ side

## Spec Interface Changes

### Eliminated

- `raw_layer_tensors(layer)` — removed entirely

### New methods on TextModelSpec

```python
def attn_params(self, layer: int) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
    """Return {name: (tensor, split_side)} for non-linear attention parameters.
    
    Examples: q_norm, k_norm, sinks, q_a_layernorm, kv_a_layernorm.
    """
    return {}

def moe_params(self, layer: int) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
    """Return {name: (tensor, split_side)} for non-expert MoE parameters.
    
    Examples: gate weight/bias, shared_gate, score_correction_bias.
    """
    return {}

def linear_attn_params(self, layer: int) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
    """Return {name: (tensor, split_side)} for non-linear linear-attention parameters.
    
    Examples: A_log, dt_bias.
    """
    return {}
```

Each returns a flat dict of `{param_name: (tensor, split_side)}`. The `_process_*` method creates the leaf module and commits the parameter. No dotted paths, no navigation.

## Summary of Bug Fixes Addressed

| # | Bug | Layer boundary | This spec's fix |
|---|---|---|---|
| 1 | `data_type` raw int in `to_cpp()` | 1→5 | DataType wrapping rule |
| 2 | `topk_method`/`scoring_func` wrong type | 1↔5 | Type fidelity rule |
| 3 | `expert_num` vector OOB | 5 | Direct assignment rule |
| 4 | `fused_moe` not forwarded | 5 | Delegation completeness rule |
| 5a | Ranks empty at init time | 2 | Lazy initialization rule |
| 5b | `fused_count` lost in commit_linear | 4 | Sharding-aware fusion (fused_count eliminated) |
| 5c | Negative dim comparison | 4 | Sharding-aware fusion (special path eliminated) |
| 6a | `LinearConfig` missing `for_rank` | 1→2 | `for_rank` protocol rule |
| 6b | `_process_raw_tensors` wrong TP | 2→3 | Eliminated (parameter ownership) |
| 6c | MoE expert CREATE before TRANSFORM | 3 | READ→TRANSFORM→CREATE→COMMIT ordering rule |

## Files Changed

### Modified files (Python)
- `lmdeploy/turbomind/deploy/configs.py` — `LinearConfig.for_rank`, DataType wrapping in all `to_cpp()`
- `lmdeploy/turbomind/deploy/text_model_loader.py` — Eliminate `_process_raw_tensors` and `_MOD_TP_ATTR`, move params into `_process_*` methods
- `lmdeploy/turbomind/deploy/transforms.py` — `chunk_linears` takes `tp`, produces TP-interleaved output; remove `fused_count` from `Linear`
- `lmdeploy/turbomind/deploy/linear.py` — Remove `fused_count` field from `Linear`
- `lmdeploy/turbomind/deploy/load_context.py` — Remove `fused_count` sharding logic from `_commit_tensors` and `commit_linear`
- `lmdeploy/turbomind/deploy/spec.py` — Add `attn_params()`, `moe_params()`, `linear_attn_params()`; remove `raw_layer_tensors()`
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — Migrate raw_layer_tensors → attn_params + moe_params
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — Migrate raw_layer_tensors → attn_params + moe_params + linear_attn_params
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — Migrate raw_layer_tensors → attn_params + moe_params
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — Migrate raw_layer_tensors → attn_params + moe_params

### Modified files (C++)
- `src/turbomind/core/module_config.h` — Fix `topk_method`/`scoring_func` to `std::string`
- `src/turbomind/models/ffn_weight.cc` — Set `fused_moe` after delegation
- `src/turbomind/models/moe_weight.cc` — Direct `expert_num` assignment, string assignment for topk_method/scoring_func

### Unchanged
- `src/turbomind/python/bind.cpp` — No changes needed
- `src/turbomind/core/module.h/cc` — No changes needed
- All other weight class files
