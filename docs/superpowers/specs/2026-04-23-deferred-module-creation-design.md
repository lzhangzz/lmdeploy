# Deferred Module Creation in TurboMind

## Problem

`Builder` creates its C++ module (`_tm.create_module(cfg)`) lazily on the
first commit — today, the side effect of `_commit_linear` /
`_commit_tensor` / `__setattr__` calling `_ensure_handles()`.  This has
three consequences:

- **Config is frozen too early.** Sharding / padding / fusion math in the
  Python builders runs during the commit sequence.  Any config field whose
  final value depends on that math (e.g. `FfnConfig.inter_size` after
  `_pad_ffn_for_tp`, `AttentionConfig.kv_head_num` after
  `repeat_kv_for_tp`) is captured from the pre-math config because the
  C++ module already exists by the time the Python math finishes updating
  the field.
- **C++ ends up recovering the truth from tensor shapes.**
  `FfnWeight::prepare()` computes `inter_size_ = w1w3->output_dim / 2`
  (or `= w1->output_dim`) and `AttentionWeight::prepare()` computes
  `kv_head_num = (local_total * tp_size - q_parts * head_num) / 2` by
  reading the sharded weight's output dim.  These are error-prone
  reconstructions because they duplicate — in a different language, from
  a different representation — logic Python already ran.
- **C++ re-runs identical padding logic.**
  `ModelWeight::prepare()` does `vocab_size_padded = round_up(vocab_size,
  tp_size)` even though Python already padded `lm_head` to exactly that
  value in `TextModelBuilder.add_lm_head`.

The existing lazy trigger "works" only because every spec happens to run
all config-mutating math before the first commit.  Introducing a new
commit anywhere earlier silently breaks config fidelity.

## Goals

1. Defer every C++ `create_module` call until an explicit `build()` step.
   All weight processing (padding, fusion, TP repeat) may mutate `self.config`
   freely before `build()`.
2. Make `build()` a one-way transition.  After `build()`, the Builder is
   frozen — no more commits, no more attachments, no more attribute writes.
3. Push authoritative values from Python into the config before
   `create_module` fires, and delete the corresponding C++-side recovery
   blocks.
4. Express attachment with a type-safe wrapper (`BuiltModule`).  A
   `parent.child = ...` assignment that would accept an unbuilt builder
   is a hard `TypeError` at the assignment site, not a silent runtime
   surprise.

## Design

### Builder lifecycle — `build()` returns a `BuiltModule`

`builder/_base.py` grows a `BuiltModule` wrapper and restructures
`Builder` around explicit `build()`.

```python
class BuiltModule:
    """Opaque handle bundle returned by Builder.build().

    The only legitimate RHS for parent.child = ... assignments.  Holds
    one C++ module handle per GPU context.
    """
    __slots__ = ('handles',)

    def __init__(self, handles: list):
        self.handles = list(handles)

    def __iter__(self):
        return iter(self.handles)

    def __len__(self):
        return len(self.handles)


class Builder:
    def __init__(self, config, contexts, tp=1, ranks=None):
        object.__setattr__(self, '_contexts', contexts)
        object.__setattr__(self, '_tp', tp)
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, 'config', config)
        object.__setattr__(self, '_pending_linears', {})
        object.__setattr__(self, '_pending_tensors', {})
        object.__setattr__(self, '_pending_children', {})   # name -> list[_tm.Module]
        object.__setattr__(self, '_handles', None)
        object.__setattr__(self, '_built', False)

    # ---- staging ----

    def _commit_linear(self, name, linear, split_side=None, model_dtype=None):
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        self._pending_linears[name] = (linear, split_side, model_dtype)

    def _commit_tensor(self, name, tensor, split_side=None, model_dtype=None):
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        self._pending_tensors[name] = (tensor, split_side, model_dtype)

    # ---- attachment: everything freezes post-build ----

    def __setattr__(self, name, value):
        if self._built:
            raise RuntimeError(
                f"{type(self).__name__} is built; cannot assign {name!r}")
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}.{name}: assign .build() output "
                f"(BuiltModule), not the Builder itself")
        if isinstance(value, BuiltModule):
            self._pending_children[name] = value.handles
            return
        object.__setattr__(self, name, value)

    def __setitem__(self, index, value):
        if self._built:
            raise RuntimeError(
                f"{type(self).__name__} is built; cannot set index {index}")
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}[{index}]: call .build() first")
        assert isinstance(value, BuiltModule), (
            f"{type(self).__name__}[{index}] requires a BuiltModule")
        self._pending_children[str(index)] = value.handles

    # ---- build: one-way transition ----

    def build(self) -> BuiltModule:
        if self._built:
            return BuiltModule(self._handles)       # idempotent; no side effects
        self._create_handles()
        object.__setattr__(self, '_built', True)    # bypass frozen-check path
        for name, (lin, side, mdt) in self._pending_linears.items():
            self._apply_linear(name, lin, side, mdt)
        for name, (t, side, mdt) in self._pending_tensors.items():
            self._apply_tensor(name, t, side, mdt)
        for name, child_handles in self._pending_children.items():
            self._attach_handles(name, child_handles)
        return BuiltModule(self._handles)

    def _create_handles(self):
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                cfg = self._cfg_for_rank(i)
                handles.append(_tm.create_module(cfg))
        object.__setattr__(self, '_handles', handles)

    def _cfg_for_rank(self, gpu_idx: int):
        """Clone self.config and set per-rank tp_rank if applicable."""
        if self._tp > 1 and hasattr(self.config, 'tp_rank'):
            cfg = self.config.clone()
            cfg.tp_rank = self._ranks[gpu_idx]
            return cfg
        return self.config

    def _attach_handles(self, name, child_handles):
        for i, (parent_h, child_h) in enumerate(zip(self._handles, child_handles)):
            with self._contexts[i]:
                parent_h.add_child_raw(name, child_h)

    # ---- apply (drained during build) ----

    def _apply_linear(self, name, linear, split_side, model_dtype):
        """Body of today's _commit_linear from the 'GPU-invariant preparation'
        comment onward.  Creates the LinearWeight child via create_child,
        packs tensors via fmt.pack, shards per rank, copies into the C++ slot.
        No logic changes — only relocation."""
        ...    # identical to current _commit_linear body

    def _apply_tensor(self, name, tensor, split_side, model_dtype):
        """Body of today's _commit_tensor.  Shards per rank and copies via
        _copy_shard_to_param.  No logic changes."""
        ...    # identical to current _commit_tensor body
```

Three invariants the design enforces:

1. **Built is terminal.** After `build()`, the Builder is inert:
   `builder.x = anything` → `RuntimeError`; `builder[i] = ...` →
   `RuntimeError`; `builder._commit_*(...)` → `AssertionError`; subclass
   helpers (`add_qkv_proj`, `add_ffn`, `add_lm_head`, …) funnel through
   `_commit_*` and therefore also fail.  `build()` itself is idempotent —
   calling it twice returns a fresh `BuiltModule` wrapping the same
   handles, with no side effects.
2. **Only `BuiltModule` crosses an assignment.** A raw Builder on the RHS
   is a `TypeError` at the assignment site.  A random Python list /
   scalar falls through to `object.__setattr__` (treated as a plain
   attribute; preserves today's semantics for writes that aren't meant
   to attach a child).
3. **Internal state mutations bypass `__setattr__`.** `build()`,
   `_create_handles()`, and the constructor use `object.__setattr__`
   to write `_built`, `_handles`, etc., so freezing never blocks internal
   transitions.  This is the convention already used by today's code.

### Authoritative Python config + C++ deletions

Three C++ files have "recover-from-shape" blocks that this refactor
removes, replacing them with Python-side updates before `build()`.

#### `src/turbomind/models/ffn_weight.cc`

Delete the shape-derived `inter_size_` recovery in `prepare()`:

```cpp
// DELETE:
if (w1w3) {
    inter_size_ = w1w3->output_dim / 2;
} else if (w1) {
    inter_size_ = w1->output_dim;
}
```

Change the constructor so `inter_size_` is per-rank from the authoritative
global value in `cfg`:

```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , inter_size_{cfg.inter_size / cfg.tp_size}   // was: inter_size_{cfg.inter_size}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu}
    , is_fused_moe_{cfg.fused_moe}
{}
```

The `prepare()` method keeps its remaining duties unchanged: the
`epilogue = kGatedSilu` assignment when `is_fused_silu_` is on, the MoE
`set_grouped(true)` propagation, and the `Module::prepare()` recursion.

Python authority — in `lmdeploy/turbomind/deploy/builder/ffn.py`,
`FfnBuilder.add_ffn` pushes the padded global right after `_pad_ffn_for_tp`
(before any commit):

```python
def add_ffn(self, w1, w2, w3):
    w1, w2, w3 = _pad_ffn_for_tp(w1, w2, w3, self._tp)
    self.config.inter_size = w1.tensors['weight'].size(-1)   # NEW: authoritative global
    ...    # remainder unchanged (fuse_w1w3, commit w1w3/w1/w3, commit w2)
```

#### `src/turbomind/models/attention_weight.cc`

Delete the entire kv_head_num recovery block from `prepare()`:

```cpp
// DELETE:
if (!w_qkv) {
    if (kv_lora_rank > 0 && kv_head_num < tp_size) {
        kv_head_num = tp_size;
    }
    return;
}
int local_total = w_qkv->output_dim / head_dim;
int q_parts     = attn_output_gate ? 2 : 1;
kv_head_num = (local_total * tp_size - q_parts * head_num) / 2;
```

`prepare()` collapses to just `Module::prepare();`.  The constructor is
unchanged — `kv_head_num(cfg.kv_head_num)` stays.

Python authority, two call sites:

1. `lmdeploy/turbomind/deploy/builder/attention.py` —
   `AttentionBuilder.add_qkv_proj` updates the cloned config after
   `repeat_kv_for_tp`:

   ```python
   def add_qkv_proj(self, q, k, v, *, gate=None):
       q, k, v, gate = dequant_mixed(q, k, v, gate, data_type=self.config.data_type)
       k, v = repeat_kv_for_tp(k, v, tp=self._tp, head_dim=self.config.head_dim)
       self.config.kv_head_num = _infer_heads(k, self.config.head_dim)   # NEW
       merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
       self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                           model_dtype=self.config.data_type)
   ```

2. `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py::attn`
   — the MLA `kv_head_num < tp_size` nudge moves from C++ to Python,
   applied to the cloned cfg before `MLABuilder` is constructed:

   ```python
   def attn(self, pfx, layer):
       cfg = self._attn_cfg.clone()
       if cfg.kv_lora_rank > 0 and cfg.kv_head_num < self.engine_cfg.attn_tp_size:
           cfg.kv_head_num = self.engine_cfg.attn_tp_size   # NEW (moved from C++)
       builder = MLABuilder(cfg, self._contexts,
                            tp=self.engine_cfg.attn_tp_size,
                            ranks=self._attn_ranks)
       # ... rest unchanged
   ```

#### `src/turbomind/models/model_weight.cc`

Replace the `round_up` duplication in `prepare()` with a single read of
the sharded `output` child's dim — no separate C++ padding logic, no new
Python-C++ binding:

```cpp
// OLD:
// vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);

// NEW:
vocab_size_padded = TM_CHECK_NOTNULL(output)->output_dim * tp_size;
```

`TM_CHECK_NOTNULL` aborts with a clear message if `output` is missing —
which would indicate a spec that skipped `add_lm_head`.  Every existing
spec calls `add_lm_head`, so the check is defensive, not a runtime risk.
The engine invokes `ModelWeight::prepare()` (via `weights_[index]->prepare()`
in `turbomind.cc`) only after `spec.model()` — including its final
`root.build()` — returns, so the `output` child has been fully
created by `TextModelBuilder`'s drain before `prepare()` reads its
`output_dim`.

The rest of `ModelWeight::prepare()` (walking children, deriving
`data_type` / `hidden_units` / `head_dim` / `kv_head_num` from the first
full-attention layer, `vocab_size` / `embedding_size` / `num_layer` from
shapes / container size, `layer_types` from per-layer dispatch) stays
untouched — it isn't the error-prone "duplicate padding logic" the
refactor targets.

#### Python → C++ authority contract (after this PR)

| Config / derived field          | Python authority (where)                                         | C++ after-PR                                                   |
| ------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| `FfnConfig.inter_size`          | `FfnBuilder.add_ffn` (padded-global, post `_pad_ffn_for_tp`)     | `inter_size_{cfg.inter_size / cfg.tp_size}` in constructor     |
| `AttentionConfig.kv_head_num` (non-MLA) | `AttentionBuilder.add_qkv_proj` (post `repeat_kv_for_tp`) | Stored from cfg; `prepare()` recovery deleted                  |
| `AttentionConfig.kv_head_num` (MLA)     | `Glm4MoeLiteSpec.attn` on cloned cfg                       | Stored from cfg; same delete                                   |
| `ModelWeight.vocab_size_padded` | `TextModelBuilder.add_lm_head` (pad tensor to `round_up(vocab, tp)`) | `TM_CHECK_NOTNULL(output)->output_dim * tp_size` in `prepare()` |

### TextModelBuilder (root)

The root is special in two ways that survive the refactor: its C++
handles already exist (created by the TurboMind runtime before Python
gets them), and it has no config object.  The new `_create_handles`
hook lets us keep the lifecycle uniform with a tiny override:

```python
class TextModelBuilder(Builder):
    def __init__(self, handles, contexts, *, tp, ranks, vocab_size, data_type):
        super().__init__(config=None, contexts=contexts, tp=tp, ranks=ranks)
        object.__setattr__(self, '_vocab_size', vocab_size)
        object.__setattr__(self, '_data_type', data_type)
        object.__setattr__(self, '_handles', list(handles))

    def _create_handles(self):
        # Root handles are pre-populated by the runtime; nothing to create.
        assert self._handles is not None

    # add_token_embeds / add_lm_head: unchanged bodies — they already
    # route through _commit_tensor / _commit_linear, which now stage.
```

`add_token_embeds` and `add_lm_head` keep their current bodies —
including the `pad_out_dim` call in `add_lm_head` that populates
`output->output_dim`, which `ModelWeight::prepare()` now relies on for
`vocab_size_padded`.

The existing `TextModelBuilder.__init__` `object.__setattr__` avalanche
(nine calls) collapses to one `super().__init__` plus three fields.

### Spec subclass migration

Single uniform rule: **every factory that returns a builder ends with
`return m.build()`**.  Assignments always see `BuiltModule` on the RHS;
child builds cascade naturally into staged parent attachments.

#### Base class (`lmdeploy/turbomind/deploy/spec.py`)

One change:

```python
def norm(self, weight, *, dim=None, data_type=None):
    cfg = make_norm_config(
        dim=dim if dim is not None else weight.shape[-1],
        data_type=data_type if data_type is not None else self._cpp_dtype(),
        norm_eps=self._norm_eps,
    )
    m = NormBuilder(cfg, self._contexts)
    m.set_weight(weight)
    return m.build()      # was: return m
```

`qk_norm` forwards to `norm` and needs no change.  The
`Qwen3_5Spec.norm` override (`return super().norm(...)`) also needs no
change — `super().norm` returns a `BuiltModule`, so the override
implicitly does too.

#### Per-module factory pattern

Example — `Qwen3TextSpec.attn`:

```python
def attn(self, pfx, layer):
    q = self._linear(f'{pfx}.q_proj')
    k = self._linear(f'{pfx}.k_proj')
    v = self._linear(f'{pfx}.v_proj')
    o = self._linear(f'{pfx}.o_proj')

    q = reorder_rotary_emb(q, self._head_dim, self._rope.dim, resolver=self._resolver)
    k = reorder_rotary_emb(k, self._head_dim, self._rope.dim, resolver=self._resolver)

    cfg = self._attn_cfg.clone()
    attn = AttentionBuilder(cfg, self._contexts,
                            tp=self.engine_cfg.attn_tp_size,
                            ranks=self._attn_ranks)

    attn.add_qkv_proj(q, k, v)             # stages w_qkv; mutates cfg.kv_head_num
    attn.add_o_proj(o)                      # stages wo
    attn.q_norm = self.qk_norm(self._get(f'{pfx}.q_norm.weight'),
                               head_dim=self._head_dim, rope_dim=self._rope.dim)
    attn.k_norm = self.qk_norm(self._get(f'{pfx}.k_norm.weight'),
                               head_dim=self._head_dim, rope_dim=self._rope.dim)
    return attn.build()                     # NEW
```

#### Container pattern (MoE)

```python
def moe(self, pfx, layer):
    if self.num_experts(layer) <= 0:
        return None
    cfg = self._moe_cfg.clone()
    cfg.layer_id   = layer
    cfg.expert_num = self._expert_nums[layer]
    cfg.inter_size = self._expert_inter_size

    m = MoeBuilder(cfg, self._contexts,
                   tp=self.engine_cfg.mlp_tp_size,
                   ranks=self._mlp_ranks)
    m.add_gate('gate', self._linear(f'{pfx}.gate'), model_dtype=self._cpp_dtype())

    experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
    for e in range(self.num_experts(layer)):
        experts[str(e)] = self.ffn(f'{pfx}.experts.{e}', layer,
                                   inter_size=self._expert_inter_size, fused_moe=True)
    m.experts = experts.build()             # NEW: experts built before attachment
    return m.build()                        # NEW
```

#### `layers()` and `model()`

```python
def layers(self, pfx):
    layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
    for i in layer_progress(self._num_layer):
        d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
        d.attention_norm = self.norm(self._get(f'{pfx}.{i}.input_layernorm.weight'))
        d.attention      = self.attn(f'{pfx}.{i}.self_attn', i)
        d.ffn_norm       = self.norm(self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))
        if self.num_experts(i) > 0:
            d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
        else:
            d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
        layers[str(i)] = d.build()          # NEW
    return layers.build()                    # NEW


def model(self):
    ec = self.engine_cfg
    root = TextModelBuilder(self._root_handles, self._contexts,
                            tp=ec.attn_tp_size * ec.attn_cp_size,
                            ranks=self._model_tp_ranks,
                            vocab_size=self._vocab_size,
                            data_type=self._cpp_dtype())
    root.add_token_embeds(self._get(self._embed_key))
    root.norm = self.norm(self._get(self._norm_key))
    lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
    root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
    root.layers = self.layers(self._layer_prefix)
    root.build()                             # NEW: final drain
```

#### Per-spec change checklist

| Factory                                                      | Change                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `TextModelSpec.norm`                                         | `return m` → `return m.build()`                              |
| `attn` / `ffn` / `moe` / `linear_attn` / helpers (`_packed_moe_ffn`, `_moe_expert_ffn`) in every spec | add `.build()` at return                                     |
| `moe` internal `experts` ModuleList                          | `m.experts = experts.build()`                                |
| `layers`                                                     | `layers[str(i)] = d.build()` per layer; `return layers.build()` |
| `model` (all 4 specs)                                        | `root.build()` as last line (replaces the implicit drain of today's first-commit trigger) |
| `FfnBuilder.add_ffn`                                         | `self.config.inter_size = w1.tensors['weight'].size(-1)` after `_pad_ffn_for_tp` |
| `AttentionBuilder.add_qkv_proj`                              | `self.config.kv_head_num = _infer_heads(k, self.config.head_dim)` after `repeat_kv_for_tp` |
| `Glm4MoeLiteSpec.attn` (MLA)                                 | nudge `cfg.kv_head_num = self.engine_cfg.attn_tp_size` on cloned cfg before `MLABuilder(...)` |

#### None-return paths

`moe()` returns `None` when `num_experts == 0`, and
`Qwen3_5Spec.ffn(..., optional=True)` returns `None` when all three
projections are absent.  `parent.x = None` is neither `Builder` nor
`BuiltModule`, so it falls through to `object.__setattr__` — sets a
Python attribute, no attachment.  Current specs route around `None`
with `if`-guards; that code is unchanged.

### Error semantics

| Mistake                                              | Result                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `d.x = unbuilt_builder`                              | `TypeError: ...: assign .build() output (BuiltModule), not the Builder itself`                    |
| `parent[i] = unbuilt_builder`                        | `TypeError: ...[i]: call .build() first`                                                          |
| `built_builder.x = anything`                         | `RuntimeError: ... is built; cannot assign 'x'`                                                    |
| `built_builder[i] = anything`                        | `RuntimeError: ... is built; cannot set index i`                                                   |
| `built_builder.add_qkv_proj(...)` (or any `add_*`)   | `AssertionError: ... is built; commit '<name>' rejected` (routes through `_commit_linear`)        |
| `parent[i] = <plain list / other type>`              | `AssertionError: ...[i] requires a BuiltModule`                                                   |
| `d.x = <plain non-Builder non-BuiltModule>`          | Falls through to `object.__setattr__` — treated as a Python attribute; no attachment (by design) |
| `m.build(); m.build()`                               | Idempotent: returns a fresh `BuiltModule` wrapping the same handles, no side effects             |

Duplicate `_commit_linear` with the same name before build: the stage
dict overwrites — **last wins**.  This matches today's "get-or-create"
behavior where the second call would re-allocate the same child.  No
spec does this today; no assert added.

### What gets deleted from `builder/_base.py`

- `_handles_created` field (replaced by `_built`)
- `_children` dict (replaced by `_pending_children`)
- `_ensure_handles()` method (replaced by `build()` + `_create_handles()`)
- The `TextModelBuilder.__init__` `object.__setattr__` avalanche
  (replaced by `super().__init__(config=None, ...)` plus three lines)
- The lazy-creation side effect in `_commit_linear` / `_commit_tensor`
  (the first-line `self._ensure_handles()` calls both go away)

### What stays untouched

- `_apply_linear` / `_apply_tensor` bodies are literal moves of today's
  `_commit_linear` / `_commit_tensor` bodies from the "GPU-invariant
  preparation" comment onward — no logic changes, only relocation.
- `_shard`, `_copy_shard_to_param`, `_cast_shard_for_tm`,
  `_infer_compute_dtype`, dtype maps, `transform_output_dim`,
  `transform_input_dim`.
- `SplitSide`, `_SPLIT_SIDE_TO_DIM`, `AttentionBuilder._PARAM_TP_RULES`.
- `_ensure_compatible_formats`, `_dequant_linear`.
- `fuse_w1w3`, `fuse_qkv`, `fuse_gdn`, `fold_kv_b`, `pad_wo_input`,
  `_pad_ffn_for_tp`, `_should_fuse_silu`, `_can_fuse_w1w3`.
- C++ bindings (`create_module`, `create_child`, `add_child_raw`,
  `Module` class, `Param.alloc`, etc.).
- `TextModelLoader` and `BaseOutputModel.export` / `export_iter` — they
  call `spec.model()` just as before; `model()`'s new `root.build()`
  final line makes the load complete.

## Verification

Per `AGENTS.md`, no unit tests exist for this subsystem; verification
is end-to-end via `scripts/test_turbomind_model.py` with ≥128 tokens
of meaningful response per run.

### Model matrix

| Model                              | Exercises                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------- |
| Qwen3 dense (trivial)              | Basic deferred build path + `root.build()` drain                           |
| Qwen3-MoE                          | `MoeBuilder` + staged `experts` ModuleList                                |
| Qwen3.5 (linear-attn variant)      | `DeltaNetBuilder`, zero-centered norm override, MoE shared expert         |
| GPT-OSS (mxfp4)                    | Quantized packed experts, sliding-window attention                        |
| GLM-4 MoE Lite                     | `MLABuilder` — exercises MLA `kv_head_num = tp_size` nudge moved to Python |
| Qwen3 + AWQ                        | Quantized w1w3 fusion; exercises `FfnBuilder` `inter_size` update          |
| Qwen3 + FP8 / GPTQ / compressed-tensors | Other quantized paths; regression coverage for fusion + commit        |

### TP coverage

At least two of the above must be run with `tp=2` (e.g., Qwen3 dense +
GPT-OSS).  This is the only path that exercises the `repeat_kv_for_tp`
and `_pad_ffn_for_tp` config mutations AND the C++ constructor's
`inter_size / tp_size` division.  Without a tp>1 run, the Python → C++
authority contract for FFN/Attention is untested.

### Regression check

Before/after token-by-token comparison on the same prompt for Qwen3
dense + GPT-OSS at `tp=2`.  The refactor is a pure restructure; any
numerical drift signals a bug in the config-field updates or the
relocated commit bodies.

### Manual smoke tests for the strictness

Not automated; run once:

- **Forgot `.build()`**: locally change any spec factory to `return m`
  instead of `return m.build()`, run the test, confirm
  `TypeError: ...: assign .build() output (BuiltModule), not the Builder itself`.
- **Post-build commit**: call `attn.add_o_proj(o)` after `attn.build()`
  in a scratch script; confirm
  `AssertionError: ... is built; commit 'wo' rejected`.

## Migration

Single atomic PR — the Python lifecycle change, the spec rewrites, and
the C++ deletions all depend on each other and can't land in halves
without keeping two lifecycles live.  Rollout order within the PR:

1. **`lmdeploy/turbomind/deploy/builder/_base.py`**: add `BuiltModule`,
   introduce `_pending_*` dicts and `_built`, add `build()` /
   `_create_handles()` / `_cfg_for_rank()` / `_attach_handles()`, split
   commit bodies into `_commit_*` (staging) and `_apply_*` (drained
   during build), rewrite `__setattr__` / `__setitem__`, remove
   `_ensure_handles` / `_handles_created` / `_children`.  Rewrite
   `TextModelBuilder.__init__` via `super().__init__(config=None, ...)`
   and the `_create_handles` override.
2. **`lmdeploy/turbomind/deploy/builder/attention.py`**: add the
   `self.config.kv_head_num = _infer_heads(k, self.config.head_dim)`
   line to `add_qkv_proj`.
3. **`lmdeploy/turbomind/deploy/builder/ffn.py`**: add the
   `self.config.inter_size = w1.tensors['weight'].size(-1)` line to
   `add_ffn`.
4. **`lmdeploy/turbomind/deploy/spec.py`**: change `TextModelSpec.norm`
   to `return m.build()`.
5. **Spec subclasses** (four files):
   `source_model/qwen3_spec.py`, `source_model/qwen3_5_spec.py`,
   `source_model/gpt_oss_spec.py`, `source_model/glm4_moe_lite_spec.py`
   — add `.build()` at every factory return, `d.build()` in `layers`,
   `experts.build()` inside `moe`, `root.build()` at end of `model()`.
   Add the MLA `cfg.kv_head_num = self.engine_cfg.attn_tp_size` nudge
   in `glm4_moe_lite_spec.py::attn`.
6. **`src/turbomind/models/ffn_weight.cc`**: delete the shape-derived
   `inter_size_` block in `prepare()`; change the constructor to
   `inter_size_{cfg.inter_size / cfg.tp_size}`.
7. **`src/turbomind/models/attention_weight.cc`**: collapse `prepare()`
   to just `Module::prepare();`.  The entire `if (!w_qkv) { ... } ...
   kv_head_num = ...` block is deleted.
8. **`src/turbomind/models/model_weight.cc`**: replace
   `vocab_size_padded = round_up(...)` with
   `vocab_size_padded = TM_CHECK_NOTNULL(output)->output_dim * tp_size;`.

Build (`ninja` from `build/`) and run the verification matrix.
