# Text-model root commits: `add_token_embeds` / `add_lm_head` on `TextModelBuilder`

Date: 2026-04-22

Scope: root-level weight commits (`tok_embeddings`, `output`) and the
TP/ranks contract on `Builder.__init__`. Touches both Python and C++.

- `src/turbomind/models/model_weight.{h,cc}`
- `src/turbomind/models/language_model.cc`
- `lmdeploy/turbomind/deploy/builder/_base.py`
- `lmdeploy/turbomind/deploy/builder/linear.py` (deleted)
- `lmdeploy/turbomind/deploy/builder/__init__.py`
- `lmdeploy/turbomind/deploy/spec.py`
- `lmdeploy/turbomind/deploy/source_model/{qwen3,qwen3_5,glm4_moe_lite,gpt_oss}_spec.py`

## Motivation

`spec.py` lines 177–203 (`TextModelSpec.token_embeds` /
`TextModelSpec.lm_head`) concentrate five co-existing antipatterns:

1. **Parallel commit lifecycle.** `LinearBuilder` +
   `make_linear_config` + `set_weight` is a bypass that creates a
   standalone `_tm.create_module(LinearConfig)` on its own and attaches
   the result as a root child via `add_child_raw`. Every other linear
   flows through `build_linear → Linear → _commit_linear →
   handle.create_child(name, cfg)`. Two commit pipelines for the same
   concept.

2. **Spec calls `_commit_*` directly.** When the bypass is removed, the
   naïve fix is to have spec call `root._commit_tensor(...)` /
   `root._commit_linear(...)`. But the repo-wide convention is that
   spec hands tensors / `Linear` bundles to **public** builder methods
   (`add_qkv_proj`, `add_ffn`, `add_gate`, `set_weight`, …) and the
   builder owns the commit. Spec never touches `_commit_*`.

3. **`lm_head` reads raw tensor instead of flowing through
   `self._linear`.** The existing code uses `self._get(key).t()` + manual
   `make_linear_config`, skipping the normalizer / format detection /
   `data_format` that every other linear gets. A quantized `lm_head` (rare
   but possible) is silently unsupported.

4. **`tok_embeddings` is a `LinearWeight` child but used as a lookup
   table.** Only `->weight` is ever read (confirmed in
   `src/turbomind/models/language_model.cc`). Wrapping it in `LinearWeight`
   is scaffolding; it should be a `Tensor` parameter on `ModelWeight`.
   Additionally, the current code pads vocab — but an embedding lookup
   never indexes past `vocab_size - 1`, so the pad is dead storage.

5. **TP is computed as `attn_tp_size * attn_cp_size` for both methods,
   which is wrong.** `self._attn_ranks` holds values in
   `[0, attn_tp_size)` (populated from `bind.cpp`'s `attn_tp_rank =
   rank(d_tp_group) / attn_cp_size`). With `tp = attn_tp × cp` the shard
   function splits the tensor into `attn_tp × cp` pieces and picks the
   first `attn_tp` of them; the remaining `(cp-1) × attn_tp` shards are
   never committed. Masked by `attn_cp_size == 1` being the common case.
   Same bug exists on the C++ side in `ModelWeight::tp_size` (which is
   initialized to `attn_tp_size * attn_cp_size` and only used to derive
   `vocab_size_padded`).

Plus a structural issue on `TextModelBuilder` specifically: its
`__init__(handles, contexts, tp=1, ranks=None)` defaults let the root
silently become `tp=1, ranks=None` — broadcast semantics — even though
its real role (owning `tok_embeddings` and `output` commits) requires
attn-TP values. The bypass papers over this by sidestepping `_commit_*`
on the root entirely. The base `Builder`'s defaults are fine for
broadcast / pure-attachment builders (`NormBuilder`, `ModuleListBuilder`,
`DecoderLayerBuilder`) which never shard, and stay as-is.

## Architecture

```
┌── C++ model_weight.h ──────────────────────────────────┐
│ #define MODEL_WEIGHT_CHILDREN(X)         \             │
│     X(LinearWeight,     output)          \             │
│     X(NormWeight,       norm)            \             │
│     X(core::ModuleList, layers)                        │
│                                                        │
│ #define MODEL_WEIGHT_PARAMS(X)           \             │
│     X(tok_embeddings)                                  │
│                                                        │
│ int attn_tp_size{};    // renamed from tp_size         │
│ int tp_rank{};                                         │
└────────────────────────────────────────────────────────┘

┌── Python builder/_base.py ─────────────────────────────┐
│ class Builder:                                         │
│     def __init__(self, config, contexts,               │
│                  tp=1, ranks=None):                    │
│         # defaults unchanged; broadcast builders       │
│         # (Norm/ModuleList/DecoderLayer) rely on them  │
│                                                        │
│ class TextModelBuilder(Builder):                       │
│     def __init__(self, handles, contexts, *,           │
│                  tp, ranks, vocab_size):               │
│         # required keyword-only                        │
│     def add_token_embeds(self, tensor): ...            │
│     def add_lm_head(self, linear): ...                 │
└────────────────────────────────────────────────────────┘

┌── Python source_model/<arch>_spec.py :: model() ───────┐
│ def model(self):                                       │
│     root = TextModelBuilder(                           │
│         self._root_handles, self._contexts,            │
│         tp=self.engine_cfg.attn_tp_size,               │
│         ranks=self._attn_ranks,                        │
│         vocab_size=self._vocab_size)                   │
│     root.add_token_embeds(self._get(self._embed_key))  │
│     root.norm = self.output_norm(self._norm_key)       │
│     lm_key = (self._embed_key if self._tie_embeddings  │
│               else 'lm_head.weight')                   │
│     root.add_lm_head(                                  │
│         self._linear(lm_key.removesuffix('.weight')))  │
│     root.layers = self.layers(self._layer_prefix)      │
└────────────────────────────────────────────────────────┘
```

### TP groups, clarified

Three parallel groups exist; they shard different things:

| Group | Size (engine config) | Ranks source | What it shards |
| --- | --- | --- | --- |
| attn TP | `attn_tp_size` | `bind.cpp` `attn_tp_rank` | attn QKV/O, **tok_embeddings**, **lm_head** |
| attn CP | `attn_cp_size` | `bind.cpp` `attn_cp_rank` | **sequences**, not weights |
| mlp TP  | `mlp_tp_size`  | `bind.cpp` `mlp_tp_rank`  | FFN / MoE |

`attn_tp_rank = rank(d_tp_group) / attn_cp_size` (see
`src/turbomind/turbomind.cc` line 243) is in `[0, attn_tp_size)`.
`self._attn_ranks[i]` (populated through
`TextModelLoader._bind_runtime`) is exactly this value per GPU. Weights in
the attn-TP group are replicated across CP peers — two GPUs with the same
`attn_tp_rank` but different `attn_cp_rank` hold the same shard.

The correct TP factor for `tok_embeddings` and `lm_head` is
`attn_tp_size` alone.

## 1. C++ `ModelWeight` reshape

### `tok_embeddings`: Tensor parameter, not `LinearWeight` child

```cpp
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)           \
    X(tok_embeddings)
```

`language_model.cc` drops the `->weight` indirection:

```cpp
// before:   weights_.tok_embeddings->weight
// after:    weights_.tok_embeddings
```

`verify()` checks the param instead of the child:

```cpp
if (!tok_embeddings) {
    missing.push_back(full_path() + ": missing tok_embeddings");
}
```

### `attn_tp_size` replaces `tp_size`

`ModelWeight::tp_size` is used in exactly one place:
`vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size)`. Today
it is initialized from `engine_param.attn_tp_size * engine_param.attn_cp_size`
— the same `× cp` bug as in Python. Rename to `attn_tp_size` and
initialize from `engine_param.attn_tp_size` alone:

```cpp
ModelWeight::ModelWeight(const EngineParam& engine_param)
    : attn_tp_size(engine_param.attn_tp_size)
    , tp_rank(engine_param.attn_tp_rank)
{
    ...
}

void ModelWeight::prepare() {
    ...
    vocab_size        = tok_embeddings.shape(0);   // tensor directly
    vocab_size_padded = round_up((size_t)vocab_size, (size_t)attn_tp_size);
    ...
}
```

Header:

```cpp
int attn_tp_size{};
int tp_rank{};
```

Nothing outside `ModelWeight` reads `tp_size` (verified across the C++
tree), so the rename is self-contained.

Effect: `vocab_size_padded` is now consistently `round_up(vocab,
attn_tp_size)` — matches the Python padding in `add_lm_head`. Sampling /
penalty / logprob kernels continue to stride on `vocab_size_padded`; the
value is smaller or equal to before (equal when `cp == 1`, strictly
smaller when `cp > 1`).

## 2. Python `TextModelBuilder` TP/ranks discipline

### Required keyword-only on `TextModelBuilder` only

```python
class Builder:
    def __init__(self, config, contexts, tp=1, ranks=None):
        # unchanged: defaults preserved for broadcast / pure-attachment
        # builders (NormBuilder, ModuleListBuilder, DecoderLayerBuilder).
        ...


class TextModelBuilder(Builder):
    def __init__(self, handles, contexts, *, tp, ranks, vocab_size):
        object.__setattr__(self, '_handles', handles)
        object.__setattr__(self, '_contexts', contexts)
        object.__setattr__(self, '_tp', tp)
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, '_vocab_size', vocab_size)
        object.__setattr__(self, '_children', {})
        object.__setattr__(self, '_handles_created', True)
        object.__setattr__(self, 'config', None)
```

`TextModelBuilder` is the only class that loses its defaults. It now
requires explicit `tp`, `ranks`, `vocab_size` at construction. This is
where the original antipattern sat — a root builder that silently
defaulted to `tp=1, ranks=None` but whose commits (`tok_embeddings`,
`output`) actually need real attn-TP values. `vocab_size` is threaded in
so `add_lm_head` can pad without asking per call.

The base `Builder` keeps its `tp=1, ranks=None` defaults. The only
construction sites that rely on them are broadcast / pure-attachment
builders (`NormBuilder`, `ModuleListBuilder`, `DecoderLayerBuilder`) —
making them explicit everywhere adds noise without catching a real bug
(`_rank_for` returns `0` when `tp <= 1`; these builders don't shard).

### Construction sites unchanged except for `TextModelBuilder`

- attn-TP builders (`AttentionBuilder`, `MLABuilder`, `DeltaNetBuilder`)
  already pass `tp=engine_cfg.attn_tp_size`, `ranks=self._attn_ranks` at
  every call site. No change.
- mlp-TP builders (`FfnBuilder`, `MoeBuilder`) already pass
  `tp=engine_cfg.mlp_tp_size`, `ranks=self._mlp_ranks`. No change.
- Broadcast builders (`NormBuilder`, `ModuleListBuilder`,
  `DecoderLayerBuilder`) keep `tp=1, ranks=None` via base defaults. No
  change at construction sites.
- `TextModelBuilder`: now constructed with required
  `tp=engine_cfg.attn_tp_size, ranks=self._attn_ranks,
  vocab_size=self._vocab_size`.

No `engine_cfg.attn_cp_size` multiplication anywhere on the Python side.
`self._attn_ranks` / `self._mlp_ranks` come from
`model_comm.attn_tp_rank(gpu)` / `model_comm.mlp_tp_rank(gpu)` via
`TextModelLoader._bind_runtime` — the values defined in `bind.cpp` lines
726–727.

### No changes to `_commit_linear` / `_commit_tensor` signatures

They keep reading `self._tp` / `self._ranks`. The discipline change is at
`TextModelBuilder` construction: every instance is initialized with
explicit attn-TP values, so `_commit_*` always sees the right values
without needing per-call overrides.

## 3. `TextModelBuilder` public commit methods

```python
from .._base import Builder, SplitSide
from ..linear import Linear, pad_out_dim


class TextModelBuilder(Builder):
    """Wraps pre-existing root `ModelWeight` handles. Owns tok_embeddings
    and lm_head commits on the root.
    """

    def __init__(self, handles, contexts, *, tp, ranks, vocab_size):
        ...

    def add_token_embeds(self, tensor):
        """Commit the raw embedding lookup as the `tok_embeddings` root param.

        Shards along hidden (output) dim by self._tp. No vocab padding —
        embedding lookup never indexes past vocab - 1.
        """
        self._commit_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT)

    def add_lm_head(self, linear):
        """Pad output dim to `round_up(vocab_size, tp)` and commit to
        the `output` LinearWeight root child.

        Works for any format: pad is applied uniformly across every
        tensor in the Linear bundle along dim=-1 (weight, scales, zeros,
        bias if present).
        """
        padded_vocab = ((self._vocab_size + self._tp - 1)
                        // self._tp) * self._tp
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format,
            data_format=linear.data_format)
        self._commit_linear('output', padded,
                            split_side=SplitSide.OUTPUT)
```

The internal commits still use the **C++ names** `'tok_embeddings'` and
`'output'`, matching `ModelWeight`'s X-macro declarations. Public Python
method names `add_token_embeds` / `add_lm_head` are the spec-facing
semantic names.

Imports now needed in `builder/_base.py` (or near `TextModelBuilder`, if
kept in the same file): `Linear`, `pad_out_dim` from `..linear`, and
`SplitSide` (already defined here).

## 4. Deletions

### `lmdeploy/turbomind/deploy/builder/linear.py` — deleted

Contained only `LinearBuilder` and `make_linear_config`. Both existed only
to serve the bypass. Zero live callers after §3.

### `lmdeploy/turbomind/deploy/builder/__init__.py`

Drop exports:

```python
# removed:
from .linear import LinearBuilder, make_linear_config
# and from __all__:
'LinearBuilder', 'make_linear_config',
```

### `lmdeploy/turbomind/deploy/spec.py`

Delete entirely:
- `TextModelSpec.token_embeds` method (lines 177–188).
- `TextModelSpec.lm_head` method (lines 190–202).
- Imports: `LinearBuilder`, `SplitSide`, `make_linear_config` from
  `.builder`; `pad_out_dim` from `.linear`.

Remaining spec.py imports:

```python
from .builder import _cpp_dtype as _cd
from .source_model.utils import (detect_layer_prefix,
                                 parse_rope_param, rope_type_to_int)
```

`TextModelSpec.token_embeds` / `lm_head` were not overridden by any
subclass — pure indirection.

## 5. Spec subclass `model()` rewrites

All four `source_model/*_spec.py` files carry the same shape change to
their `model()` methods. Example (Qwen3):

Before:
```python
def model(self):
    root = TextModelBuilder(self._root_handles, self._contexts)
    root.tok_embeddings = self.token_embeds(self._embed_key)
    root.norm = self.output_norm(self._norm_key)
    lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
    root.output = self.lm_head(lm_key)
    root.layers = self.layers(self._layer_prefix)
```

After:
```python
def model(self):
    root = TextModelBuilder(
        self._root_handles, self._contexts,
        tp=self.engine_cfg.attn_tp_size,
        ranks=self._attn_ranks,
        vocab_size=self._vocab_size)
    root.add_token_embeds(self._get(self._embed_key))
    root.norm = self.output_norm(self._norm_key)
    lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
    root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
    root.layers = self.layers(self._layer_prefix)
```

`root.norm` and `root.layers` keep the child-binding pattern (via
`__setattr__` → `add_child_raw`) — that operation is TP-agnostic and
doesn't care about root's stored tp/ranks.

Only one construction site changes in each spec file — the
`TextModelBuilder(...)` call. Every other builder construction
(`AttentionBuilder`, `FfnBuilder`, `MoeBuilder`, `DeltaNetBuilder`,
`MLABuilder`, `NormBuilder`, `ModuleListBuilder`, `DecoderLayerBuilder`)
stays as-is.

GLM4-MoE-Lite never ties embeddings, so its `lm_key` line is
unconditional:

```python
root.add_lm_head(self._linear('lm_head'))
```

Otherwise identical.

## 6. Data flow

### `add_token_embeds`

```
params[self._embed_key]  ──►  raw [vocab, hidden] tensor (no normalizer)
                              │
                              ▼  root.add_token_embeds(tensor)
                          _commit_tensor('tok_embeddings', tensor,
                                         split_side=SplitSide.OUTPUT)
                              │
                              ▼  per GPU
                          rank  = self._rank_for(gpu)   # attn_tp_rank
                          shard = tensor.split(hidden/tp, dim=-1)[rank]
                          handle.param('tok_embeddings')
                              .alloc([vocab, hidden/tp], dtype)
                              .copy_from(shard)
```

Per-rank shape: `[vocab, hidden / attn_tp_size]`. Vocab stays unpadded —
C++ lookup never indexes past `vocab - 1`. CP peers get the same shard.

### `add_lm_head`

```
self._linear(prefix)  ──►  Linear {tensors, weight_format, data_format}
                          │   (build_linear dispatches on format;
                          │    trivial normalizer does .t() → [hidden, vocab])
                          ▼  root.add_lm_head(linear)
                      pad each tensor.dim=-1 to padded_vocab
                                = round_up(vocab, attn_tp_size)
                          │
                          ▼
                      _commit_linear('output', padded_linear,
                                     split_side=SplitSide.OUTPUT)
                          │
                          ▼  per GPU
                      rank  = self._rank_for(gpu)
                      handle.create_child('output', LinearConfig{...})
                      for kind in padded_linear.tensors:
                          shard = ... dim=-1 by tp ... [rank]
                          _copy_shard_to_param(child, kind, shard)
```

Per-rank `output->weight` shape: `[hidden, padded_vocab / attn_tp_size]`.

## 7. Invariants

1. Spec never calls `_commit_*` on a builder directly. Every commit flows
   through a public builder method (`add_X`, `set_weight`).
2. `Builder.__init__` (and `TextModelBuilder.__init__`) require tp/ranks
   keyword-only. Every construction site sources them from
   `engine_cfg` (tp size) and the per-GPU ranks lists populated from
   `bind.cpp` (`attn_tp_rank`, `mlp_tp_rank`) via `TextModelLoader`.
3. `attn_tp_size` and `attn_cp_size` are never multiplied together when
   sizing weight shards. CP does not shard weights.
4. `tok_embeddings` is a Tensor parameter on `ModelWeight`, unpadded
   along vocab. `output` is a `LinearWeight` child, padded to
   `round_up(vocab, attn_tp_size)`.
5. `ModelWeight::vocab_size_padded == round_up(vocab_size, attn_tp_size)`,
   matching the Python pad in `add_lm_head`.
6. `LinearBuilder`, `make_linear_config`, `builder/linear.py` do not
   exist after this change.

## 8. Edge cases

### Tied embeddings

`lm_key = self._embed_key` → same checkpoint tensor consumed twice:
- `add_token_embeds`: raw `[vocab, hidden]`, unpadded, sharded along
  hidden.
- `add_lm_head`: via `self._linear(prefix)` (trivial normalizer `.t()` →
  `[hidden, vocab]`), padded along vocab, sharded along vocab.

Both commits target independent C++ slots (Tensor param vs. LinearWeight
child). Correct without special cases.

### Quantized `lm_head`

`self._linear(prefix)` dispatches to the detected format's normalizer
(AWQ / GPTQ / compressed-tensors / FP8 / MXFP4 / trivial). `add_lm_head`
pads every tensor in the bundle along `dim=-1`. For AWQ / GPTQ /
compressed-tensors the block-grouping is along K (input), not N (output),
so output-dim padding by a small amount (`< attn_tp_size`, usually ≤ 16)
is compatible.

FP8 / MXFP4 `lm_head` would pad scales block-sized along output too —
pathological but never seen in practice (lm_head is always trivial in
released checkpoints). No assert; if encountered, it surfaces as a
dimension mismatch in `_commit_linear`'s uniform TP-split validation.

### Multi-modal prefix

Qwen3.5 as multimodal root has weights under
`model.language_model.*`. `detect_layer_prefix` returns
`_embed_key = 'model.language_model.embed_tokens.weight'`. After
`.removesuffix('.weight')` it becomes
`'model.language_model.embed_tokens'`; `self._linear(prefix)` looks up
`{prefix}.weight` and `{prefix}.scales` etc. correctly.

### `attn_cp_size > 1`

With the fix, `tp = attn_tp_size` (not × cp). Shard function splits
`attn_tp_size` ways; ranks in `[0, attn_tp_size)` pick their shards. CP
peers share the same shard (they share `attn_tp_rank`). Without the fix
(current code), shards were `hidden / (attn_tp × cp)` wide but ranks only
covered the first `attn_tp` — downstream attention received too-narrow
inputs and produced gibberish whenever `cp > 1`.

### Broadcast / pure-attachment builders

`NormBuilder`, `ModuleListBuilder`, `DecoderLayerBuilder` all pass
`tp=1, ranks=None` at construction. `_rank_for` returns 0 when `tp <= 1`,
`_shard` passes tensor through untouched. Broadcast semantics preserved;
explicit at every call site.

## 9. Testing

Per `AGENTS.md`: `scripts/test_turbomind_model.py` with ≥128 tokens of
coherent output each. Check `get_gpu_usage` first. Use `model-server` MCP
for locally cached models.

| Case | TP | Why |
| --- | --- | --- |
| Qwen3 trivial dense | 1 | Baseline: semantics preserved vs. today (cp=1). |
| Qwen3 trivial dense | 2 | Real TP sharding of tok_embeddings + lm_head. |
| AWQ quantized model | 1, 2 | Exercises `self._linear(prefix)` through quantized path; lm_head typically trivial but dispatch must not regress. |
| Tied-embeddings model (e.g. GPT-OSS if tied, or Qwen3 small variant) | 1 | Same tensor → two independent slots. |
| Qwen3.5 (DeltaNet + multimodal prefix) | 1 | Multi-modal `_embed_key` path. |
| GLM4-MoE-Lite (MLA) | 2 | MLA path untouched, exercises the Builder tp/ranks change across builders. |
| Optional: model with `attn_cp_size > 1` if available | N | Validates the cp-bug fix; otherwise preventive. |

Failure criteria: gibberish, dimension asserts, crashes → halt and bisect.

## 10. Commit cadence

One atomic commit. C++ and Python changes are tightly coupled
(`ModelWeight`'s shape change requires the Python commit-path rewrite
landing together). The implementation plan may split into finer tasks
internally but the validated checkpoint is unified.

## 11. Summary of deletions and additions

### Deleted

**Python:**
- `lmdeploy/turbomind/deploy/builder/linear.py` (file)
- `LinearBuilder` class
- `make_linear_config` function
- `TextModelSpec.token_embeds` method
- `TextModelSpec.lm_head` method
- Default values `tp=1, ranks=None` on `TextModelBuilder.__init__`
  (base `Builder` keeps its defaults — broadcast builders still rely
  on them)
- `spec.py` imports: `LinearBuilder`, `SplitSide`, `make_linear_config`,
  `pad_out_dim`

**C++:**
- `LinearWeight tok_embeddings` child declaration on `ModelWeight`
- `ModelWeight::tp_size` field (replaced by `attn_tp_size`)
- `->weight` indirection on `tok_embeddings` access in
  `language_model.cc`

### Added

**Python:**
- `TextModelBuilder.__init__` required `vocab_size` kwarg
- `TextModelBuilder.add_token_embeds(tensor)`
- `TextModelBuilder.add_lm_head(linear)`

**C++:**
- `X(tok_embeddings)` entry in `MODEL_WEIGHT_PARAMS`
- `ModelWeight::attn_tp_size` field (renamed from `tp_size`, initialized
  from `engine_param.attn_tp_size`)

### Behavior changes visible at runtime

1. `attn_cp_size > 1` deployments: tok_embeddings and lm_head are now
   sharded correctly along attn TP only. Previously under-shared.
2. `vocab_size_padded = round_up(vocab, attn_tp_size)` (was `× cp`).
   Same value when `cp == 1`; smaller when `cp > 1`. Sampling /
   penalty / logprob kernels see the reduced width.
3. `tok_embeddings` shape in C++ becomes `[vocab, hidden / attn_tp_size]`
   (unpadded along vocab). Previously padded along vocab. Embedding
   lookup results identical — the padded rows were dead storage.
4. Constructing `TextModelBuilder` without explicit `tp`, `ranks`, and
   `vocab_size` now raises `TypeError`. Previously the missing args
   silently defaulted to `tp=1, ranks=None` and `vocab_size` didn't
   exist — the bypass path papered over it. Base `Builder` is
   unaffected.
