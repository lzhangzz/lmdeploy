# Text-model root commits: `add_token_embeds` / `add_lm_head` on `TextModelBuilder`

Date: 2026-04-22

Scope: root-level weight commits (`tok_embeddings`, `output`) on the
model's `ModelWeight` root, plus a proper per-GPU `model_tp_rank` on
the Python side. Touches both C++ and Python.

- `src/turbomind/models/model_weight.{h,cc}`
- `src/turbomind/models/language_model.cc`
- `src/turbomind/models/llama/llama_params.h`
- `src/turbomind/turbomind.{h,cc}`
- `src/turbomind/python/bind.cpp`
- `lmdeploy/turbomind/deploy/builder/_base.py`
- `lmdeploy/turbomind/deploy/builder/linear.py` (deleted)
- `lmdeploy/turbomind/deploy/builder/__init__.py`
- `lmdeploy/turbomind/deploy/target_model/base.py`
- `lmdeploy/turbomind/deploy/text_model_loader.py`
- `lmdeploy/turbomind/deploy/spec.py`
- `lmdeploy/turbomind/deploy/source_model/{qwen3,qwen3_5,glm4_moe_lite,gpt_oss}_spec.py`

## Motivation

`spec.py` lines 177–203 (`TextModelSpec.token_embeds` /
`TextModelSpec.lm_head`) concentrate five co-existing antipatterns:

1. **Parallel commit lifecycle.** `LinearBuilder` +
   `make_linear_config` + `set_weight` is a bypass that creates a
   standalone `_tm.create_module(LinearConfig)` on its own and
   attaches the result as a root child via `add_child_raw`. Every
   other linear flows through `build_linear → Linear →
   _commit_linear → handle.create_child(name, cfg)`. Two commit
   pipelines for the same concept.

2. **Spec calls `_commit_*` directly (would, if we removed the
   bypass the naïve way).** The repo-wide convention is that spec
   hands tensors / `Linear` bundles to **public** builder methods
   (`add_qkv_proj`, `add_ffn`, `add_gate`, `set_weight`, …) and the
   builder owns the commit. Spec never touches `_commit_*`.

3. **`lm_head` reads raw tensor instead of flowing through
   `self._linear`.** The existing code uses `self._get(key).t()` +
   manual `make_linear_config`, skipping the normalizer / format
   detection / `data_format` that every other linear gets. A
   quantized `lm_head` (rare but possible) is silently unsupported.

4. **`tok_embeddings` is a `LinearWeight` child but used as a lookup
   table.** Only `->weight` is ever read in
   `src/turbomind/models/language_model.cc`. Wrapping it in
   `LinearWeight` is scaffolding; it should be a `Tensor` parameter
   on `ModelWeight`. Additionally, the current code pads vocab —
   the embedding lookup never indexes past `vocab_size - 1`, so
   padded rows are dead storage.

5. **The TP rank handed to the commits is wrong.** Both
   `token_embeds` and `lm_head` pass `tp = attn_tp_size *
   attn_cp_size` (correct — these weights belong to the `d_tp_group`
   of that size) but feed `self._attn_ranks`, which holds
   per-GPU `attn_tp_rank` (`rank(d_tp_group) / attn_cp_size`) and
   therefore lives in `[0, attn_tp_size)`. Feeding that list to a
   `_shard(..., tp = attn_tp × cp, rank)` call only ever indexes the
   first `attn_tp_size` of the `attn_tp_size × attn_cp_size` shards;
   the remaining `(cp - 1) × attn_tp` shards of weight are never
   committed to any GPU, and CP peers that should hold **different**
   shards (because they're members of the same `d_tp_group`) end up
   holding identical ones. Masked entirely by `attn_cp_size == 1`
   being the common case. The fix is to expose a per-GPU
   `model_tp_rank = rank(d_tp_group)` in `[0, attn_tp × cp)`, carry
   it onto the spec, and use it for these commits.

Plus a structural issue on `TextModelBuilder` specifically: its
`__init__(handles, contexts, tp=1, ranks=None)` defaults let the
root silently become `tp=1, ranks=None` — broadcast semantics —
even though its real role (owning `tok_embeddings` and `output`
commits) requires real tp-group values. The bypass papers over
this by sidestepping `_commit_*` on the root entirely. The base
`Builder`'s defaults are fine for broadcast / pure-attachment
builders (`NormBuilder`, `ModuleListBuilder`, `DecoderLayerBuilder`)
which never shard, and stay as-is.

## Architecture

```
┌── C++ ─────────────────────────────────────────────────┐
│ model_weight.h: tok_embeddings moves CHILDREN→PARAMS   │
│                                                        │
│ #define MODEL_WEIGHT_CHILDREN(X)         \             │
│     X(LinearWeight,     output)          \             │
│     X(NormWeight,       norm)            \             │
│     X(core::ModuleList, layers)                        │
│                                                        │
│ #define MODEL_WEIGHT_PARAMS(X)           \             │
│     X(tok_embeddings)                                  │
│                                                        │
│ llama_params.h: EngineParam gains                      │
│     int model_tp_rank = 0;                             │
│ turbomind.cc: populates from rank(d_tp_group)          │
│ turbomind.h / bind.cpp: exposes GetModelTpRank /       │
│     .def("model_tp_rank", ...)                         │
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
│                  tp, ranks, vocab_size, data_type):    │
│         # required keyword-only                        │
│     def add_token_embeds(self, tensor): ...            │
│     def add_lm_head(self, linear): ...                 │
└────────────────────────────────────────────────────────┘

┌── Python source_model/<arch>_spec.py :: model() ───────┐
│ def model(self):                                       │
│     ec  = self.engine_cfg                              │
│     root = TextModelBuilder(                           │
│         self._root_handles, self._contexts,            │
│         tp=ec.attn_tp_size * ec.attn_cp_size,          │
│         ranks=self._model_tp_ranks,                    │
│         vocab_size=self._vocab_size,                   │
│         data_type=self._cpp_dtype())                   │
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

| Group | Size | Per-GPU rank source | What it shards |
| --- | --- | --- | --- |
| **model TP** (this refactor) | `attn_tp_size × attn_cp_size` | new `model_tp_rank` via `bind.cpp` | **tok_embeddings**, **lm_head** |
| attn TP | `attn_tp_size` | existing `attn_tp_rank` | attn QKV / O proj |
| mlp TP  | `mlp_tp_size`  | existing `mlp_tp_rank` | FFN / MoE |
| attn CP | `attn_cp_size` | existing `attn_cp_rank` | sequences, not weights |

`rank(d_tp_group)` lives in `[0, attn_tp × cp)` and is the correct
shard index for weights that belong to the whole `d_tp_group` — that
is, `tok_embeddings` and `lm_head`. Today turbomind.cc divides it
down to `attn_tp_rank = rank(d_tp_group) / attn_cp_size`; we add a
sibling `model_tp_rank = rank(d_tp_group)` (undivided) so Python can
pick the right shards.

## 1. C++ changes

### 1.1 `tok_embeddings`: Tensor parameter, not `LinearWeight` child

```cpp
// src/turbomind/models/model_weight.h
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)           \
    X(tok_embeddings)
```

`model_weight.cc`:

```cpp
// prepare():  read shape from the Tensor directly
vocab_size        = tok_embeddings.shape(0);
vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);  // unchanged

// verify():  check param presence instead of child
if (!tok_embeddings) {
    missing.push_back(full_path() + ": missing tok_embeddings");
}
```

`ModelWeight::tp_size` stays as initialized today (`attn_tp_size *
attn_cp_size`). No rename, no init change — it's the correct TP
group size for `vocab_size_padded`.

`language_model.cc` drops one layer of indirection:

```cpp
// before:   weights_.tok_embeddings->weight
// after:    weights_.tok_embeddings
```

Line 194 is the only access.

### 1.2 Expose `model_tp_rank`

Add a new per-GPU rank, structured like the existing `attn_tp_rank` /
`mlp_tp_rank`:

```cpp
// src/turbomind/models/llama/llama_params.h
struct EngineParam : EngineConfig {
    int outer_dp_rank = 0;
    int attn_dp_rank  = 0;
    int attn_tp_rank  = 0;
    int attn_cp_rank  = 0;
    int mlp_tp_rank   = 0;
    int model_tp_rank = 0;   // NEW: rank(d_tp_group), in [0, attn_tp × cp)

    int max_forward_token_num = 0;
};
```

```cpp
// src/turbomind/turbomind.cc  (inside the per-GPU init block around line 243)
p.model_tp_rank = c.d_comm->rank(c.d_tp_group);
p.attn_tp_rank  = p.model_tp_rank / p.attn_cp_size;   // same value as today
p.mlp_tp_rank   = c.d_comm->rank(0);

// ...below, alongside GetAttnTpRank / GetMlpTpRank:
int TurboMind::GetModelTpRank(int index)
{
    return impl_->engine_params_.at(index).model_tp_rank;
}
```

```cpp
// src/turbomind/turbomind.h
int GetModelTpRank(int index);
```

```cpp
// src/turbomind/python/bind.cpp  (alongside lines 726–727)
.def("attn_tp_rank",  &TurboMind::GetAttnTpRank,  "index"_a)
.def("mlp_tp_rank",   &TurboMind::GetMlpTpRank,   "index"_a)
.def("model_tp_rank", &TurboMind::GetModelTpRank, "index"_a);
```

Single-GPU (`comm_size_ == 1`) and `attn_cp_size == 1` cases
both produce `model_tp_rank == attn_tp_rank` (by construction);
the new field only diverges when `attn_cp_size > 1`.

## 2. Python plumbing: `model_tp_ranks` onto the spec

`model_tp_ranks` travels Python-side through exactly the same path
as `attn_tp_ranks` / `mlp_tp_ranks`, with one new collection point
and one new `bind_runtime` argument.

### 2.1 `target_model/base.py`

`BaseOutputModel.tp_ranks(index)` returns attn + mlp today; extend it
to return model too. Keeping the existing return shape as a tuple
(old length 2 → new length 3) keeps readers minimal:

```python
def tp_ranks(self, index: int):
    return (self.model_comm.attn_tp_rank(index),
            self.model_comm.mlp_tp_rank(index),
            self.model_comm.model_tp_rank(index))
```

### 2.2 `text_model_loader.py`

Collect the third rank list and pass it to the spec:

```python
def _bind_runtime(self):
    model = self.model
    attn_ranks  = [model.tp_ranks(gpu)[0] for gpu in range(model.gpu_count)]
    mlp_ranks   = [model.tp_ranks(gpu)[1] for gpu in range(model.gpu_count)]
    model_ranks = [model.tp_ranks(gpu)[2] for gpu in range(model.gpu_count)]
    ...
    model.spec.bind_runtime(
        contexts=contexts,
        root_handles=handles,
        attn_ranks=attn_ranks,
        mlp_ranks=mlp_ranks,
        model_tp_ranks=model_ranks,
    )
```

### 2.3 `spec.py`

`TextModelSpec.bind_runtime` gains `model_tp_ranks`; storage field
is `self._model_tp_ranks`:

```python
def bind_runtime(self, *, contexts, root_handles,
                 attn_ranks, mlp_ranks, model_tp_ranks):
    self._contexts       = contexts
    self._root_handles   = root_handles
    self._attn_ranks     = attn_ranks
    self._mlp_ranks      = mlp_ranks
    self._model_tp_ranks = model_tp_ranks
```

No other spec method reads it — only `model()` uses it for
`TextModelBuilder`. `attn_ranks` / `mlp_ranks` are untouched; all
other builders keep using them as today.

## 3. `TextModelBuilder` public commit methods

```python
# lmdeploy/turbomind/deploy/builder/_base.py

class TextModelBuilder(Builder):
    """Wraps pre-existing root `ModelWeight` handles. Owns tok_embeddings
    and lm_head commits on the root.
    """

    def __init__(self, handles, contexts, *,
                 tp, ranks, vocab_size, data_type):
        object.__setattr__(self, '_handles', handles)
        object.__setattr__(self, '_contexts', contexts)
        object.__setattr__(self, '_tp', tp)
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, '_vocab_size', vocab_size)
        object.__setattr__(self, '_data_type', data_type)
        object.__setattr__(self, '_children', {})
        object.__setattr__(self, '_handles_created', True)
        object.__setattr__(self, 'config', None)

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

        Uniform pad along dim=-1 works for trivial / AWQ / GPTQ /
        compressed-tensors / MXFP4 (block_out is None or 1). FP8 lm_head
        (block_out = 128) would misalign scales under naive padding —
        not a configuration used by any released checkpoint; if
        encountered it surfaces downstream as a dim-mismatch /
        TP-split validation error in `_commit_linear`.
        """
        padded_vocab = ((self._vocab_size + self._tp - 1)
                        // self._tp) * self._tp
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format,
            data_format=linear.data_format)
        self._commit_linear('output', padded,
                            split_side=SplitSide.OUTPUT,
                            model_dtype=self._data_type)
```

Internal commits use the **C++ names** `'tok_embeddings'` and
`'output'`, matching `ModelWeight`'s X-macro declarations. Public
Python method names `add_token_embeds` / `add_lm_head` are
spec-facing semantic names.

`TextModelBuilder` stays in `builder/_base.py`. That file already
imports `Linear` from `..linear` and defines `SplitSide` locally.
The only new import is `pad_out_dim`:

```python
# before: from ..linear import Linear
# after:  from ..linear import Linear, pad_out_dim
```

### Required keyword-only: rationale

`TextModelBuilder` is the only builder that loses its defaults.
`tp`, `ranks`, `vocab_size`, `data_type` are required keyword-only.
This closes the antipattern of a root builder silently defaulting
to `tp=1, ranks=None` while its commits actually need real tp-group
values. `vocab_size` drives padding in `add_lm_head`; `data_type`
is forwarded to `_commit_linear(..., model_dtype=...)`.

Base `Builder` keeps its `tp=1, ranks=None` defaults for
broadcast / pure-attachment builders (`NormBuilder`,
`ModuleListBuilder`, `DecoderLayerBuilder`). Forcing all of them
explicit adds noise without catching a real bug — `_rank_for`
returns 0 when `tp <= 1`, and these builders don't shard.

### Why `model_dtype` is threaded through `add_lm_head`

`_commit_linear` derives the compute dtype from `model_dtype` when
provided, otherwise from `_infer_compute_dtype(linear)`. The
fallback reads `linear.tensors['weight'].dtype` — for trivial this
is `bfloat16` / `float16` (correct). But for AWQ / GPTQ /
compressed-tensors, the post-normalizer weight is `torch.uint8` and
`_infer_compute_dtype` returns `TYPE_UINT8` — which becomes
`LinearConfig.data_type` and is the **wrong** compute dtype. Every
other quantized-weight commit in the repo passes
`model_dtype=<config>.data_type` explicitly
(`AttentionBuilder.add_qkv_proj`, `MoeBuilder.add_gate`, …). To
make quantized `lm_head` actually work (motivation #3),
`add_lm_head` threads it the same way.

## 4. Deletions

### `lmdeploy/turbomind/deploy/builder/linear.py` — deleted

Contained only `LinearBuilder` and `make_linear_config`. Both
existed only to serve the bypass. Zero live callers after §3.

### `lmdeploy/turbomind/deploy/builder/__init__.py`

Drop `LinearBuilder` / `make_linear_config` from imports and
`__all__`.

### `lmdeploy/turbomind/deploy/spec.py`

Delete:
- `TextModelSpec.token_embeds` method (lines 177–188).
- `TextModelSpec.lm_head` method (lines 190–202).
- Imports: `LinearBuilder`, `SplitSide`, `make_linear_config` from
  `.builder`; `pad_out_dim` from `.linear`.

Remaining spec.py imports from these modules:

```python
from .builder import _cpp_dtype as _cd
```

`TextModelSpec.token_embeds` / `lm_head` were not overridden by any
subclass — pure indirection.

## 5. Spec subclass `model()` rewrites

All four `source_model/*_spec.py` files carry the same shape
change. Example (Qwen3):

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
    ec = self.engine_cfg
    root = TextModelBuilder(
        self._root_handles, self._contexts,
        tp=ec.attn_tp_size * ec.attn_cp_size,
        ranks=self._model_tp_ranks,
        vocab_size=self._vocab_size,
        data_type=self._cpp_dtype())
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
`MLABuilder`, `NormBuilder`, `ModuleListBuilder`,
`DecoderLayerBuilder`) stays as-is.

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
                          rank  = self._rank_for(gpu)    # model_tp_rank
                          shard = tensor.split(hidden/tp, dim=-1)[rank]
                          handle.param('tok_embeddings')
                              .alloc([vocab, hidden/tp], dtype)
                              .copy_from(shard)
```

Per-rank shape: `[vocab, hidden / (attn_tp × cp)]`. Every GPU in the
tp group holds a distinct shard. Vocab stays unpadded — C++ lookup
never indexes past `vocab - 1`.

### `add_lm_head`

```
self._linear(prefix)  ──►  Linear {tensors, weight_format, data_format}
                          │   (build_linear dispatches on format;
                          │    trivial normalizer does .t() → [hidden, vocab])
                          ▼  root.add_lm_head(linear)
                      pad each tensor.dim=-1 to padded_vocab
                                = round_up(vocab, attn_tp × cp)
                          │
                          ▼
                      _commit_linear('output', padded_linear,
                                     split_side=SplitSide.OUTPUT,
                                     model_dtype=self._data_type)
                          │
                          ▼  per GPU
                      rank  = self._rank_for(gpu)    # model_tp_rank
                      handle.create_child('output', LinearConfig{...})
                      for kind in padded_linear.tensors:
                          shard = ... dim=-1 by tp ... [rank]
                          _copy_shard_to_param(child, kind, shard)
```

Per-rank `output->weight` shape:
`[hidden, padded_vocab / (attn_tp × cp)]`.

## 7. Invariants

1. Spec never calls `_commit_*` on a builder directly. Every commit
   flows through a public builder method (`add_X`, `set_weight`).
2. `TextModelBuilder.__init__` requires `tp`, `ranks`, `vocab_size`,
   `data_type` keyword-only. `tp` is sized to the `d_tp_group`
   (`attn_tp_size × attn_cp_size`); `ranks` is
   `self._model_tp_ranks` (each element is
   `bind.cpp::model_tp_rank`). Base `Builder` keeps its `tp=1,
   ranks=None` defaults for broadcast / pure-attachment builders.
3. `tok_embeddings` is a Tensor parameter on `ModelWeight`,
   unpadded along vocab. `output` is a `LinearWeight` child, padded
   to `round_up(vocab, attn_tp × cp)`.
4. `ModelWeight::tp_size`, `vocab_size_padded`, and
   `language_model.cc::tp_size_` all continue to equal
   `attn_tp × cp`. No C++ sizing changes beyond the structural
   `tok_embeddings` move.
5. For every GPU, `model_tp_rank ∈ [0, attn_tp × cp)` and every
   shard index in that range is covered by exactly
   `attn_dp_size × outer_dp_size` GPUs — the shard is replicated
   across DP replicas, not divided further.
6. `LinearBuilder`, `make_linear_config`, `builder/linear.py` do
   not exist after this change.

## 8. Edge cases

### Tied embeddings

`lm_key = self._embed_key` → same checkpoint tensor consumed twice:
- `add_token_embeds`: raw `[vocab, hidden]`, unpadded, sharded
  along hidden.
- `add_lm_head`: via `self._linear(prefix)` (trivial normalizer
  `.t()` → `[hidden, vocab]`), padded along vocab, sharded along
  vocab.

Both commits target independent C++ slots (Tensor param vs.
LinearWeight child). Correct without special cases.

### Quantized `lm_head`

`self._linear(prefix)` dispatches to the detected format's
normalizer (AWQ / GPTQ / compressed-tensors / FP8 / MXFP4 /
trivial). `add_lm_head` pads every tensor in the bundle along
`dim=-1` and passes `model_dtype` into `_commit_linear` (see §3).

- **Trivial / MXFP4 / AWQ / GPTQ / compressed-tensors**:
  `block_out` is `None`. Weight, scales, zeros, bias all pad
  uniformly along the output dim. Works.
- **FP8**: `block_out = 128`. Scales are `[K/128, N/128]` after
  `.t()` — block-structured along the output dim too. A naive
  `pad_out_dim` on `N/128` would misalign the block structure.
  Pathological and never seen in practice (`lm_head` is always
  trivial in released checkpoints). No guard; if encountered it
  surfaces downstream as a dim-mismatch / TP-split-validation
  failure in `_commit_linear`.

### Multi-modal prefix

Qwen3.5 as multimodal root has weights under
`model.language_model.*`. `detect_layer_prefix` returns
`_embed_key = 'model.language_model.embed_tokens.weight'`. After
`.removesuffix('.weight')` it becomes
`'model.language_model.embed_tokens'`; `self._linear(prefix)`
looks up `{prefix}.weight` etc. correctly.

### `attn_cp_size > 1`

With the `model_tp_rank` exposure, this becomes a first-class
tested path instead of a latent bug:
- Each GPU in the `d_tp_group` has a distinct
  `rank(d_tp_group) ∈ [0, attn_tp × cp)`. Feeding
  `self._model_tp_ranks` to a `tp = attn_tp × cp` split picks
  each GPU's own shard.
- CP peers within the same `d_tp_group` are different members of
  it (they share attn_tp_rank but differ in cp_rank); the
  `d_tp_group` AllGather in `language_model.cc` correctly
  reassembles hidden / vocab from the `attn_tp × cp` shards.
- `ModelWeight::vocab_size_padded`,
  `language_model.cc::tp_size_`, and the AllGather group all
  continue to use `attn_tp × cp` — unchanged.

Compared to today: the current code silently garbles content
whenever `cp > 1` (shards 0..attn_tp−1 get committed to
`attn_tp × cp` GPUs in pairs; shards attn_tp..attn_tp×cp−1 are
zeros from `alloc`). This refactor fixes that end-to-end.

### Broadcast / pure-attachment builders

`NormBuilder`, `ModuleListBuilder`, `DecoderLayerBuilder` rely on
the base `Builder`'s `tp=1, ranks=None` defaults (unchanged).
`_rank_for` returns 0 when `tp <= 1`; `_shard` passes tensor
through untouched. Broadcast semantics preserved without any
construction-site churn.

## 9. Testing

Per `AGENTS.md`: `scripts/test_turbomind_model.py`, ≥ 128 tokens
of coherent output each. Check `get_gpu_usage` first. Use
`model-server` MCP for locally cached models.

| Case | TP / CP | Why |
| --- | --- | --- |
| Qwen3 trivial dense | tp=1, cp=1 | Baseline. |
| Qwen3 trivial dense | tp=2, cp=1 | Real TP sharding of tok_embeddings + lm_head. |
| AWQ quantized model | tp=1,2 / cp=1 | Exercises `self._linear(prefix)` + `model_dtype` threading. |
| Tied-embeddings model (e.g. GPT-OSS if tied, or a Qwen3 small variant) | tp=1 / cp=1 | Same tensor → two independent C++ slots. |
| Qwen3.5 (DeltaNet + multimodal prefix) | tp=1 / cp=1 | Multi-modal `_embed_key` path. |
| GLM4-MoE-Lite (MLA) | tp=2 / cp=1 | MLA path untouched; unconditional `'lm_head'` key. |
| Any model | tp=2, cp=2 (if hardware allows) | First-class validation of the cp > 1 fix; confirms `model_tp_rank` plumbing is correct and embeddings are not garbled. |

Failure criteria: gibberish, dimension asserts, crashes → halt and
bisect.

## 10. Commit cadence

One atomic commit. C++ and Python changes are tightly coupled:

- The `model_tp_rank` pybind is needed by Python, which needs
  Python-side consumption, which is needed by the Python commit
  rewrite, which assumes `tok_embeddings` is a Tensor param — all
  of which land together.

The implementation plan may split into finer tasks internally but
the validated checkpoint is unified.

## 11. Summary of deletions and additions

### Deleted

**Python:**
- `lmdeploy/turbomind/deploy/builder/linear.py` (file)
- `LinearBuilder` class
- `make_linear_config` function
- `TextModelSpec.token_embeds` method
- `TextModelSpec.lm_head` method
- Default values `tp=1, ranks=None` on `TextModelBuilder.__init__`
  (base `Builder` keeps its defaults — broadcast builders still
  rely on them)
- `spec.py` imports: `LinearBuilder`, `SplitSide`,
  `make_linear_config`, `pad_out_dim`

**C++:**
- `LinearWeight tok_embeddings` child declaration on `ModelWeight`
- `->weight` indirection on `tok_embeddings` access in
  `language_model.cc`

### Added

**Python:**
- `TextModelBuilder.add_token_embeds(tensor)`,
  `TextModelBuilder.add_lm_head(linear)`
- `TextModelBuilder.__init__` required `vocab_size` and `data_type`
  kwargs (plus `tp` / `ranks` now required keyword-only)
- `pad_out_dim` added to `builder/_base.py`'s existing
  `from ..linear import Linear` import
- `TextModelSpec._model_tp_ranks` field; `bind_runtime` gains
  `model_tp_ranks` kwarg
- `BaseOutputModel.tp_ranks` returns a 3-tuple (adds model rank)
- `TextModelLoader._bind_runtime` collects and forwards
  `model_tp_ranks`

**C++:**
- `X(tok_embeddings)` entry in `MODEL_WEIGHT_PARAMS`
- `EngineParam::model_tp_rank` field
- Initialization in `turbomind.cc`:
  `p.model_tp_rank = c.d_comm->rank(c.d_tp_group)`
- `TurboMind::GetModelTpRank(int index)` getter
- `.def("model_tp_rank", &TurboMind::GetModelTpRank, "index"_a)`
  in `bind.cpp`

### Behavior changes visible at runtime

1. `tok_embeddings` shape in C++ becomes `[vocab, hidden /
   (attn_tp × cp)]` (unpadded along vocab). Previously it was
   padded along vocab to `[padded_vocab, hidden / (attn_tp × cp)]`.
   Embedding lookup results identical — padded rows were dead
   storage.
2. `cp > 1` deployments now function end-to-end. Today, weights
   are silently garbled because `self._attn_ranks` covers only
   `[0, attn_tp)` but the split factor is `attn_tp × cp`. With
   `model_tp_rank`, each GPU picks its own shard and the existing
   `d_tp_group` AllGather in `language_model.cc` recovers the full
   hidden / vocab dim.
3. `cp = 1` deployments behave identically to today —
   `model_tp_rank == attn_tp_rank` and all sizes / ranks collapse
   to the same values.
4. Constructing `TextModelBuilder` without explicit `tp`, `ranks`,
   `vocab_size`, `data_type` now raises `TypeError`. Previously
   the missing args silently defaulted to `tp=1, ranks=None` and
   `vocab_size` / `data_type` didn't exist — the bypass papered
   over it. Base `Builder` is unaffected.
