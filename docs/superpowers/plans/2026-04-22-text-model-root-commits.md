# TextModelBuilder Root Commits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `LinearBuilder` / `make_linear_config` / `set_weight` bypass used by `TextModelSpec.token_embeds` and `TextModelSpec.lm_head` with proper public methods (`add_token_embeds`, `add_lm_head`) on `TextModelBuilder`. Make `tok_embeddings` a Tensor parameter on `ModelWeight`. Expose a new per-GPU `model_tp_rank = rank(d_tp_group)` so the correct shard index is used for weights that belong to the full `d_tp_group` (size `attn_tp_size × attn_cp_size`). Fix the cp>1 correctness bug end-to-end.

**Architecture:** Three atomic commits in dependency order. Commit 1 adds the C++ `model_tp_rank` (additive; no live consumer yet). Commit 2 plumbs `model_tp_ranks` through Python (`BaseOutputModel.tp_ranks` → `TextModelLoader._bind_runtime` → `TextModelSpec.bind_runtime`). Commit 3 rewrites the C++ `ModelWeight` shape + the Python commit path in one atomic step (they are tightly coupled — cannot land independently). Testing happens after Commit 3, per AGENTS.md.

**Tech Stack:** C++ (CUDA host, pybind11), Python 3, ninja build, `scripts/test_turbomind_model.py` for end-to-end smoke tests.

**Spec:** `docs/superpowers/specs/2026-04-22-text-model-root-commits-design.md`.

---

## Files touched overview

| File | Task |
|---|---|
| `src/turbomind/models/llama/llama_params.h` | 1 |
| `src/turbomind/turbomind.h` | 1 |
| `src/turbomind/turbomind.cc` | 1 |
| `src/turbomind/python/bind.cpp` | 1 |
| `lmdeploy/turbomind/deploy/target_model/base.py` | 2 |
| `lmdeploy/turbomind/deploy/text_model_loader.py` | 2 |
| `lmdeploy/turbomind/deploy/spec.py` | 2, 3 |
| `src/turbomind/models/model_weight.h` | 3 |
| `src/turbomind/models/model_weight.cc` | 3 |
| `src/turbomind/models/language_model.cc` | 3 |
| `lmdeploy/turbomind/deploy/builder/_base.py` | 3 |
| `lmdeploy/turbomind/deploy/builder/linear.py` | 3 (deleted) |
| `lmdeploy/turbomind/deploy/builder/__init__.py` | 3 |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | 3 |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | 3 |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | 3 |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | 3 |

---

## Task 1: C++ `model_tp_rank` plumbing

**Files:**
- Modify: `src/turbomind/models/llama/llama_params.h:11-21`
- Modify: `src/turbomind/turbomind.h:48-52`
- Modify: `src/turbomind/turbomind.cc:238-245` (init) and `498-506` (getter)
- Modify: `src/turbomind/python/bind.cpp:726-727`

Goal: expose a new per-GPU rank `model_tp_rank = rank(d_tp_group)` (undivided — lives in `[0, attn_tp × cp)`) through the same `EngineParam` / `TurboMind` getter / pybind path used for `attn_tp_rank` and `mlp_tp_rank`. No Python consumer yet; this commit is purely additive and must compile and pass existing tests.

- [ ] **Step 1.1: Add `model_tp_rank` field to `EngineParam`**

Edit `src/turbomind/models/llama/llama_params.h`. Insert `model_tp_rank` alongside the other per-GPU ranks:

```cpp
struct EngineParam : EngineConfig {
    // Runtime-derived fields (set in CreateContext)
    int outer_dp_rank = 0;
    int attn_dp_rank = 0;
    int attn_tp_rank = 0;
    int attn_cp_rank = 0;
    int mlp_tp_rank = 0;
    int model_tp_rank = 0;   // rank(d_tp_group), in [0, attn_tp_size × attn_cp_size)

    // Derived field (set in Impl ctor)
    int max_forward_token_num = 0;
};
```

- [ ] **Step 1.2: Initialize `model_tp_rank` in `turbomind.cc`**

Edit `src/turbomind/turbomind.cc`. Inside the `if (comm_size_ > 1)` block (around line 227), compute `rank(d_tp_group)` once and derive both `model_tp_rank` (undivided) and `attn_tp_rank` (divided by cp) from it:

```cpp
if (comm_size_ > 1) {
    c.d_comm = CreateDeviceCommunicator(communicator_type_, comm_size_, inner_rank, c.h_comm);

    c.d_tp_group = 0;
    c.d_cp_group = 0;

    if (p.attn_dp_size > 1) {  // has attn_dp
        c.d_tp_group   = c.d_comm->Split(tp_color, 0, 0);
        p.attn_dp_rank = c.h_dp_group->rank();
    }

    if (p.attn_cp_size > 1) {  // has attn_cp
        c.d_cp_group   = c.d_comm->Split(cp_color, 0, 0);
        p.attn_cp_rank = c.d_comm->rank(c.d_cp_group);
    }

    p.model_tp_rank = c.d_comm->rank(c.d_tp_group);
    p.attn_tp_rank  = p.model_tp_rank / p.attn_cp_size;
    p.mlp_tp_rank   = c.d_comm->rank(0);
}
```

The `attn_tp_rank` line changes only in form — the numeric value is identical (`rank(d_tp_group) / attn_cp_size`, same as today). When `comm_size_ == 1` (single-GPU), the whole block is skipped and `model_tp_rank` stays at its default-init `0`, matching `attn_tp_rank`.

- [ ] **Step 1.3: Declare `GetModelTpRank` in `turbomind.h`**

Edit `src/turbomind/turbomind.h`. Alongside the existing getters:

```cpp
    /// Attention TP rank for GPU *index*.
    int GetAttnTpRank(int index);

    /// MLP TP rank for GPU *index*.
    int GetMlpTpRank(int index);

    /// Model-level TP rank (rank within d_tp_group) for GPU *index*.
    int GetModelTpRank(int index);
```

- [ ] **Step 1.4: Implement `GetModelTpRank` in `turbomind.cc`**

Edit `src/turbomind/turbomind.cc`. Add the implementation alongside the existing getters (around line 498):

```cpp
int TurboMind::GetAttnTpRank(int index)
{
    return impl_->engine_params_.at(index).attn_tp_rank;
}

int TurboMind::GetMlpTpRank(int index)
{
    return impl_->engine_params_.at(index).mlp_tp_rank;
}

int TurboMind::GetModelTpRank(int index)
{
    return impl_->engine_params_.at(index).model_tp_rank;
}
```

- [ ] **Step 1.5: Expose `model_tp_rank` in `bind.cpp`**

Edit `src/turbomind/python/bind.cpp`. Append a new `.def` after the existing `attn_tp_rank` / `mlp_tp_rank` bindings (lines 726–727):

```cpp
        .def("attn_tp_rank",  &TurboMind::GetAttnTpRank,  "index"_a)
        .def("mlp_tp_rank",   &TurboMind::GetMlpTpRank,   "index"_a)
        .def("model_tp_rank", &TurboMind::GetModelTpRank, "index"_a);
```

(The trailing semicolon moves from the `mlp_tp_rank` line to the new `model_tp_rank` line.)

- [ ] **Step 1.6: Build**

From the repo root:

```bash
cd build && ninja _turbomind
```

Expected: clean build. The new field and getter are unused on the Python side (no consumer yet), so no link failures.

- [ ] **Step 1.7: Smoke-test unchanged behavior**

Query `get_gpu_usage` for an empty GPU first. Then run a trivial Qwen3 smoke test to confirm nothing regressed:

```bash
python scripts/test_turbomind_model.py \
    <qwen3_path> <cache_dir> 1 <gpu_id>
```

Expected: ≥ 128 coherent tokens of output (not gibberish). Use `model-server` MCP tool (`list_models`) to locate the model path / cache dir if unsure.

- [ ] **Step 1.8: Commit**

```bash
git add src/turbomind/models/llama/llama_params.h \
        src/turbomind/turbomind.h \
        src/turbomind/turbomind.cc \
        src/turbomind/python/bind.cpp
git commit -m "$(cat <<'EOF'
turbomind: expose per-GPU model_tp_rank

Add EngineParam::model_tp_rank = rank(d_tp_group), derived alongside
attn_tp_rank (which is model_tp_rank / attn_cp_size). Expose via
TurboMind::GetModelTpRank and .def("model_tp_rank", ...) in bind.cpp.

No live consumer yet; this is the C++ half of the TextModelBuilder
root-commit refactor (see
docs/superpowers/specs/2026-04-22-text-model-root-commits-design.md).
EOF
)"
```

---

## Task 2: Python `model_tp_ranks` plumbing

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py:38-40`
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:20-39`
- Modify: `lmdeploy/turbomind/deploy/spec.py:125-129`

Goal: carry the new `model_tp_rank` from `bind.cpp` through `BaseOutputModel.tp_ranks` → `TextModelLoader._bind_runtime` → `TextModelSpec.bind_runtime`, storing it as `self._model_tp_ranks` on the spec. Still no live consumer in the builder — Task 3 plugs it in. This commit must pass the existing Qwen3 smoke test unchanged.

- [ ] **Step 2.1: Return `model_tp_rank` from `BaseOutputModel.tp_ranks`**

Edit `lmdeploy/turbomind/deploy/target_model/base.py` lines 38–40:

```python
    def tp_ranks(self, index: int):
        return (self.model_comm.attn_tp_rank(index),
                self.model_comm.mlp_tp_rank(index),
                self.model_comm.model_tp_rank(index))
```

The return tuple length grows from 2 to 3. The only other caller is `TextModelLoader._bind_runtime` (verified by `rg`), and it accesses elements by index — the change is backward-compatible for that usage.

- [ ] **Step 2.2: Collect `model_tp_ranks` in `TextModelLoader._bind_runtime`**

Edit `lmdeploy/turbomind/deploy/text_model_loader.py`. Replace the body of `_bind_runtime` with:

```python
    def _bind_runtime(self):
        model = self.model
        attn_ranks = [model.tp_ranks(gpu)[0]
                      for gpu in range(model.gpu_count)]
        mlp_ranks = [model.tp_ranks(gpu)[1]
                     for gpu in range(model.gpu_count)]
        model_tp_ranks = [model.tp_ranks(gpu)[2]
                          for gpu in range(model.gpu_count)]
        handles = []
        contexts = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
            contexts.append(model.context(gpu))
        model.spec.bind_runtime(
            contexts=contexts,
            root_handles=handles,
            attn_ranks=attn_ranks,
            mlp_ranks=mlp_ranks,
            model_tp_ranks=model_tp_ranks,
        )
```

- [ ] **Step 2.3: Accept `model_tp_ranks` in `TextModelSpec.bind_runtime`**

Edit `lmdeploy/turbomind/deploy/spec.py` lines 125–129:

```python
    def bind_runtime(self, *, contexts, root_handles,
                     attn_ranks, mlp_ranks, model_tp_ranks):
        self._contexts = contexts
        self._root_handles = root_handles
        self._attn_ranks = attn_ranks
        self._mlp_ranks = mlp_ranks
        self._model_tp_ranks = model_tp_ranks
```

- [ ] **Step 2.4: Smoke-test the plumbing**

`self._model_tp_ranks` is stored but unused; existing tests should pass unchanged. Run:

```bash
python scripts/test_turbomind_model.py \
    <qwen3_path> <cache_dir> 1 <gpu_id>
```

Expected: ≥ 128 coherent tokens. If the pipeline import fails with a `TypeError` about `model_tp_ranks`, a subclass of `TextModelSpec` is overriding `bind_runtime`; verify it doesn't (none should, as of spec §2.3).

- [ ] **Step 2.5: Commit**

```bash
git add lmdeploy/turbomind/deploy/target_model/base.py \
        lmdeploy/turbomind/deploy/text_model_loader.py \
        lmdeploy/turbomind/deploy/spec.py
git commit -m "$(cat <<'EOF'
deploy: plumb model_tp_ranks onto TextModelSpec

BaseOutputModel.tp_ranks now returns a 3-tuple (attn, mlp, model).
TextModelLoader._bind_runtime collects the third list per GPU and
forwards it into TextModelSpec.bind_runtime, where it lands as
self._model_tp_ranks. No consumer yet; Task 3 wires it into
TextModelBuilder. See
docs/superpowers/specs/2026-04-22-text-model-root-commits-design.md.
EOF
)"
```

---

## Task 3: Convert `tok_embeddings` + rewrite `TextModelBuilder` / specs (atomic)

**Files:**
- Modify: `src/turbomind/models/model_weight.h:36-45` (X-macros)
- Modify: `src/turbomind/models/model_weight.cc:20-51, 76-86`
- Modify: `src/turbomind/models/language_model.cc:194`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:12, 590-611`
- Delete: `lmdeploy/turbomind/deploy/builder/linear.py`
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py:5-32`
- Modify: `lmdeploy/turbomind/deploy/spec.py:12-14, 177-203`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:105-111`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:148-154`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:153-158`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:114-120`

Goal: move `tok_embeddings` from a `LinearWeight` child to a Tensor param on `ModelWeight`, replace the Python `LinearBuilder` bypass with `TextModelBuilder.add_token_embeds` / `add_lm_head`, delete `builder/linear.py`, and rewrite each spec subclass's `model()` to construct the root with proper `tp` / `ranks` / `vocab_size` / `data_type`. These changes must land together: the C++ X-macro change breaks the Python `LinearBuilder` path, and the Python rewrite assumes `tok_embeddings` is a param slot on the root handle.

### 3A — C++: `tok_embeddings` as Tensor param

- [ ] **Step 3.1: Move `tok_embeddings` in the X-macros**

Edit `src/turbomind/models/model_weight.h` lines 36–43:

```cpp
    // --- X-macro field lists ---
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)           \
    X(tok_embeddings)

    TM_MODULE_DECLARE(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)
```

- [ ] **Step 3.2: Update `ModelWeight::prepare` for Tensor-param lookup**

Edit `src/turbomind/models/model_weight.cc`. In `prepare()` (lines 20–51), change the `vocab_size` derivation to read the tensor shape directly:

```cpp
void ModelWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child) child->prepare();
    });

    auto* l0 = layer(0);
    TM_CHECK(l0);
    // Find first full-attention layer (linear-attn layers have no attention child)
    DecoderLayerWeight* attn_layer = nullptr;
    for (int i = 0; i < (int)layers->size(); ++i) {
        if (layer(i)->attention) {
            attn_layer = layer(i);
            break;
        }
    }
    TM_CHECK(attn_layer) << "No full-attention layer found";
    data_type    = attn_layer->attention->data_type;
    hidden_units = attn_layer->attention->hidden_dim;
    head_dim     = attn_layer->attention->head_dim;
    kv_head_num  = attn_layer->attention->kv_head_num;

    vocab_size        = tok_embeddings.shape(0);     // tensor param now
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);

    layer_types.resize(num_layer);
    for (int i = 0; i < num_layer; ++i) {
        layer_types[i] = layer(i)->linear_attn ? 1 : 0;
    }
}
```

Exactly one line changes: `tok_embeddings->weight.shape(0)` → `tok_embeddings.shape(0)`.

- [ ] **Step 3.3: Update `ModelWeight::verify` for the param**

Edit `src/turbomind/models/model_weight.cc` lines 76–86:

```cpp
bool ModelWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!tok_embeddings) {
        missing.push_back(full_path() + ": missing tok_embeddings");
    }
    if (!norm) {
        missing.push_back(full_path() + ": missing norm");
    }
    return missing.empty();
}
```

No textual change needed — `!tok_embeddings` already works (Tensor's `operator bool()` reports whether the param has a buffer). The line stays as-is; just confirm it is still present after the X-macro change. The X-macro refactor auto-generates the backing field as a `Tensor` (instead of `std::unique_ptr<LinearWeight>`), but the null-check surface is identical.

- [ ] **Step 3.4: Drop `->weight` indirection in `language_model.cc`**

Edit `src/turbomind/models/language_model.cc` line 194:

```cpp
    const int hidden_units = weights_.hidden_units;

    const auto& embedding_table = weights_.tok_embeddings;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units);
```

The `->weight` suffix is dropped: `weights_.tok_embeddings->weight` → `weights_.tok_embeddings`. The rest of `LookupEmbedding` consumes `embedding_table` (a local reference) and is unchanged.

- [ ] **Step 3.5: Build C++**

```bash
cd build && ninja _turbomind
```

Expected: clean build. If anything outside `ModelWeight` was reading `tok_embeddings->weight`, the compile fails — fix by removing the `->weight` there too (no such site is known per the design `grep` check).

### 3B — Python: TextModelBuilder rewrite and spec cleanup

- [ ] **Step 3.6: Import `pad_out_dim` in `builder/_base.py`**

Edit `lmdeploy/turbomind/deploy/builder/_base.py` line 12:

```python
# before:
from ..linear import Linear
# after:
from ..linear import Linear, pad_out_dim
```

- [ ] **Step 3.7: Replace `TextModelBuilder` with the new version**

Edit `lmdeploy/turbomind/deploy/builder/_base.py` lines 590–611 (the section-header comment at 590–592 plus the current `TextModelBuilder` class body at 595–611). Replace wholesale:

```python
# ---------------------------------------------------------------------------
# TextModelBuilder -- wraps pre-existing root handles
# ---------------------------------------------------------------------------


class TextModelBuilder(Builder):
    """Special case Builder that wraps pre-existing root C++ module handles.

    Unlike regular Builders, ``TextModelBuilder`` does NOT create new modules
    in ``__init__``.  The root handles already exist (created by the
    BaseOutputModel / TurboMind runtime).

    Owns ``tok_embeddings`` (Tensor param) and ``output`` (LinearWeight
    child) commits on the root via ``add_token_embeds`` / ``add_lm_head``.
    """

    def __init__(self, handles, contexts, *,
                 tp, ranks, vocab_size, data_type):
        # Bypass Builder.__init__ which calls _tm.create_module.
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
        """Commit the raw embedding lookup as the ``tok_embeddings`` root param.

        Shards along hidden (output) dim by ``self._tp``. No vocab padding —
        embedding lookup never indexes past ``vocab - 1``.
        """
        self._commit_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT)

    def add_lm_head(self, linear):
        """Pad output dim to ``round_up(vocab_size, tp)`` and commit to the
        ``output`` LinearWeight root child.

        Works for every checkpoint format in use today — trivial / AWQ /
        GPTQ / compressed-tensors / MXFP4 all have ``block_out is None``,
        so padding every tensor in the bundle along ``dim=-1`` keeps the
        format-specific block structure intact. FP8 ``lm_head``
        (``block_out == 128``) would misalign scales under naive padding
        but is not a configuration used by any released checkpoint.
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

`SplitSide` is already defined earlier in `_base.py`; `Linear` and `pad_out_dim` come from step 3.6.

- [ ] **Step 3.8: Delete `builder/linear.py`**

```bash
rm lmdeploy/turbomind/deploy/builder/linear.py
```

- [ ] **Step 3.9: Drop `LinearBuilder` / `make_linear_config` exports from `builder/__init__.py`**

Edit `lmdeploy/turbomind/deploy/builder/__init__.py`. Replace with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Builder sub-package — spec-driven module loading for TurboMind."""
from __future__ import annotations

from ._base import (Builder, TextModelBuilder, SplitSide,
                    _cpp_dtype, _act_type_id, _torch_dtype_to_cpp)
from .attention import AttentionBuilder
from .deltanet import DeltaNetBuilder
from .decoder_layer import DecoderLayerBuilder, DecoderLayerConfig
from .ffn import FfnBuilder, fuse_w1w3
from .mla import MLABuilder
from .moe import MoeBuilder
from .module_list import ModuleListBuilder, ModuleListConfig
from .norm import NormBuilder, make_norm_config

__all__ = [
    # Base
    'Builder', 'TextModelBuilder', 'SplitSide',
    '_cpp_dtype', '_act_type_id', '_torch_dtype_to_cpp',
    # Builders
    'AttentionBuilder', 'FfnBuilder', 'MoeBuilder',
    'DeltaNetBuilder', 'MLABuilder',
    'DecoderLayerBuilder', 'ModuleListBuilder',
    'NormBuilder',
    # Primitive config wrappers
    'make_norm_config',
    # C++ config re-exports
    'DecoderLayerConfig', 'ModuleListConfig',
    # Helper functions
    'fuse_w1w3',
]
```

Removed: `LinearBuilder`, `make_linear_config`, the `from .linear import ...` line.

- [ ] **Step 3.10: Delete `token_embeds` / `lm_head` methods and clean imports in `spec.py`**

Edit `lmdeploy/turbomind/deploy/spec.py`. Replace lines 12–14 (imports):

```python
# before:
from .builder import LinearBuilder, SplitSide, _cpp_dtype as _cd
from .builder import make_linear_config
from .linear import pad_out_dim
# after:
from .builder import _cpp_dtype as _cd
```

Update the class docstring at line 34 (it claims the base provides universal primitives that no longer exist):

```python
# before:
      - Factory method NAMES (attn/ffn/moe/linear_attn/mla/norm/...)
        are a convention for readability, NOT a protocol. Signatures may
        differ across subclasses; the base provides no stubs except the
        universal text-model primitives (token_embeds, lm_head).
# after:
      - Factory method NAMES (attn/ffn/moe/linear_attn/mla/norm/...)
        are a convention for readability, NOT a protocol. Signatures
        may differ across subclasses. The base class provides no
        factory stubs; every subclass implements its own model()
        that calls root.add_token_embeds / root.add_lm_head on a
        TextModelBuilder for the root-level commits.
```

Delete the `token_embeds` / `lm_head` methods entirely (lines 177–203 in the original file), along with the now-empty `# Text-model universals` section header banner immediately above them. The file ends cleanly after `_apply_rope`.

(The stale comment `# Primitive config wrappers (still used by default token_embeds/lm_head)` in `builder/__init__.py` is already dropped by the full rewrite in step 3.9.)

### 3C — Spec subclass `model()` rewrites

- [ ] **Step 3.11: Rewrite `Qwen3TextSpec.model` in `qwen3_spec.py`**

Edit `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` lines 105–111:

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

- [ ] **Step 3.12: Rewrite `Qwen3_5Spec.model` in `qwen3_5_spec.py`**

Edit `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` lines 148–154:

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

- [ ] **Step 3.13: Rewrite `Glm4MoeLiteSpec.model` in `glm4_moe_lite_spec.py`**

Edit `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` lines 153–158. GLM-4 never ties embeddings, so the `lm_key` line is unconditional:

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
        root.add_lm_head(self._linear('lm_head'))  # GLM: never tied
        root.layers = self.layers(self._layer_prefix)
```

- [ ] **Step 3.14: Rewrite `GptOssSpec.model` in `gpt_oss_spec.py`**

Edit `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` lines 114–120:

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

### 3D — Build and verify

- [ ] **Step 3.15: Final build sanity check**

```bash
cd build && ninja
```

Step 3.5 already built `_turbomind` after the C++ edits. Steps 3.6–3.14 were Python-only and don't participate in the C++ build graph, so this invocation is typically a no-op — run it anyway to confirm nothing regressed. Expected: "ninja: no work to do." or a clean rebuild of any incremental targets. If any C++ call site still references `tok_embeddings->weight`, fix it (spec-time grep found only `language_model.cc:194`, which was updated in step 3.4).

- [ ] **Step 3.16: Smoke-test trivial Qwen3 at tp=1**

First `get_gpu_usage` for an empty GPU; then:

```bash
python scripts/test_turbomind_model.py \
    <qwen3_path> <cache_dir> 1 <gpu_id>
```

Expected: ≥ 128 coherent tokens. Gibberish ⇒ bisect (most likely `add_token_embeds` shape mismatch or `add_lm_head` data_type).

- [ ] **Step 3.17: Smoke-test trivial Qwen3 at tp=2**

Check two empty GPUs; then:

```bash
python scripts/test_turbomind_model.py \
    <qwen3_path> <cache_dir> 2 <gpu0,gpu1>
```

Expected: ≥ 128 coherent tokens. Validates real sharding of `tok_embeddings` and `output` by `model_tp_rank`.

- [ ] **Step 3.18: Smoke-test an AWQ quantized model at tp=1**

Pick an AWQ model from the local cache (query `model-server` MCP for `list_models`). Then:

```bash
python scripts/test_turbomind_model.py \
    <awq_path> <cache_dir> 1 <gpu_id>
```

Expected: ≥ 128 coherent tokens. Validates the `model_dtype` threading in `add_lm_head` (AWQ per-layer linears use `self._linear(...)` through `_commit_linear` with the correct compute dtype; the new `add_lm_head` forwards it identically).

- [ ] **Step 3.19: Smoke-test a tied-embeddings model at tp=1**

Pick a model where `tie_word_embeddings` is true (most small Qwen3 variants; query HF config or `get_model_config` MCP to confirm). Then:

```bash
python scripts/test_turbomind_model.py \
    <tied_path> <cache_dir> 1 <gpu_id>
```

Expected: ≥ 128 coherent tokens. Validates that the same checkpoint tensor, consumed once via `_get(key)` for `tok_embeddings` and once via `_linear(key.removesuffix('.weight'))` for `output`, produces a coherent model.

- [ ] **Step 3.20: Smoke-test GLM4-MoE-Lite (MLA) at tp=2**

```bash
python scripts/test_turbomind_model.py \
    <glm4_moe_lite_path> <cache_dir> 2 <gpu0,gpu1>
```

Expected: ≥ 128 coherent tokens. Validates the unconditional-`'lm_head'` path and MLA attention (unaffected by this refactor).

- [ ] **Step 3.21: Smoke-test Qwen3.5 (DeltaNet + multimodal prefix) at tp=1**

```bash
python scripts/test_turbomind_model.py \
    <qwen3_5_path> <cache_dir> 1 <gpu_id>
```

Expected: ≥ 128 coherent tokens. Validates multi-modal `_embed_key` handling (`'model.language_model.embed_tokens.weight'` → `.removesuffix('.weight')` → `'model.language_model.embed_tokens'`).

- [ ] **Step 3.22: (If hardware allows) Smoke-test cp > 1**

Only run if a model + hardware configuration with `attn_cp_size > 1` is available. The cp>1 path is intended to work end-to-end after this refactor. If the available config is cp=2, attn_tp=2 on 4 GPUs:

```bash
python scripts/test_turbomind_model.py \
    <model_path> <cache_dir> 4 <gpu0,gpu1,gpu2,gpu3>
```

(Adjust the CLI flags to enable cp if the script supports it, or configure via `engine_config` in a custom harness.)

Expected: ≥ 128 coherent tokens. Before this refactor, cp > 1 silently garbled output; after, this is the definitive regression test that the `model_tp_rank` fix is correct. If no cp > 1 setup is available, skip this step — the cp=1 tests above cover the non-CP paths and the `model_tp_rank` value collapses to `attn_tp_rank` anyway.

- [ ] **Step 3.23: Commit**

```bash
git add src/turbomind/models/model_weight.h \
        src/turbomind/models/model_weight.cc \
        src/turbomind/models/language_model.cc \
        lmdeploy/turbomind/deploy/builder/_base.py \
        lmdeploy/turbomind/deploy/builder/__init__.py \
        lmdeploy/turbomind/deploy/spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git rm lmdeploy/turbomind/deploy/builder/linear.py
git commit -m "$(cat <<'EOF'
deploy: TextModelBuilder owns tok_embeddings/output commits

Move tok_embeddings from a LinearWeight child to a Tensor parameter on
ModelWeight. Drop the LinearBuilder/make_linear_config/set_weight
bypass in TextModelSpec.token_embeds/lm_head and delete builder/linear.py.
Add TextModelBuilder.add_token_embeds(tensor) and add_lm_head(linear)
with required kwargs (tp, ranks, vocab_size, data_type); rewrite every
source_model/*_spec.py model() to construct the root with
tp=attn_tp*attn_cp and ranks=self._model_tp_ranks.

lm_head now flows through self._linear(prefix) like every other linear
(with model_dtype threaded for quantized support). tok_embeddings is
unpadded along vocab; the previously-padded rows were dead storage.

Using model_tp_rank (exposed in the earlier commit) for the
attn_tp*attn_cp split fixes the cp>1 bug end-to-end: previously
self._attn_ranks (in [0, attn_tp)) couldn't index all attn_tp*cp
shards, so (cp-1)*attn_tp of the shards were never committed and
CP peers held identical (wrong) content.

See docs/superpowers/specs/2026-04-22-text-model-root-commits-design.md.
EOF
)"
```

---

## Commit summary

After completing all tasks, `git log --oneline` should show three new commits on the feature branch:

```
<sha> deploy: TextModelBuilder owns tok_embeddings/output commits
<sha> deploy: plumb model_tp_ranks onto TextModelSpec
<sha> turbomind: expose per-GPU model_tp_rank
```
