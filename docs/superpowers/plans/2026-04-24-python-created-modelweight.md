# Python-Created ModelWeight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `ModelWeight` onto the standard `_tm.create_module(cfg)` path by introducing a new C++ `ModelRoot` sentinel that lives in `TurboMind::Impl::weights_` and holds the Python-created `ModelWeight` as its `text_model` child. Drop `TextModelBuilder`'s `_create_handles` no-op override. Narrow `Builder.__setattr__`'s post-build guard to the `BuiltModule` branch, eliminating `object.__setattr__` from all ordinary builder code.

**Architecture:** New `ModelRoot: core::Module` owns the weight-loading CUDA stream + pool-backed allocator (moved off `ModelWeight`) and declares one child slot `text_model : ModelWeight` via `TM_MODULE_DECLARE`. `TurboMind::CreateWeights` splits into `CreateContext` (promoted from private) + `CreateRoot` (new — returns the sentinel). `ModelWeight` gets a `ModelWeightConfig` + `ModuleRegistry` entry; its ctor now takes the config. Python's `TextModelBuilder` becomes a regular `Builder`: `super().build()` creates the `ModelWeight` via standard `_tm.create_module`, then a loop calls `root.add_child_raw('text_model', text_model)` on each sentinel.

**Tech Stack:** C++ (`src/turbomind`), CUDA via `ninja` from `build/`, pybind11 (`src/turbomind/python/bind.cpp`), Python (`lmdeploy/turbomind`). No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-04-24-python-created-modelweight-design.md`

---

## File Structure

### New C++ files

- **Create:** `src/turbomind/models/model_root.h` — declares `ModelRoot : core::Module`, one child slot `text_model : ModelWeight`, `context()` / `stream()` / `allocator()` / `text_model_ptr()` accessors, `prepare()` override.
- **Create:** `src/turbomind/models/model_root.cc` — ctor creates stream + allocator, `prepare()` asserts `text_model` is attached then recurses via `Module::prepare()`, `TM_MODULE_METHODS` for the one child.

### Modified C++ files

- **Modify:** `src/turbomind/models/CMakeLists.txt` — add `model_root.cc` to `TM_MODELS_SRC`.
- **Modify:** `src/turbomind/models/model_weight.h` — add `core::ModelWeightConfig` with `tp_size`/`tp_rank`; change ctor to take the config; remove `context()`/`stream()`/`allocator()` accessors and the `stream_`/`alloca_` private fields; drop `<llama_params.h>` include.
- **Modify:** `src/turbomind/models/model_weight.cc` — rewrite ctor to init from config; add anonymous-namespace `ModuleRegistrar` for `"ModelWeight"`.
- **Modify:** `src/turbomind/turbomind.h` — replace `CreateWeights` with `CreateContext` + `CreateRoot` in the public API.
- **Modify:** `src/turbomind/turbomind.cc` — `weights_` element type → `shared_ptr<ModelRoot>`; promote private `CreateContext` to public `Impl::CreateContext`; add `Impl::CreateRoot`; rewrite every `weights_[i]->field` / `*weights_[i]` read site through `text_model_ptr()`; update `weight_context(index)` to read stream/alloca off `ModelRoot`; include `model_root.h`.
- **Modify:** `src/turbomind/python/bind.cpp` — add `bind_config<core::ModelWeightConfig>`; replace `.def("create_weights", ...)` with `.def("create_context", ...)` + `.def("create_root", ...)`.

### Modified Python files

- **Modify:** `lmdeploy/turbomind/deploy/builder/_base.py` — narrow `Builder.__setattr__`'s `_built` guard to the `BuiltModule` branch; drop `object.__setattr__` from `Builder.__init__` and `build()`; refactor `TextModelBuilder` to take `config` + `root_handles=`, delete its `_create_handles` override, add a `build()` override that attaches to sentinel roots.
- **Modify:** `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — `model()` prologue constructs a `_tm.ModelWeightConfig()`; `TextModelBuilder(...)` call takes the cfg positionally and `root_handles=` as kwarg.
- **Modify:** `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — same.
- **Modify:** `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — same.
- **Modify:** `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — same.
- **Modify:** `lmdeploy/turbomind/turbomind.py` — `_create_weight` per-device lambda now calls `create_context(device_id)` then `create_root(device_id)`.

### Not modified

- `lmdeploy/turbomind/deploy/text_model_loader.py` — `model.root(gpu)` return type is `Module*` regardless of whether it's a `ModelWeight` or `ModelRoot`; the Python consumer only uses `add_child_raw`/`child`.
- `lmdeploy/turbomind/deploy/spec.py` — `TextModelSpec.bind_runtime` signature unchanged; `self._root_handles` now refers to sentinel roots but semantically plays the same role.
- No unit tests for this subsystem per `AGENTS.md`.

---

## Task 1: Atomic refactor

The tree is broken mid-task (C++ type change at step 5/7 and pybind API split at step 8 require Python-side updates in steps 9/14 to run end-to-end). Complete all 14 code edits before building or running any smoke test. Commit once at the end of Task 1 after the tp=2 smoke test passes.

**Files in this task:**
- Create: `src/turbomind/models/model_root.h`
- Create: `src/turbomind/models/model_root.cc`
- Modify: `src/turbomind/models/CMakeLists.txt`
- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`
- Modify: `src/turbomind/turbomind.h`
- Modify: `src/turbomind/turbomind.cc`
- Modify: `src/turbomind/python/bind.cpp`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- Modify: `lmdeploy/turbomind/turbomind.py`

---

- [ ] **Step 1: Create `src/turbomind/models/model_root.h`**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/model_weight.h"

namespace turbomind {

/// Sentinel root for the weight tree.  Lives in TurboMind::Impl::weights_
/// and owns the CUDA stream + pool-backed allocator used during weight
/// loading.  Python creates a ModelWeight via _tm.create_module and
/// attaches it as the `text_model` child via add_child_raw.
class ModelRoot: public core::Module {
public:
    const char* type() const override { return "ModelRoot"; }

    ModelRoot();
    ~ModelRoot() override;

    void prepare() override;

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    const core::Stream&    stream() const    { return stream_; }
    const core::Allocator& allocator() const { return alloca_; }

    /// Convenience accessor.  Nullptr before Python attaches via
    /// `add_child_raw('text_model', ...)`.
    ModelWeight* text_model_ptr() const { return text_model.get(); }

#define MODEL_ROOT_CHILDREN(X) \
    X(ModelWeight, text_model)

#define MODEL_ROOT_PARAMS(X)

    TM_MODULE_DECLARE(ModelRoot, MODEL_ROOT_CHILDREN, MODEL_ROOT_PARAMS)

private:
    core::Stream    stream_{};
    core::Allocator alloca_{};
};

}  // namespace turbomind
```

- [ ] **Step 2: Create `src/turbomind/models/model_root.cc`**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_root.h"
#include "src/turbomind/core/check.h"

namespace turbomind {

ModelRoot::ModelRoot()
{
    // CUDA device is already set by CudaDeviceGuard in TurboMind::CreateRoot.
    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, /*use_default_pool=*/true};
}

ModelRoot::~ModelRoot() = default;

void ModelRoot::prepare()
{
    TM_CHECK(text_model)
        << "ModelRoot::prepare: text_model not attached; did the spec "
           "forget root.build()?";
    Module::prepare();
}

TM_MODULE_METHODS(ModelRoot, MODEL_ROOT_CHILDREN, MODEL_ROOT_PARAMS)

}  // namespace turbomind
```

- [ ] **Step 3: Add `model_root.cc` to `src/turbomind/models/CMakeLists.txt`**

Insert `model_root.cc` between `decoder_layer_weight.cc` and `model_weight.cc` so related sources stay grouped. Edit: change

```cmake
        decoder_layer_weight.cc
        model_weight.cc
```

to

```cmake
        decoder_layer_weight.cc
        model_weight.cc
        model_root.cc
```

- [ ] **Step 4: Rewrite `src/turbomind/models/model_weight.h`**

Replace the entire file contents with:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include <vector>

namespace turbomind::core {

struct ModelWeightConfig: ModuleConfig {
    ModelWeightConfig(): ModuleConfig{"ModelWeight"} {}

#define MODEL_WEIGHT_FIELDS(X) \
    X(int, tp_size) \
    X(int, tp_rank)

    MODEL_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(ModelWeightConfig, MODEL_WEIGHT_FIELDS)

#undef MODEL_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class DecoderLayerWeight;

/// Root weight module for a model. Owns the full weight tree.
class ModelWeight: public core::Module {
public:
    const char* type() const override { return "ModelWeight"; }

    ModelWeight() = default;

    explicit ModelWeight(const core::ModelWeightConfig& cfg);

    void                    prepare() override;
    bool                    verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)           \
    X(tok_embeddings)

    TM_MODULE_DECLARE(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)

    // --- Accessors ---
    DecoderLayerWeight*               layer(int i) const;
    std::vector<DecoderLayerWeight*>  layers_list() const;

    // --- Derived in prepare() from children -- public for direct access ---
    DataType    data_type{};
    int         hidden_units{};
    int         vocab_size{};
    int         vocab_size_padded{};
    int         embedding_size{};
    int         num_layer{};
    int         head_dim{};
    int         kv_head_num{};
    std::vector<int> layer_types;

    // --- From ModelWeightConfig at construction ---
    int         tp_size{};
    int         tp_rank{};

private:
    mutable std::vector<DecoderLayerWeight*> layers_cache_;
};

}  // namespace turbomind
```

Three deletions vs. today's file:
- `#include "src/turbomind/models/llama/llama_params.h"` — no longer referencing `EngineParam`.
- `core::ContextGuard context() const`, `stream() const`, `allocator() const` — moved to `ModelRoot`.
- `core::Stream stream_{}`, `core::Allocator alloca_{}` — ditto.

Plus one addition: `struct ModelWeightConfig` in the `turbomind::core` namespace before the `turbomind` namespace.

- [ ] **Step 5: Rewrite `src/turbomind/models/model_weight.cc`**

Replace the entire file contents with:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/core/registry.h"

namespace turbomind {

ModelWeight::ModelWeight(const core::ModelWeightConfig& cfg)
    : tp_size(cfg.tp_size)
    , tp_rank(cfg.tp_rank)
{
    // Stream/allocator moved to ModelRoot; nothing to allocate here.
}

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

    vocab_size        = tok_embeddings.shape(0);
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = TM_CHECK_NOTNULL(output)->output_dim * tp_size;

    layer_types.resize(num_layer);
    for (int i = 0; i < num_layer; ++i) {
        layer_types[i] = layer(i)->linear_attn ? 1 : 0;
    }
}

DecoderLayerWeight* ModelWeight::layer(int i) const
{
    if (!layers) {
        return nullptr;
    }
    return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
}

std::vector<DecoderLayerWeight*> ModelWeight::layers_list() const
{
    if (!layers_cache_.empty()) {
        return layers_cache_;
    }
    if (!layers) {
        return {};
    }
    layers_cache_.resize(layers->size());
    for (int i = 0; i < layers->size(); ++i) {
        layers_cache_[i] = static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
    }
    return layers_cache_;
}

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

namespace {
struct ModelWeightRegistrar {
    ModelWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "ModelWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<ModelWeight>(
                    static_cast<const core::ModelWeightConfig&>(cfg));
            });
    }
};
static ModelWeightRegistrar _model_weight_reg;
}  // anonymous namespace

TM_MODULE_METHODS(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)

}  // namespace turbomind
```

Two changes vs. today's file:
- Ctor body: replaces `stream_ = core::Stream::create(); alloca_ = core::Allocator{...}` with a comment and `tp_size(cfg.tp_size), tp_rank(cfg.tp_rank)` in the initializer list.
- New anonymous-namespace `ModelWeightRegistrar` registering `"ModelWeight"` with the registry.

The `#include "src/turbomind/core/registry.h"` is needed for `ModuleRegistry::instance()`.

- [ ] **Step 6: Update `src/turbomind/turbomind.h` public API**

Replace this block at lines 26 in the class body:

```cpp
    void CreateWeights(int index);
```

with:

```cpp
    void          CreateContext(int index);
    core::Module* CreateRoot(int index);
```

- [ ] **Step 7: Rewrite relevant parts of `src/turbomind/turbomind.cc`**

Five edits in this file. Complete all five before moving to step 8.

**7a.** Add the new include near the existing `model_weight.h` include (around line 20):

```cpp
#include "src/turbomind/models/model_root.h"
```

**7b.** Change the type of `Impl::weights_` at the declaration (line 56):

From:
```cpp
    vector<shared_ptr<ModelWeight>> weights_;
```

To:
```cpp
    vector<shared_ptr<ModelRoot>>   weights_;
```

**7c.** Replace `Impl::CreateWeights` (lines 81-88):

From:
```cpp
    void CreateWeights(int index)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);

        CreateContext(index);

        weights_[index] = std::make_shared<ModelWeight>(engine_params_.at(index));
    }
```

To:
```cpp
    core::Module* CreateRoot(int index)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);
        TM_CHECK(contexts_[index] != nullptr)
            << "CreateContext(" << index << ") must run before CreateRoot";
        weights_[index] = std::make_shared<ModelRoot>();
        return weights_[index].get();
    }
```

**7d.** Add the public `TurboMind::CreateContext` + `CreateRoot` thunks. `Impl::CreateContext` already exists (declared at line 102, body at line 193); no changes to it. The new public methods just forward to `Impl::CreateContext` / `Impl::CreateRoot`.

Replace the existing public `TurboMind::CreateWeights` thunk (lines 447-450):

```cpp
void TurboMind::CreateWeights(int index)
{
    return impl_->CreateWeights(index);
}
```

With:

```cpp
void TurboMind::CreateContext(int index)
{
    return impl_->CreateContext(index);
}

core::Module* TurboMind::CreateRoot(int index)
{
    return impl_->CreateRoot(index);
}
```

**7e.** Rewrite every read site that dereferenced `ModelWeight` fields through `weights_[...]` — they now go through `text_model_ptr()`. Concretely:

- Line 77 (inside `CreateRequest`): `weights_[0]->vocab_size` → `weights_[0]->text_model_ptr()->vocab_size`
- Line 78: `weights_[0]->hidden_units` → `weights_[0]->text_model_ptr()->hidden_units`
- Line 280 (inside `CreateEngine`, LanguageModel ctor arg): `*weights_[index]` → `*weights_[index]->text_model_ptr()`
- Line 286 (inside `CreateEngine`, Engine ctor arg): `*weights_[index]` → `*weights_[index]->text_model_ptr()`
- Line 368 (inside `WarmUp`): `weights_[index]->vocab_size` → `weights_[index]->text_model_ptr()->vocab_size`

Unchanged: the `ProcessWeights` block at lines 90-97 stays verbatim — `weights_[index]->context()` and `weights_[index]->prepare()` both now resolve on `ModelRoot` (with the same signatures). The `~Impl` teardown order at lines 137-146 is unchanged; `weights_[i] = {};` still works.

**7f.** Update `weight_context` (line 457-462). Replace:

```cpp
std::pair<core::Stream, core::Allocator> TurboMind::weight_context(int index)
{
    auto& mw = impl_->weights_.at(index);
    TM_CHECK(mw != nullptr);
    return {mw->stream(), mw->allocator()};
}
```

With:

```cpp
std::pair<core::Stream, core::Allocator> TurboMind::weight_context(int index)
{
    auto& root = impl_->weights_.at(index);
    TM_CHECK(root != nullptr);
    return {root->stream(), root->allocator()};
}
```

Functionally identical — `ModelRoot::stream()` / `ModelRoot::allocator()` have the same return types as the removed `ModelWeight` accessors; only the variable name changes for clarity.

- [ ] **Step 8: Update `src/turbomind/python/bind.cpp`**

Two edits.

**8a.** Add the new config binding at line 461 (right after `bind_config<turbomind::core::DecoderLayerConfig>`):

```cpp
    bind_config<turbomind::core::ModelWeightConfig>(m, "ModelWeightConfig");
```

**8b.** Replace the `.def("create_weights", ...)` method at line 698. From:

```cpp
        .def("create_weights", &TurboMind::CreateWeights, py::call_guard<py::gil_scoped_release>(), "index"_a)
```

To:

```cpp
        .def("create_context",
             &TurboMind::CreateContext,
             py::call_guard<py::gil_scoped_release>(),
             "index"_a)
        .def("create_root",
             [](TurboMind* model, int index) -> ft::core::Module* {
                 return model->CreateRoot(index);
             },
             py::return_value_policy::reference,
             py::call_guard<py::gil_scoped_release>(),
             "index"_a)
```

The `create_root` lambda mirrors the existing `root` method's `reference` policy since the returned `Module*` is owned by the `shared_ptr` in `Impl::weights_`.

- [ ] **Step 9: Rewrite `Builder` and `TextModelBuilder` in `lmdeploy/turbomind/deploy/builder/_base.py`**

Five localized edits in the file, all inside the classes at the bottom (starting at line 347).

**9a.** Replace `Builder.__init__` body (lines 361-386). From:

```python
    def __init__(self, config, contexts, tp=1, ranks=None):
        """Initialise the builder with staging dicts (no C++ creation yet).
        ...
        """
        # Fields read unconditionally by __setattr__ (e.g. the _built guard)
        # must be initialized via object.__setattr__ so that subsequent normal
        # assignment (self.x = y) can pass through __setattr__ without error.
        object.__setattr__(self, '_built', False)
        self._contexts = contexts
        self._tp = tp
        self._ranks = ranks
        self.config = config
        self._pending_linears = {}
        self._pending_tensors = {}
        self._pending_children = {}
        self._handles = None
```

To:

```python
    def __init__(self, config, contexts, tp=1, ranks=None):
        """Initialise the builder with staging dicts (no C++ creation yet).

        Parameters
        ----------
        config : C++ config struct
            Config with ``clone()`` method and optionally ``tp_rank`` field.
        contexts : list
            GPU context managers (one per GPU).
        tp : int
            Tensor parallelism degree.
        ranks : list[int] | None
            Per-GPU TP ranks.
        """
        # `_built` must be set first: __setattr__ reads it inside the
        # BuiltModule branch.  Bool is not a BuiltModule, so the normal
        # fall-through assigns it via object.__setattr__ at the end of
        # __setattr__.
        self._built = False
        self._contexts = contexts
        self._tp = tp
        self._ranks = ranks
        self.config = config
        self._pending_linears = {}
        self._pending_tensors = {}
        self._pending_children = {}
        self._handles = None
```

**9b.** Replace `Builder.__setattr__` (lines 392-403). From:

```python
    def __setattr__(self, name: str, value):
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
```

To:

```python
    def __setattr__(self, name: str, value):
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}.{name}: assign .build() output "
                f"(BuiltModule), not the Builder itself")
        if isinstance(value, BuiltModule):
            if self._built:
                raise RuntimeError(
                    f"{type(self).__name__} is built; "
                    f"cannot assign {name!r}")
            self._pending_children[name] = value.handles
            return
        object.__setattr__(self, name, value)
```

The guard moves inside the `BuiltModule` branch (the only branch that stages a child). Plain post-build attribute writes fall through to `object.__setattr__` and are silently accepted — they don't stage anything and don't touch C++ state. `__setitem__` (line 405) keeps its top-level `if self._built` guard since every `__setitem__` call stages a child.

**9c.** In `Builder.build()` (lines 458-483), replace the `_built` flip. From:

```python
        self._create_handles()

        # Bypass the frozen-after-build guard for the state transition itself
        object.__setattr__(self, '_built', True)
```

To:

```python
        self._create_handles()

        # True is not BuiltModule; falls through to plain assignment.
        self._built = True
```

Rest of `build()` body (the three drain loops and the `return BuiltModule(...)`) unchanged.

**9d.** Replace `TextModelBuilder.__init__` (lines 655-661). From:

```python
    def __init__(self, handles, contexts, *,
                 tp, ranks, vocab_size, data_type):
        # Delegate to Builder.__init__ with config=None (no create_module).
        super().__init__(config=None, contexts=contexts, tp=tp, ranks=ranks)
        self._vocab_size = vocab_size
        self._data_type = data_type
        self._handles = handles
```

To:

```python
    def __init__(self, config, contexts, *, root_handles,
                 tp, ranks, vocab_size, data_type):
        super().__init__(config=config, contexts=contexts, tp=tp, ranks=ranks)
        self._root_handles = root_handles
        self._vocab_size = vocab_size
        self._data_type = data_type
```

Signature changes: `handles` (positional) → `config` (positional) + `root_handles` (keyword). `super().__init__` now passes the real `config` instead of `None`. The `self._handles = handles` line is gone — `super()._create_handles()` will populate `self._handles` during `build()`.

**9e.** Replace `TextModelBuilder._create_handles` (lines 663-666). From:

```python
    def _create_handles(self):
        """Root handles already exist — no-op."""
        assert self._handles is not None, (
            "TextModelBuilder._handles must be set before build()")
```

To (delete the method entirely, and add a `build()` override instead):

```python
    def build(self) -> BuiltModule:
        """Create ModelWeight via _tm.create_module (via super), then
        attach each per-GPU ModelWeight handle to its sentinel root
        via add_child_raw.
        """
        built = super().build()
        for i, (root, text_model) in enumerate(
                zip(self._root_handles, built.handles)):
            with self._contexts[i]:
                root.add_child_raw('text_model', text_model)
        return built
```

After these five edits the class docstring at line 645 is stale (it describes the old "wraps pre-existing root C++ module handles" behavior). Update it to:

```python
class TextModelBuilder(Builder):
    """Builder for the root ModelWeight.

    Constructs a ModelWeight via ``_tm.create_module(ModelWeightConfig)``
    on each context (inherited Builder machinery), then attaches it to
    externally-owned ``ModelRoot`` sentinel handles as their
    ``text_model`` child during ``build()``.

    Owns ``tok_embeddings`` (Tensor param) and ``output`` (LinearWeight
    child) commits on the ModelWeight via ``add_token_embeds`` /
    ``add_lm_head``.
    """
```

The `add_token_embeds` and `add_lm_head` bodies (lines 668-698) are unchanged.

- [ ] **Step 10: Update `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py::model()`**

Replace the `model()` method body (lines 100-113). From:

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
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

To:

```python
    def model(self):
        ec = self.engine_cfg
        cfg = _tm.ModelWeightConfig()
        cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
        root = TextModelBuilder(
            cfg, self._contexts,
            root_handles=self._root_handles,
            tp=ec.attn_tp_size * ec.attn_cp_size,
            ranks=self._model_tp_ranks,
            vocab_size=self._vocab_size,
            data_type=self._cpp_dtype())
        root.add_token_embeds(self._get(self._embed_key))
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

Two lines added (`cfg = _tm.ModelWeightConfig(); cfg.tp_size = ...`); the `TextModelBuilder(...)` call's first two args change from `(self._root_handles, self._contexts, ...)` to `(cfg, self._contexts, root_handles=self._root_handles, ...)`. Everything else byte-for-byte unchanged.

- [ ] **Step 11: Update `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py::model()`**

Replace the `model()` method body (lines 141-154). From:

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
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

To:

```python
    def model(self):
        ec = self.engine_cfg
        cfg = _tm.ModelWeightConfig()
        cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
        root = TextModelBuilder(
            cfg, self._contexts,
            root_handles=self._root_handles,
            tp=ec.attn_tp_size * ec.attn_cp_size,
            ranks=self._model_tp_ranks,
            vocab_size=self._vocab_size,
            data_type=self._cpp_dtype())
        root.add_token_embeds(self._get(self._embed_key))
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

- [ ] **Step 12: Update `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py::model()`**

Replace the `model()` method body (lines 107-120). From:

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
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

To:

```python
    def model(self):
        ec = self.engine_cfg
        cfg = _tm.ModelWeightConfig()
        cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
        root = TextModelBuilder(
            cfg, self._contexts,
            root_handles=self._root_handles,
            tp=ec.attn_tp_size * ec.attn_cp_size,
            ranks=self._model_tp_ranks,
            vocab_size=self._vocab_size,
            data_type=self._cpp_dtype())
        root.add_token_embeds(self._get(self._embed_key))
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

- [ ] **Step 13: Update `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py::model()`**

Replace the `model()` method body (lines 148-160). GLM uses a different `lm_head` call than the other three — it always calls `self._linear('lm_head')` (no tied-embed branch). From:

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
        root.norm = self.norm(self._get(self._norm_key))
        root.add_lm_head(self._linear('lm_head'))  # GLM: never tied
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

To:

```python
    def model(self):
        ec = self.engine_cfg
        cfg = _tm.ModelWeightConfig()
        cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
        root = TextModelBuilder(
            cfg, self._contexts,
            root_handles=self._root_handles,
            tp=ec.attn_tp_size * ec.attn_cp_size,
            ranks=self._model_tp_ranks,
            vocab_size=self._vocab_size,
            data_type=self._cpp_dtype())
        root.add_token_embeds(self._get(self._embed_key))
        root.norm = self.norm(self._get(self._norm_key))
        root.add_lm_head(self._linear('lm_head'))  # GLM: never tied
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

- [ ] **Step 14: Update `lmdeploy/turbomind/turbomind.py::_create_weight`**

Replace `_create_weight` (lines 192-204). From:

```python
    def _create_weight(self, model_comm):
        """Allocate weight buffer, load params if from_workspace."""

        # create weight
        def _create_weight_func(device_id):
            model_comm.create_weights(device_id)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_create_weight_func, device_id))
            for future in futures:
                future.result()
```

To:

```python
    def _create_weight(self, model_comm):
        """Create per-GPU Context + empty ModelRoot sentinel.

        Runs both C++ init steps sequentially per device, inside a
        ThreadPoolExecutor so all ranks enter ``create_context``
        concurrently and hit its ``h_global->Sync()`` barriers together.
        ``create_root`` itself has no collectives, so it can follow
        synchronously on each thread.
        """

        def _create_weight_func(device_id):
            model_comm.create_context(device_id)
            model_comm.create_root(device_id)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_create_weight_func, device_id))
            for future in futures:
                future.result()
```

- [ ] **Step 15: Build**

```bash
cd build && ninja
```

Expected: build succeeds. If errors about `ModelWeight(const EngineParam&)` appear, step 5 didn't land — the old constructor is still being matched. If errors about `weights_[i]->vocab_size`, step 7e missed a site — grep `weights_\[` in `turbomind.cc` and make sure every access goes through `text_model_ptr()` (except in `ProcessWeights` / `~Impl` / the setter in `CreateRoot`). If `CreateWeights` is undefined, step 6 or 7c was skipped.

- [ ] **Step 16: Smoke test — Qwen3 dense, tp=1**

Before running:
1. Call MCP tool `user-model-server::list_models` to find a Qwen3 dense model.
2. Call `user-model-server::get_model_cache_path` for the chosen model; note the `cache_dir`.
3. Call MCP tool `user-gpu-monitor::get_gpu_usage` to pick one empty GPU (replace `0` with it in the command).

Run:

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> 1 0
```

Expected: the `--- response begin ---` … `--- response end ---` block contains at least 128 tokens of coherent text relevant to the script's prompt. If gibberish or a Python exception at `root.build()`:
- `TypeError: __init__() got an unexpected keyword argument 'root_handles'` → step 9d didn't land.
- `AttributeError: module '_turbomind' has no attribute 'ModelWeightConfig'` → step 8a didn't land.
- `RuntimeError: model_comm has no attribute 'create_context'` → step 8b or step 14 didn't land.
- `TM_CHECK` abort "text_model not attached" → `root.build()` didn't run; verify step 10-13 kept `root.build()` as the last line of each `model()`.
- Gibberish → probably a field mismatch in `ModelWeightConfig` (wrong `tp_size`); recheck step 10.

**DO NOT commit until the response is coherent.**

- [ ] **Step 17: Smoke test — Qwen3 dense, tp=2**

Before running: pick two empty GPUs via `get_gpu_usage` (e.g. `0,1`).

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent response. This exercises `Builder._cfg_for_rank`'s per-rank `tp_rank` push onto the `ModelWeightConfig` clone — confirms the registered factory + config flow works for multi-GPU. If gibberish or a CUDA error, check that `ModelWeight::tp_size` = 2 at runtime (add a temporary log in `ModelWeight::prepare`); if it's 0 or 1, the `cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size` assignment in the spec didn't survive.

**DO NOT commit until this passes.**

- [ ] **Step 18: Commit**

```bash
git add src/turbomind/models/model_root.h \
        src/turbomind/models/model_root.cc \
        src/turbomind/models/CMakeLists.txt \
        src/turbomind/models/model_weight.h \
        src/turbomind/models/model_weight.cc \
        src/turbomind/turbomind.h \
        src/turbomind/turbomind.cc \
        src/turbomind/python/bind.cpp \
        lmdeploy/turbomind/deploy/builder/_base.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py \
        lmdeploy/turbomind/turbomind.py

git commit -m "$(cat <<'EOF'
tm: create ModelWeight from Python via ModelRoot sentinel

Promote ModelWeight onto the standard _tm.create_module path. Add a
new C++ ModelRoot sentinel that lives in TurboMind::Impl::weights_,
owns the weight-loading stream+allocator (moved off ModelWeight), and
declares a single child slot text_model : ModelWeight. Python creates
a ModelWeight via _tm.create_module(ModelWeightConfig) and attaches it
to the sentinel via add_child_raw during TextModelBuilder.build().

ModelWeight loses its EngineParam ctor in favour of ModelWeightConfig
(tp_size, tp_rank) and gains a ModuleRegistry entry. TurboMind splits
CreateWeights into public CreateContext + CreateRoot; the python
turbomind.py _create_weight driver calls both per GPU.

TextModelBuilder becomes a regular Builder; its _create_handles
override is deleted, its build() is extended to attach the built
ModelWeight to each sentinel. Each spec's model() adds two lines to
construct a _tm.ModelWeightConfig and passes it as the first positional
arg, with root_handles moved to a keyword.

Builder.__setattr__'s post-build guard narrows to the BuiltModule
branch — plain post-build attribute writes now silently succeed (no
staging, no C++ mutation). This eliminates object.__setattr__ from
all ordinary Builder code.
EOF
)"
```

---

## Task 2: Full verification matrix

Run the complete model matrix from the spec's Verification section to catch anything Task 1's single smoke test missed. This task makes **no code changes**.

Before each row:
1. MCP `user-gpu-monitor::get_gpu_usage` to pick empty GPUs.
2. MCP `user-model-server::list_models` + `get_model_config` / `get_model_cache_path` to locate the model.
3. Confirm the model's `model_type` in HF config matches the row's target.

- [ ] **Step 1: Qwen3-MoE, tp=2**

Pick any Qwen3-MoE checkpoint (HF config has `num_local_experts > 0` or `model_type == 'qwen3_moe'`). Run at tp=2:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent text. Exercises `MoeBuilder` + staged experts `ModuleListBuilder` + sentinel attachment.

- [ ] **Step 2: Qwen3.5 linear-attention variant, tp=2**

Pick a Qwen3.5 model whose HF config has `layer_types` containing `linear_attention`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens. Exercises `DeltaNetBuilder`, zero-centered norm override, MoE shared-expert path through the new sentinel.

- [ ] **Step 3: GPT-OSS (mxfp4), tp=2**

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens. Exercises MXFP4 + packed experts + sliding-window attention. `ModelWeight::prepare`'s `vocab_size_padded = output->output_dim * tp_size` should work identically (the `output` child is created by the spec's `add_lm_head` and attached through the normal `_pending_children` loop).

- [ ] **Step 4: GLM-4 MoE Lite (MLA), tp=2**

Pick a GLM-4 MoE Lite checkpoint (MLA):

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens. Exercises the MLA `kv_head_num = tp_size` nudge (moved to Python in the deferred-creation refactor) under the new root lifecycle. A crash here with missing MLA tensors indicates the cloned cfg path in `Glm4MoeLiteSpec.attn` didn't survive step 13.

- [ ] **Step 5: Qwen3 + AWQ, tp=1**

Pick a Qwen3 AWQ checkpoint (`quantization_config.quant_method == "awq"`):

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens. Exercises quantized w1w3 fusion through the new `TextModelBuilder.build()` path.

- [ ] **Step 6: Qwen3 + FP8, tp=1**

Pick a Qwen3-FP8 checkpoint (`quantization_config.quant_method == "fp8"` or similar):

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens. FP8 format + `_ensure_compatible_formats` dequant path under the new lifecycle.

- [ ] **Step 7: Qwen3 + GPTQ, tp=1**

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens. GPTQ + `synthesize_zeros` path.

- [ ] **Step 8: Qwen3 + compressed-tensors, tp=1**

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens.

- [ ] **Step 9: No commit — verification only**

Task 2 makes no code changes. If any of Steps 1-8 produce gibberish or errors:
- Gibberish on all rows → likely a `ModelWeightConfig` field mismatch; recheck step 10's `cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size`.
- Gibberish only on quantized rows → quantized weight path regression; the `_apply_linear` body itself didn't change in this refactor, so this would be unexpected — grep for `weights_[` and `text_model_ptr()` sites in `turbomind.cc` to confirm every dereference post-prepare got the `text_model_ptr()` rewrite.
- `TM_CHECK` abort "text_model not attached" on a specific row → that spec's `root.build()` didn't survive step 10-13; re-inspect the spec's `model()` method.

Resolve with an additional fix task, not by editing the Task 1 commit.

---

## Task 3: Manual strictness smoke tests (optional, one-off)

These confirm the new error paths fire cleanly. Not automated; not committed. Revert each change after.

- [ ] **Step 1: Missing `root.build()`**

Temporarily delete the `root.build()` line from `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py::model()`. Run any smoke test (e.g., Qwen3 dense tp=1).

Expected traceback with `TM_CHECK` abort containing:

```
ModelRoot::prepare: text_model not attached; did the spec forget root.build()?
```

Revert the deletion.

- [ ] **Step 2: Missing `ModelWeight` registrar**

Temporarily comment out the `static ModelWeightRegistrar _model_weight_reg;` line in `src/turbomind/models/model_weight.cc`. Rebuild (`cd build && ninja`) and run any smoke test.

Expected: `_tm.create_module(ModelWeightConfig)` returns nullptr; Python aborts at `root.build()` → `super().build()` → `_create_handles()` with a null handle in `_handles`, propagating to an `AttributeError` on the `.add_child_raw(...)` call in `TextModelBuilder.build()` (or a similar clear failure in the C++ layer).

Revert the comment. Rebuild.

- [ ] **Step 3: Narrowed `_built` guard — plain assignment OK**

In a Python REPL after Task 1 is committed:

```python
>>> import _turbomind as _tm
>>> from lmdeploy.turbomind.deploy.builder._base import Builder, BuiltModule
>>>
>>> cfg = _tm.NormConfig()
>>> cfg.dim = 128
>>> cfg.data_type = _tm.DataType.TYPE_BF16
>>> cfg.norm_eps = 1e-5
>>>
>>> # Stub contexts for the builder; we never actually build here.
>>> class _NullCtx:
...     def __enter__(self): return self
...     def __exit__(self, *a): pass
>>>
>>> b = Builder(cfg, [_NullCtx()])
>>> b._built = True                     # simulate post-build state
>>> b.x = 42                            # NEW: silently accepts
>>> b.x
42
>>> try:
...     b.y = BuiltModule([object()])   # still raises post-build
... except RuntimeError as e:
...     print("OK:", e)
OK: Builder is built; cannot assign 'y'
```

Expected: plain `b.x = 42` succeeds without RuntimeError; `BuiltModule` RHS still raises RuntimeError.

---

## Summary

| Task | Files                                                                                                       | Commit message |
|------|-------------------------------------------------------------------------------------------------------------|----------------|
| 1    | 2 new C++ files; 4 modified C++ files; 1 modified CMakeLists.txt; 6 modified Python files                   | `tm: create ModelWeight from Python via ModelRoot sentinel` |
| 2    | None                                                                                                        | (no commit — verification only) |
| 3    | None                                                                                                        | (no commit — one-off strictness checks) |

Total: 1 commit.
