# Python-Created `ModelWeight` with C++ Sentinel Root

## Problem

After the deferred-module-creation refactor (2026-04-23), every `Builder`
subclass follows a uniform lifecycle: `__init__` → stage commits →
`build()` calls `_tm.create_module(cfg)` per context → drain → return
`BuiltModule`. Every `Module`-derived C++ class is registered in
`ModuleRegistry::instance()` and created through that path.

`ModelWeight` is the lone exception:

- C++ pre-creates it in `TurboMind::Impl::CreateWeights(index)` via a
  non-standard constructor `ModelWeight(const EngineParam&)`. This
  constructor does two unrelated things — copy `tp_size`/`tp_rank` from
  the engine param, and allocate a dedicated weight-loading CUDA stream +
  pool-backed allocator on `ModelWeight` itself (`stream_`, `alloca_`).
- Python reads the pre-created handles via `tm.root(index)` and wraps
  them in `TextModelBuilder`, whose `_create_handles` override is a
  no-op. Every other Builder runs `_tm.create_module` there.
- `ModelWeight` is the only `Module` subclass missing a
  `ModuleRegistry::instance().register_type("ModelWeight", ...)` entry.
  It can't be built through `_tm.create_module` even if a Python caller
  wanted to.

The asymmetry forces two unrelated special cases (one in C++, one in
Python) that mutually enable each other. Removing it eliminates
the `TextModelBuilder._create_handles` override, drops the `EngineParam`
dependency from `ModelWeight`, and makes `weights_[index]`'s ownership
shape the same as every other module slot (parent owns child via
`unique_ptr` after `add_child_raw`).

A related cleanup: `Builder.__setattr__`'s top-level `if self._built`
guard forces `Builder.__init__` to bootstrap `self._built = False` with
`object.__setattr__` (because `__setattr__` would otherwise read
`_built` before it exists). After landing this refactor the last place
still using that bootstrap is `TextModelBuilder`. With a one-line
restructure of the guard, `object.__setattr__` disappears from all
ordinary Builder code.

## Goals

1. `ModelWeight` becomes a registered `Module` created from Python via
   `_tm.create_module(ModelWeightConfig)`, same as every other module.
2. Introduce `ModelRoot` — a C++ sentinel that lives in
   `TurboMind::Impl::weights_`, owns the weight-loading stream/allocator,
   and has a single child slot `text_model : ModelWeight`. The
   Python-created `ModelWeight` is attached via the existing
   `add_child_raw` pathway.
3. `TextModelBuilder` becomes a regular `Builder` subclass — no
   `_create_handles` override, no pre-existing-handle special case.
4. Split `TurboMind::CreateWeights` into `CreateContext` (already
   private; promoted to public) + new `CreateRoot`. Python drives both.
5. Narrow `Builder.__setattr__`'s post-build guard to the `BuiltModule`
   branch only, eliminating every `object.__setattr__` call in
   `Builder.__init__` / `build()` / `TextModelBuilder`.

## Design

### `src/turbomind/models/model_root.{h,cc}` — new sentinel class

One composite module with a single typed child slot. Owns the
weight-loading stream/allocator that `ModelWeight` used to own.

```cpp
// model_root.h
namespace turbomind {

class ModelWeight;

/// Sentinel root for the weight tree.  Lives in TurboMind::Impl::weights_
/// and owns the CUDA stream + pool-backed allocator used during weight
/// loading.  Python creates a ModelWeight via _tm.create_module and
/// attaches it as the `text_model` child via add_child_raw.
class ModelRoot: public core::Module {
public:
    const char* type() const override { return "ModelRoot"; }

    ModelRoot();        // creates stream_ + alloca_
    ~ModelRoot() override;

    void prepare() override;     // asserts text_model is attached, then recurses

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    const core::Stream&    stream() const    { return stream_; }
    const core::Allocator& allocator() const { return alloca_; }

    // Convenience accessor; nullptr before Python attaches.
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

```cpp
// model_root.cc
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
    Module::prepare();     // recurses into text_model
}

TM_MODULE_METHODS(ModelRoot, MODEL_ROOT_CHILDREN, MODEL_ROOT_PARAMS)

}  // namespace turbomind
```

No `ModuleRegistry::instance().register_type("ModelRoot", ...)`. The
sentinel is only created via `TurboMind::CreateRoot`, which
is the one place where the device guard + stream ownership is set up
correctly. Exposing it through `_tm.create_module` would tempt callers
to create detached instances with no device affinity.

Why override `prepare()` instead of `verify()`: `Module::verify()` is
currently defined on every composite but never actually called from the
TurboMind pipeline — `ProcessWeights` only invokes `prepare()`.
`Module::prepare()`'s default body walks `for_each_child` and skips
null children via an `if (child)` guard, which would silently accept a
missing `text_model` and only blow up later at an accessor deref
(SIGSEGV in `CreateEngine` or `CreateRequest`). The override makes the
mistake fail loudly at `ProcessWeights` time with a pointer to the
likely cause.

### `src/turbomind/models/model_weight.{h,cc}` — lose special-case state

Drop the weight-loading stream/allocator; they move to `ModelRoot`.
Add `ModelWeightConfig` + registrar so Python can construct via
`_tm.create_module`.

```cpp
// model_weight.h additions
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
```

`ModelWeight` header changes:

- Constructor signature: `explicit ModelWeight(const core::ModelWeightConfig&)`
  (was `explicit ModelWeight(const EngineParam&)`).
- Remove `context()` / `stream()` / `allocator()` public accessors.
- Remove private members `stream_`, `alloca_`.
- Remove `#include "src/turbomind/models/llama/llama_params.h"` from the
  header — `EngineParam` is no longer referenced.
- Public field block comment "From EngineParam at construction"
  becomes "From ModelWeightConfig at construction". Fields
  (`tp_size`, `tp_rank`) unchanged.

`ModelWeight` implementation changes:

```cpp
ModelWeight::ModelWeight(const core::ModelWeightConfig& cfg)
    : tp_size(cfg.tp_size)
    , tp_rank(cfg.tp_rank)
{
    // Stream/allocator moved to ModelRoot; nothing to do here.
}
```

Add the registrar (anonymous-namespace pattern mirroring
`decoder_layer_weight.cc`):

```cpp
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
```

`ModelWeight::prepare()`, `verify()`, `layer()`, `layers_list()`, and
the public derived fields (`data_type`, `vocab_size`, `vocab_size_padded`,
`hidden_units`, etc.) are all unchanged.

### `src/turbomind/turbomind.{h,cc}` — split CreateWeights

`vector<shared_ptr<ModelWeight>> weights_` becomes
`vector<shared_ptr<ModelRoot>> weights_`. `CreateWeights` splits into
two public methods.

```cpp
// turbomind.h — public API changes
class TurboMind {
public:
    // Was: void CreateWeights(int index);
    // Now:
    void CreateContext(int index);
    core::Module* CreateRoot(int index);

    /// Returns the ModelRoot for GPU `index`. nullptr before CreateRoot.
    core::Module* root(int index);

    /// Returns the Stream and Allocator for GPU `index`'s weight tree.
    std::pair<core::Stream, core::Allocator> weight_context(int index);

    // ... rest unchanged ...
};
```

```cpp
// turbomind.cc — Impl methods
void CreateContext(int index)   // body: today's private CreateContext, verbatim
{
    // ... unchanged, including both h_global->Sync() barriers ...
}

core::Module* CreateRoot(int index)
{
    CudaDeviceGuard dev_guard(engine_param_.devices[index]);
    TM_CHECK(contexts_[index] != nullptr)
        << "CreateContext(" << index << ") must run before CreateRoot";
    weights_[index] = std::make_shared<ModelRoot>();
    return weights_[index].get();
}

// CreateWeights is deleted.
```

Lifetime note: `weights_[index]` owns the `ModelRoot` via
`shared_ptr`. The `ModelRoot`'s single child `text_model`, installed
by Python via `add_child_raw`, is owned by the root as `unique_ptr<ModelWeight>`
(via `TM_CHILD_MEMBER`). `~Impl` tears down
`engines_` → `contexts_` → `weights_` in the existing order;
`~ModelRoot` cascades to destroy the attached `ModelWeight` before
the stream/allocator are released.

Internal access-site rewrites — every `weights_[i]->field` or
`*weights_[i]` in `turbomind.cc` now goes through `text_model_ptr()`:

| Before                                              | After                                                           |
| --------------------------------------------------- | --------------------------------------------------------------- |
| `weights_[0]->vocab_size`                           | `weights_[0]->text_model_ptr()->vocab_size`                     |
| `weights_[0]->hidden_units`                         | `weights_[0]->text_model_ptr()->hidden_units`                   |
| `weights_[index]->vocab_size` (`WarmUp`)            | `weights_[index]->text_model_ptr()->vocab_size`                 |
| `*weights_[index]` (`LanguageModel` / `Engine` ctors) | `*weights_[index]->text_model_ptr()`                          |
| `weights_[index]->context()` (`ProcessWeights`)     | unchanged — `ModelRoot::context()` has the same signature       |
| `weights_[index]->prepare()` (`ProcessWeights`)     | unchanged — recurses via `Module::prepare()` into `text_model`  |
| `mw->stream()` / `mw->allocator()` (`weight_context`) | `root->stream()` / `root->allocator()`                        |

The free function `TurboMind::root(int index)` and
`TurboMind::weight_context(int index)` keep their public signatures; only
the implementation reads from the `ModelRoot` instead.

### `src/turbomind/python/bind.cpp` — config + API changes

Two edits in `PYBIND11_MODULE`:

1. Register the new config next to the others:

   ```cpp
   bind_config<turbomind::core::ModelWeightConfig>(m, "ModelWeightConfig");
   ```

2. Replace the `create_weights` method binding with `create_context`
   and `create_root`:

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

   `create_root` returns a non-owning `Module*` (the shared_ptr in
   `Impl::weights_` keeps the sentinel alive for the lifetime of
   `TurboMind`). Same `reference` policy as the existing `.def("root", ...)`.

`ModelRoot` itself needs no pybind class — it is accessed as
`ft::core::Module*`, and `add_child_raw` / `child` / `prepare`
are already exposed on the `core::Module` base binding.

### `lmdeploy/turbomind/deploy/builder/_base.py` — narrow the guard, drop the override

Two edits:

1. **`Builder.__init__` / `__setattr__` / `build()`** — move the
   `_built` guard inside the `BuiltModule` branch. `object.__setattr__`
   goes away entirely.

   ```python
   class Builder:
       def __init__(self, config, contexts, tp=1, ranks=None):
           self._built = False        # bool is not BuiltModule; falls through
           self._contexts = contexts
           self._tp = tp
           self._ranks = ranks
           self.config = config
           self._pending_linears = {}
           self._pending_tensors = {}
           self._pending_children = {}
           self._handles = None

       def __setattr__(self, name, value):
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

       def build(self) -> BuiltModule:
           if self._built:
               return BuiltModule(self._handles)
           self._create_handles()
           self._built = True         # plain assignment works
           for name, (lin, side, mdt) in self._pending_linears.items():
               self._apply_linear(name, lin, side, mdt)
           for name, (t, side, mdt) in self._pending_tensors.items():
               self._apply_tensor(name, t, side, mdt)
           for name, child_handles in self._pending_children.items():
               self._attach_handles(name, child_handles)
           return BuiltModule(self._handles)
   ```

   `__setitem__`, `_commit_linear`, `_commit_tensor`,
   `_create_handles`, `_cfg_for_rank`, `_attach_handles`,
   `_apply_linear`, `_apply_tensor` — all unchanged. `__setitem__`
   already only handles `BuiltModule` values, so its top-level
   `if self._built` guard correctly stays at the top.

2. **`TextModelBuilder`** — becomes a regular `Builder`. The
   `_create_handles` override is deleted; `build()` calls
   `super().build()` and then attaches the built `ModelWeight` handles
   to the sentinel roots.

   ```python
   class TextModelBuilder(Builder):
       """Builder for the root ModelWeight.

       Constructs a ModelWeight via `_tm.create_module(ModelWeightConfig)`
       (standard Builder path), then attaches it to externally-owned
       ModelRoot sentinel handles as their `text_model` child.
       """

       def __init__(self, config, contexts, *, root_handles,
                    tp, ranks, vocab_size, data_type):
           super().__init__(config=config, contexts=contexts,
                            tp=tp, ranks=ranks)
           self._root_handles = root_handles
           self._vocab_size = vocab_size
           self._data_type = data_type

       def build(self) -> BuiltModule:
           built = super().build()
           for i, (root, text_model) in enumerate(
                   zip(self._root_handles, built.handles)):
               with self._contexts[i]:
                   root.add_child_raw('text_model', text_model)
           return built

       # add_token_embeds, add_lm_head — bodies unchanged
       ...
   ```

   `super().__init__` sets `self._built = False` first (before any
   TextModelBuilder-specific field), so the post-build
   guard-that-reads-`_built` never encounters an unset attribute.

### Spec updates

**`lmdeploy/turbomind/deploy/spec.py`** — `TextModelSpec.bind_runtime`
parameter list is unchanged; `self._root_handles` now refers to
`ModelRoot*` handles (the sentinel), but nothing in the Python code
depends on the C++ type beyond "it accepts `add_child_raw`" and
"its `child('text_model')` is the real `ModelWeight` after attach".
No rename required.

**Each spec's `model()` method** (`qwen3_spec.py`, `qwen3_5_spec.py`,
`gpt_oss_spec.py`, `glm4_moe_lite_spec.py`) gains two lines at the top
and stays byte-for-byte otherwise:

```python
def model(self):
    ec = self.engine_cfg
    cfg = _tm.ModelWeightConfig()
    cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
    # cfg.tp_rank filled per-rank by Builder._cfg_for_rank

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

The existing `root.build()` as the last line (added during the
deferred-creation refactor) is now the actual trigger for
`_tm.create_module(ModelWeightConfig)` + the post-build
sentinel attachment loop. Forgetting it leaves the sentinel without a
`text_model` child, and `ProcessWeights(i)` → `ModelRoot::prepare()`
fails immediately with the `TM_CHECK` message pointing at `root.build()`.

The `_tm.ModelWeightConfig()` import chains through the existing
`_turbomind` import each spec already uses. No new imports needed.

### `lmdeploy/turbomind/turbomind.py` — split `_create_weight`

One function body change. The per-device lambda now calls both
C++ methods in sequence; each device still runs in its own
ThreadPoolExecutor task.

```python
def _create_weight(self, model_comm):
    """Allocate weight buffer, load params if from_workspace."""

    def _create_weight_func(device_id):
        model_comm.create_context(device_id)
        model_comm.create_root(device_id)

    with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
        futures = [executor.submit(_create_weight_func, device_id)
                   for device_id in range(self.gpu_count)]
        for future in futures:
            future.result()
```

Barrier semantics are preserved: `CreateContext`'s two
`h_global->Sync()` calls are collective across all ranks. All ranks
enter `create_context` concurrently (one per executor task), sync at
the barriers, then each rank independently runs `create_root` (which
has no collectives). No new inter-rank synchronisation is required.

### `lmdeploy/turbomind/deploy/text_model_loader.py`

Unchanged. `model.root(gpu)` still returns a `Module*` — today
`ModelWeight*`, tomorrow `ModelRoot*`. The Python consumer doesn't
care which type it is.

## Error semantics — what changes

| Mistake                                              | Before                                                           | After                                                                |
| ---------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| `d.x = unbuilt_builder`                              | `TypeError: ...: assign .build() output (BuiltModule) ...`       | unchanged                                                            |
| `parent[i] = unbuilt_builder`                        | `TypeError: ...[i]: call .build() first`                         | unchanged                                                            |
| `built_builder.x = some_built_module`                | `RuntimeError: ... is built; cannot assign 'x'`                  | unchanged (guard moved inside BuiltModule branch)                    |
| `built_builder[i] = some_built_module`               | `RuntimeError: ... is built; cannot set index i`                 | unchanged (`__setitem__`'s top-level guard stays)                    |
| `built_builder.add_qkv_proj(...)`                    | `AssertionError: ... is built; commit '...' rejected`            | unchanged (`_commit_*` asserts stay)                                 |
| `built_builder.x = 42`                               | `RuntimeError: ... is built; cannot assign 'x'`                  | **silently assigns** — no staging, no C++ mutation (new)             |
| Spec omits `root.build()`                            | silent — ModelWeight never built; later commits crash or hang    | `TM_CHECK` abort in `ModelRoot::prepare()`: "text_model not attached" |

The "plain post-build assignment" row is the one deliberate relaxation.
Plain attribute writes post-build don't stage anything and don't mutate
C++ state; the guard was over-defensive. The real footgun — silent
no-op child staging — stays a loud error.

## Verification

Same matrix as the deferred-creation refactor (per `AGENTS.md`, no
unit tests for this subsystem; end-to-end via
`scripts/test_turbomind_model.py` with ≥128 tokens of meaningful
response per run).

| Model                              | Exercises                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| Qwen3 dense (tp=1)                 | `TextModelBuilder` as a regular `Builder`; `ModelWeightConfig` flow               |
| Qwen3 dense (tp=2)                 | Per-rank `tp_rank` via `Builder._cfg_for_rank`; `ModelRoot` stream sharing        |
| Qwen3-MoE (tp=2)                   | Container attachments still work through the sentinel                             |
| GLM-4 MoE Lite (tp=2)              | MLA path + ModelRoot — confirms the sentinel doesn't interfere with MLA recovery  |
| GPT-OSS (tp=2)                     | mxfp4 packed experts + sliding-window attn under the new lifecycle                |
| Qwen3 + AWQ / FP8 / GPTQ / compressed-tensors (tp=1) | Quantized path regressions — `_apply_linear` still routes through ModelWeight |

### Manual strictness smoke tests (one-off)

- Locally change a spec to drop `root.build()`; run any model; expect a
  `TM_CHECK` abort from `ModelRoot::prepare()` with message
  "text_model not attached; did the spec forget root.build()?".
- Locally delete `ModelWeight`'s `ModuleRegistrar`; run any model;
  expect `_tm.create_module(ModelWeightConfig)` to return nullptr and
  Python to abort at `root.build()` with the module-create failure.

## Migration

Single atomic PR — the C++ type change (`weights_[index]` from
`shared_ptr<ModelWeight>` to `shared_ptr<ModelRoot>`), the pybind API
change (`create_weights` → `create_context` + `create_root`), and the
Python spec change (new `ModelWeightConfig` + `TextModelBuilder` no
longer overriding `_create_handles`) all depend on each other; there's
no half-state that compiles and runs.

Rollout order within the PR:

1. **`src/turbomind/models/model_root.{h,cc}`** — new files defining
   `ModelRoot` and its `TM_MODULE_METHODS`. No other file includes
   `model_root.h` yet.
2. **`src/turbomind/models/model_weight.h`** — add
   `ModelWeightConfig`; change ctor signature to
   `ModelWeight(const core::ModelWeightConfig&)`; remove
   `context()`/`stream()`/`allocator()` public accessors; remove
   `stream_`/`alloca_` private fields; remove
   `<llama_params.h>` include.
3. **`src/turbomind/models/model_weight.cc`** — rewrite constructor body
   (drop stream/alloc setup, init `tp_size`/`tp_rank` from config);
   add anonymous-namespace registrar for `"ModelWeight"`.
4. **`src/turbomind/turbomind.h`** — public API: replace
   `CreateWeights` with `CreateContext` + `CreateRoot`.
5. **`src/turbomind/turbomind.cc`** — change `weights_` element type to
   `shared_ptr<ModelRoot>`; delete `CreateWeights` method on `Impl`;
   move body of the private `CreateContext` to become a public
   `Impl::CreateContext` (no logic change); add `Impl::CreateRoot`;
   rewrite every `weights_[i]->field` / `*weights_[i]` read site to go
   through `text_model_ptr()`; update `weight_context(index)` to read
   stream/alloca from the `ModelRoot`.
6. **`src/turbomind/python/bind.cpp`** — add
   `bind_config<ModelWeightConfig>`; replace the `.def("create_weights", ...)`
   binding with `.def("create_context", ...)` +
   `.def("create_root", ...)`.
7. **`lmdeploy/turbomind/deploy/builder/_base.py`** — five localized
   edits:
   - `Builder.__setattr__`: move the `if self._built` guard inside
     the `BuiltModule` branch (the only branch that stages a child).
   - `Builder.__init__`: replace
     `object.__setattr__(self, '_built', False)` with plain
     `self._built = False`.
   - `Builder.build()`: replace
     `object.__setattr__(self, '_built', True)` with plain
     `self._built = True`.
   - `TextModelBuilder.__init__`: change signature from
     `(handles, contexts, *, tp, ranks, vocab_size, data_type)` to
     `(config, contexts, *, root_handles, tp, ranks, vocab_size, data_type)`;
     swap `self._handles = handles` for `self._root_handles = root_handles`;
     drop `config=None` from the `super().__init__` call (pass `config`).
   - `TextModelBuilder`: delete the `_create_handles` override; add a
     `build()` override that calls `super().build()` and loops
     `root.add_child_raw('text_model', text_model)` under each context.
8. **`lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` +
   `qwen3_5_spec.py` + `gpt_oss_spec.py` + `glm4_moe_lite_spec.py`** —
   each spec's `model()` gains a two-line prologue:

   ```python
   cfg = _tm.ModelWeightConfig()
   cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
   ```

   and the `TextModelBuilder(...)` call moves from
   `(self._root_handles, self._contexts, tp=..., ...)` to
   `(cfg, self._contexts, root_handles=self._root_handles, tp=..., ...)`
   — `cfg` becomes the first positional arg, `_root_handles` becomes a
   keyword arg. Every other arg keeps its existing keyword form. The
   rest of the method body (all the `root.add_*`, `root.norm = ...`,
   `root.layers = ...`, `root.build()` lines) is byte-for-byte unchanged.
9. **`lmdeploy/turbomind/turbomind.py`** — rewrite `_create_weight`
   per-device lambda to call `create_context(device_id)` then
   `create_root(device_id)`.

The tree is broken between steps 2 and 6 (C++ types/signatures change
piecewise), and the C++/Python API mismatch persists until step 9
(Python still calling `create_weights` before step 9 would fail). All
nine steps must land together before the first build + smoke test.
Build (`ninja` from `build/`) after step 9 and run the verification
matrix.
