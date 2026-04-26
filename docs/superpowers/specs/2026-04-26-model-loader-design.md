# ModelLoader Design: Replacing BaseOutputModel + TextModelLoader

## Problem

The current pipeline has three layers between `TurboMind` and the model specs, but two of them add no value:

- **`TextModelLoader`** (43 lines) — a trivial glue class that extracts GPU handles from `BaseOutputModel` and passes them to `spec.bind_runtime()`.
- **`BaseOutputModel`** — a lifecycle driver with a vacuous subclass (`TurbomindModel`) and an unused registry (`OUTPUT_MODELS`). Its only real jobs are creating checkpoint loaders and calling `spec.model()`.
- **Specs** — own layout and weight transformation but cannot export on their own.

The boundaries are unclear: `TextModelLoader` is decomposable inline, `TurbomindModel` is `class TurbomindModel(BaseOutputModel): pass`, and the `OUTPUT_MODELS` registry has only one entry.

## Design

Replace all three layers with a single `ModelLoader` class and the spec.

### ModelLoader

A single concrete class (no base class, no registry) that coordinates loading a spec's weights into the TurboMind runtime.

**File:** `deploy/model_loader.py`

```python
import torch

from .loader import create_loader


class ModelLoader:
    """Coordinates loading a spec's weights into the TurboMind runtime."""

    def __init__(self, spec, model_comm, gpu_count, model_path):
        self.spec = spec
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self._bind_runtime()

    def _bind_runtime(self):
        mc = self.model_comm
        attn_ranks  = [mc.attn_tp_rank(g)  for g in range(self.gpu_count)]
        mlp_ranks   = [mc.mlp_tp_rank(g)   for g in range(self.gpu_count)]
        model_tp    = [mc.model_tp_rank(g)  for g in range(self.gpu_count)]
        contexts = [mc.context(g) for g in range(self.gpu_count)]
        handles  = [mc.root(g)    for g in range(self.gpu_count)]
        self.spec.bind_runtime(
            contexts=contexts, root_handles=handles,
            attn_ranks=attn_ranks, mlp_ranks=mlp_ranks,
            model_tp_ranks=model_tp,
        )

    def export(self):
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()

    def export_iter(self):
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        yield -1
        torch.cuda.empty_cache()
```

Key points:
- Absorbs `TextModelLoader._bind_runtime()` inline (talks to `model_comm` directly, no wrapper methods).
- Absorbs `BaseOutputModel.export()` and `export_iter()`.
- Two-phase spec init preserved: spec is constructed without handles, then `bind_runtime()` is called by `ModelLoader.__init__()`.
- Top-level imports, no local imports.

### Changes to TurboMind

`turbomind.py`:

- `_from_hf()`: replaces `OUTPUT_MODELS.get('tm')(...)` with `ModelLoader(spec=spec, model_comm=model_comm, gpu_count=self.gpu_count, model_path=model_path)`.
- `_load_weights()`: calls `self._model_loader.export()` instead of `self._tm_model.export()`.
- `update_params()`: uses `self._model_loader` instead of `self._tm_model`.
- `self._tm_model` renamed to `self._model_loader`.

No other methods are affected — `sleep()`, `wakeup()`, `_process_weights()`, `_create_engine()` use `self.model_comm` directly.

### What gets deleted

| File | Reason |
|------|--------|
| `deploy/text_model_loader.py` | Absorbed by `ModelLoader._bind_runtime()` |
| `deploy/target_model/base.py` | `BaseOutputModel` + `OUTPUT_MODELS` registry gone |
| `deploy/target_model/fp.py` | `TurbomindModel` gone |
| `deploy/target_model/` directory | Empty after above deletions |

### What doesn't change

- **Specs** (`source_model/*.py`, `spec.py`) — `bind_runtime()`, `set_params()`, `model()`, `_get()`, `_linear()` unchanged.
- **Builders** (`builder/*.py`) — unchanged.
- **Loader** (`loader.py`) — `create_loader()` used the same way.
- **Converter** (`converter.py`) — `get_tm_config()` unchanged.

## Relationship summary

```
Before:  TurboMind → TurbomindModel(BaseOutputModel) → TextModelLoader → spec
After:   TurboMind → ModelLoader → spec
```
