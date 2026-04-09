# Distributor Extract Design

Date: 2026-04-09

## Summary

Extract `LayerWriter` from `text_model_loader.py` into its own `distributor.py` module, rename it to `Distributor`, and unify `_load_global` to use the same GPU-transparent API as `_load_layer`.

## Motivation

Three problems with the current code:

1. **Misleading name.** `LayerWriter` wraps N GPU handles with bound TP config. It works at any scope (root, layer, child), but the name implies it is layer-specific.
2. **Inconsistent abstraction.** `_load_layer` uses `LayerWriter` for GPU transparency. `_load_global` manually iterates GPUs and calls raw `create_child` / `commit_tensor` on individual handles — the exact pattern `LayerWriter` was created to eliminate.
3. **Coupling.** The GPU distribution class lives inside the loader file, making it harder to reason about responsibilities.

## Approach: Minimal Extract + Unify

### 1. New file: `distributor.py`

`Distributor` class with identical API to current `LayerWriter`:

```python
class Distributor:
    """Wraps N GPU handles for a single logical module.

    Distributes create_child / commit_linear / commit_tensor
    across all GPUs with bound TP configuration.
    """

    def __init__(self, handles, tp=1, ranks=None)
    def create_child(self, name, config, tp=None, ranks=None) -> Distributor
    def commit_linear(self, name, linear, split_side=None, model_dtype=None)
    def commit_tensor(self, name, tensor, split_side=None)
```

Internal helpers (`tp_size` property, `_rank_for(gpu_idx)`) are unchanged.

### 2. Changes to `text_model_loader.py`

**Import:** `from .distributor import Distributor`

**New factory method** (parallels `_layer_writer`):

```python
def _root_distributor(self) -> Distributor:
    handles = [self.model.root(gpu)
               for gpu in range(self.model.gpu_count)
               if self.model.root(gpu) is not None]
    return Distributor(handles)
```

**Rewrite `_load_global`** to use Distributor instead of manual GPU iteration:

- `_root_distributor()` replaces the per-GPU `root = self.model.root(gpu)` loop
- `Distributor.create_child(tp=tp, ranks=self._attn_ranks)` replaces raw `root.create_child(cfg.to_cpp())` + per-GPU rank management
- `child.commit_tensor(...)` replaces raw `commit_tensor(handle, ...)`
- Typed `NormConfig` replaces raw `_tm.NormConfig()` for the final norm — the conversion is identical (`NormConfig.to_cpp()` produces the same C++ config)

**No changes to `_process_*` methods** beyond `LayerWriter` -> `Distributor` rename.

### 3. Files changed

| File | Change |
|------|--------|
| `deploy/distributor.py` | New file: `Distributor` class |
| `deploy/text_model_loader.py` | Import `Distributor`, add `_root_distributor()`, rewrite `_load_global`, rename refs |

No changes to: `load_context.py`, `configs.py`, `spec.py`, C++ code, or other callers.

## Verification

Test models with TP=1 and TP=2, verifying both `_load_global` (tok_embeddings, norm, output) and `_load_layer` (all `_process_*` methods) produce correct results. Request at least 128 tokens per test and verify the response is meaningful.
