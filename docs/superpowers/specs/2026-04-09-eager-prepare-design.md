# Eager Prepare Design

Date: 2026-04-09

## Summary

Replace the lazy `_ensure_ranks` / `_layer_writer` / `_root_distributor` pattern with an explicit `prepare()` call. Store root and layers distributors as attributes. Eliminate three methods from `TextModelLoader`.

## Motivation

Three problems with the current code:

1. **Deceptive lazy init.** `_ensure_ranks` is called at the start of `_load_layer` and `_load_global`, disguising one-time initialization as a recurring guard. It computes ranks and returns silently on subsequent calls, making control flow hard to follow.
2. **Per-call distributor creation.** `_layer_writer` and `_root_distributor` create new `Distributor` objects every time they are called. The root and `layers` container never change — they should be created once.
3. **Unnecessary method count.** Three private methods (`_ensure_ranks`, `_layer_writer`, `_root_distributor`) exist solely to work around the lazy-init timing gap. An explicit `prepare()` collapses all three.

## Approach

### 1. `TextModelLoader.prepare()`

Called once from `turbomind.py` after `model_comm` and `gpu_count` are set. Replaces `_ensure_ranks`:

```python
def prepare(self):
    """Eagerly initialize distributors. Called after model_comm is set."""
    self._attn_ranks = [self.model.tp_ranks(gpu)[0]
                        for gpu in range(self.model.gpu_count)]
    self._mlp_ranks = [self.model.tp_ranks(gpu)[1]
                       for gpu in range(self.model.gpu_count)]
    self._root = self._root_distributor()
    self._layers = self._root.create_child('layers', ModuleListConfig())
```

### 2. Call site in `turbomind.py`

In `_from_hf`, after setting `model_comm` and `gpu_count`:

```python
tm_model.gpu_count = self.gpu_count
tm_model.model.prepare()
```

### 3. Methods deleted

| Method | Reason |
|--------|--------|
| `_ensure_ranks` | Replaced by `prepare()`, called once |
| `_layer_writer` | Replaced by `self._layers.create_child(str(layer), DecoderLayerConfig())` |
| `_root_distributor` | Absorbed into `prepare()` |

### 4. Updated callers

**`_load_layer`:** Replace `_ensure_ranks()` + `self._layer_writer(layer)` with `self._layers.create_child(str(layer), DecoderLayerConfig())`.

**`_load_global`:** Replace `_ensure_ranks()` + `self._root_distributor()` with `self._root`.

**`__init__`:** Initialize `self._root = None` and `self._layers = None` (set by `prepare()`).

### 5. Files changed

| File | Change |
|------|--------|
| `deploy/text_model_loader.py` | Add `prepare()`, delete `_ensure_ranks`/`_layer_writer`/`_root_distributor`, update `_load_layer`/`_load_global`/`__init__` |
| `turbomind.py` | Add `tm_model.model.prepare()` after `gpu_count` assignment |

No changes to: `distributor.py`, `load_context.py`, `configs.py`, `spec.py`, C++ code.

## Verification

Test models with TP=1 and TP=2, verifying both `_load_global` and `_load_layer` produce correct results.
