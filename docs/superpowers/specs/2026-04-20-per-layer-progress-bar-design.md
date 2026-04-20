# Per-Layer Progress Bar for Turbomind Conversion

## Goal

Restore a meaningful progress bar during Turbomind weight conversion. After a
prior refactor that switched from streaming params to a single batched
`loader.all_items()` call, the bar in `BaseOutputModel.export()` degraded to a
stub that jumps from 0/1 to 1/1 at the very end. Users see no progress during
the slow part of conversion (per-layer weight commits).

## Current State

`lmdeploy/turbomind/deploy/target_model/base.py` `export()`:

```python
def export(self) -> None:
    from tqdm import tqdm
    import torch
    from ..loader import create_loader
    pbar = tqdm(total=1, desc='Convert to turbomind format', leave=False)
    loader = create_loader(self.model_path, self.spec._layer_pattern,
                           self.spec._loader_mappings)
    self.spec.set_params(loader.all_items())
    self.spec.model()
    torch.cuda.empty_cache()
    pbar.update(1)
    pbar.close()
```

`total=1` with a single `update(1)` after the entire `spec.model()` call is
effectively a no-op. Meanwhile, every `TextModelSpec` subclass already has a
`layers()` method whose `for i in range(self._num_layer)` loop dominates wall
time — that is the natural place to tick a bar.

Affected specs (4 total):

- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

## Design

### New helper in `source_model/utils.py`

A single free function produces the per-layer tqdm iterable. Every spec uses it;
formatting changes happen in one place.

```python
def layer_progress(num_layers: int):
    """tqdm iterable for spec.layers() per-layer conversion loops."""
    from tqdm import tqdm
    return tqdm(range(num_layers), desc='Loading', leave=False)
```

- `desc='Loading'` — short, matches common ML-tooling convention
  (HuggingFace uses "Loading checkpoint shards" etc.).
- `leave=False` — bar clears after completion, preserving prior UX.
- `total` is inferred from `range(num_layers)`.
- `tqdm` import is lazy (function-local) so `utils.py` stays import-cheap.

### Remove the dead pbar from `BaseOutputModel.export()`

After the change:

```python
def export(self) -> None:
    import torch
    from ..loader import create_loader
    loader = create_loader(self.model_path, self.spec._layer_pattern,
                           self.spec._loader_mappings)
    self.spec.set_params(loader.all_items())
    self.spec.model()
    torch.cuda.empty_cache()
```

Drops the `from tqdm import tqdm` import and the `pbar` creation / update /
close triad. The per-layer bar inside `spec.layers()` now carries all progress
reporting.

### Update each spec's `layers()` loop

In each of the 4 specs:

1. Add `layer_progress` to the existing `from .utils import ...` line.
2. Change the loop header from
   `for i in range(self._num_layer):` to
   `for i in layer_progress(self._num_layer):`.

No other changes to spec body. The loop body (building `DecoderLayerBuilder`,
attaching attention/ffn/moe/norms) is unchanged.

## Behavior

- `export()` path: user sees `Loading:   k/N` updating as each decoder layer is
  built, then the bar clears.
- `export_iter()` path: the same per-layer bar appears, since `export_iter()`
  also calls `self.spec.model()` → `spec.layers()`. This is acceptable — the
  streaming variant doesn't manage its own conflicting progress display.
- Non-layer work (token embeds, output norm, lm_head) runs outside the loop and
  is fast — no visible progress, which is fine because it is a small fraction
  of wall time.

## Files Changed

| File | Change |
|---|---|
| `lmdeploy/turbomind/deploy/source_model/utils.py` | Add `layer_progress(num_layers)` free function |
| `lmdeploy/turbomind/deploy/target_model/base.py` | Remove dead `pbar` and tqdm import from `export()` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Import `layer_progress`; use it in `layers()` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Same |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Same |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Same |

## Scope / Non-Goals

- **Layers-only progress.** `total = num_layers` only; we do not add embed /
  output_norm / lm_head to the total. Those phases flash by fast enough that
  the added complexity isn't worth it.
- **No suppression flag.** `export_iter()` gets the same bar; no
  `show_progress=False` plumbing on `spec.model()` or `spec.layers()`.
- **No new tests.** This is cosmetic terminal output; manual verification by
  running a conversion on any supported model is sufficient.
- **No changes to the four specs' business logic** — only the loop header and
  one import line per file.
