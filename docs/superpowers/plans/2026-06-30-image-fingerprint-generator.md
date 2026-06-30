# Image Fingerprint Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a real per-image SHA-256 fingerprint in the Qwen3.5 TurboMind converter so the already-landed fingerprint *consumer* (`Qwen3_5VitItem.fingerprint` -> `Sequence::multimodal_spans` -> scheduler fold/compare) actually reuses image-span KV and skips the ViT across requests that share an image -- i.e. un-dormant image-span caching.

**Architecture:** Phase A (this PR), post-preprocess. A Qwen3.5-specific helper `_image_fingerprint` computes `SHA256(modality, grid_thw, second_per_grid, pixel_values.bytes)` from fields already on the `input_mm` dict at converter time. A second helper `_resolve_fingerprint` adds the `is-not-None` pre-placed-override hook (absent -> compute; present -> use as-is, so an explicit `b''` stays empty/dormant) so the future pre-preprocess generator (phase B, skip-preprocessing) needs no converter change. The converter calls `_resolve_fingerprint(input_mm)` instead of the current `input_mm.get('fingerprint', b'')` stand-in. No C++ engine / scheduler / prefix-trie / binding / contract change -- only the generator is missing.

**Tech Stack:** Python (lmdeploy TurboMind model module + VLM harness + pytest), pybind11 `_turbomind` (importable on CPU for the unit test), torch, hashlib/struct, CUDA (end-to-end verification only).

Spec: `docs/superpowers/specs/2026-06-30-image-fingerprint-generator-design.md`.

## How this plan handles testing (read first)

- **Pure-function unit gate (Task 1, CPU, sandbox-runnable):** `_image_fingerprint` / `_resolve_fingerprint` are exercised by a new pytest module. `_turbomind` imports on CPU (verified: `import _turbomind` succeeds without a GPU), so the test can `from lmdeploy.turbomind.models.qwen3_5 import _image_fingerprint, _resolve_fingerprint` and run in the sandbox.
- **Behavioral gate (Task 4, GPU, MUST run outside the sandbox):** the existing VLM harness `scripts/vlm_prefix_cache_check.py` (updated in Task 3) drives the real converter path and self-validates with a `images_encoded` oracle (sum of `images_batched` across the run) plus greedy text-equality. The text regression `scripts/test_turbomind_model.py` (AS IS) guards the shared path.
- **GPU rule:** Task 4 touches CUDA and MUST run outside the sandbox (`required_permissions: ["all"]`). The sandbox has no NVIDIA driver. Always check an empty GPU with `nvidia-smi` first.
- **No worktree:** per workspace rules, execute in the current workspace -- do NOT create a git worktree.
- **Verify the response every time** in Task 4: >=128 tokens, coherent, on-topic; gibberish = bug.

## File map (what each touched file is responsible for)

- Modify `lmdeploy/turbomind/models/qwen3_5.py` -- add `import hashlib` / `import struct`; add module-level `_image_fingerprint(input_mm)` and `_resolve_fingerprint(input_mm)`; change `to_turbomind_multimodal` to call `_resolve_fingerprint`.
- Create `tests/test_lmdeploy/test_vl/test_image_fingerprint.py` -- CPU-only pytest for the two helpers (determinism, field-sensitivity, bfloat16, `is-not-None` routing, 32-byte / never-all-zero output).
- Modify `scripts/vlm_prefix_cache_check.py` -- remove the stand-in `install_fingerprint_patch`; reuse/distinct now exercise the real generator; dormant is driven by a test-only `install_dormant_patch` that pre-places `fingerprint = b''`.

---

## Task 1: `_image_fingerprint` + `_resolve_fingerprint` helpers (TDD, CPU)

**Files:**
- Create: `tests/test_lmdeploy/test_vl/test_image_fingerprint.py`
- Modify: `lmdeploy/turbomind/models/qwen3_5.py` (add imports + two module-level helpers)

- [ ] **Step 1: Write the failing test**

Create `tests/test_lmdeploy/test_vl/test_image_fingerprint.py`:

```python
# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.turbomind.models.qwen3_5 import _image_fingerprint, _resolve_fingerprint
from lmdeploy.vl.constants import Modality


def _img_mm(pixels=None, grid=(1, 28, 28)):
    if pixels is None:
        pixels = torch.zeros(28 * 28, 3 * 16 * 16, dtype=torch.float32)
    return {
        'modality': Modality.IMAGE,
        'pixel_values': pixels,
        'image_grid_thw': list(grid),
        'offset': (0, 1),
    }


def _vid_mm(pixels=None, grid=(2, 28, 28), spg=1.0):
    if pixels is None:
        pixels = torch.zeros(2 * 28 * 28, 3 * 16 * 16, dtype=torch.float32)
    return {
        'modality': Modality.VIDEO,
        'pixel_values_videos': pixels,
        'video_grid_thw': list(grid),
        'offset': (0, 1),
        'second_per_grid': spg,
    }


def test_output_is_32_bytes_and_never_all_zero():
    fp = _image_fingerprint(_img_mm())
    assert isinstance(fp, bytes) and len(fp) == 32
    assert fp != b'\x00' * 32


def test_identical_inputs_match():
    assert _image_fingerprint(_img_mm()) == _image_fingerprint(_img_mm())


def test_different_pixels_differ():
    a = _image_fingerprint(_img_mm(pixels=torch.ones(28 * 28, 3 * 16 * 16)))
    b = _image_fingerprint(_img_mm(pixels=torch.full((28 * 28, 3 * 16 * 16), 2.0)))
    assert a != b


def test_different_grid_thw_differ():
    # same prod (784) -> same pixel shape/bytes, different (t,h,w) split
    a = _image_fingerprint(_img_mm(grid=(1, 28, 28)))
    b = _image_fingerprint(_img_mm(grid=(1, 14, 56)))
    assert a != b


def test_grid_thw_as_tensor_or_list_equivalent():
    a = _image_fingerprint(_img_mm(grid=[1, 28, 28]))
    b = _image_fingerprint(_img_mm(grid=torch.tensor([1, 28, 28])))
    assert a == b


def test_different_modality_differ():
    # identical pixel bytes + identical grid + no spg; only the modality byte differs
    pv = torch.zeros(28 * 28, 3 * 16 * 16, dtype=torch.float32)
    img = {'modality': Modality.IMAGE, 'pixel_values': pv,
           'image_grid_thw': [1, 28, 28], 'offset': (0, 1)}
    vid = {'modality': Modality.VIDEO, 'pixel_values_videos': pv,
           'video_grid_thw': [1, 28, 28], 'offset': (0, 1)}
    assert _image_fingerprint(img) != _image_fingerprint(vid)


def test_second_per_grid_present_vs_none_differ():
    a = _image_fingerprint(_vid_mm(spg=None))
    b = _image_fingerprint(_vid_mm(spg=1.0))
    assert a != b


def test_different_second_per_grid_differ():
    a = _image_fingerprint(_vid_mm(spg=1.0))
    b = _image_fingerprint(_vid_mm(spg=2.0))
    assert a != b


def test_bfloat16_pixels_supported_and_stable():
    pv = torch.zeros(28 * 28, 3 * 16 * 16, dtype=torch.bfloat16)
    fp = _image_fingerprint(_img_mm(pixels=pv))
    assert len(fp) == 32 and fp != b'\x00' * 32
    assert fp == _image_fingerprint(_img_mm(pixels=pv))


def test_resolve_absent_computes_real_digest():
    mm = _img_mm()
    mm.pop('fingerprint', None)
    assert _resolve_fingerprint(mm) == _image_fingerprint(mm)
    assert len(_resolve_fingerprint(mm)) == 32


def test_resolve_present_digest_is_used_unchanged():
    digest = b'\x01' * 32
    mm = _img_mm()
    mm['fingerprint'] = digest
    assert _resolve_fingerprint(mm) == digest


def test_resolve_present_empty_stays_empty():
    # the is-not-None hook: an explicit b'' must NOT fall through to compute
    mm = _img_mm()
    mm['fingerprint'] = b''
    assert _resolve_fingerprint(mm) == b''
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_lmdeploy/test_vl/test_image_fingerprint.py -v`
Expected: collection error -- `ImportError: cannot import name '_image_fingerprint' from 'lmdeploy.turbomind.models.qwen3_5'` (the helpers do not exist yet).

- [ ] **Step 3: Add the imports**

In `lmdeploy/turbomind/models/qwen3_5.py`, the stdlib block at the top is currently:

```python
import math
import re
from typing import TYPE_CHECKING, Any
```

Insert `import hashlib` BEFORE `import math` and `import struct` AFTER `import re` (i.e. between `import re` and `from typing import ...`), so the block becomes (ruff/isort: straight imports sorted first, then from-imports within the section):

```python
import hashlib
import math
import re
import struct
from typing import TYPE_CHECKING, Any
```

(`torch` and `from lmdeploy.vl.constants import Modality` are imported lower down; leave them untouched.)

- [ ] **Step 4: Add the two helpers**

Add these as module-level functions in `lmdeploy/turbomind/models/qwen3_5.py`, immediately before `class Qwen3_5VisionModel:` at line 331 (i.e. after the `_split_packed_vision_qkv` helper at lines 325-328 -- co-located with the other vision-converter helpers `_assert_trivial` / `_pad_head_dim_in` / `_split_packed_vision_qkv` that sit right above the class whose `to_turbomind_multimodal` calls them):

```python
def _image_fingerprint(input_mm: dict) -> bytes:
    """SHA-256 over the Qwen3.5 ViT-forward inputs plus the mRoPE scalar.

    Post-preprocess (phase A): every input is already on the item dict. Two
    requests hash equal iff their ViT embeddings and cached LM KV for the image
    span are identical -- i.e. reuse is correct.
    """
    modality = input_mm['modality']
    is_video = modality in (Modality.VIDEO, Modality.VIDEO.value)
    pv   = input_mm['pixel_values_videos'] if is_video else input_mm['pixel_values']
    gthw = input_mm['video_grid_thw']      if is_video else input_mm['image_grid_thw']
    if isinstance(gthw, torch.Tensor):
        values = gthw.flatten().tolist()
    else:
        values = list(gthw)
    t, h, w = int(values[0]), int(values[1]), int(values[2])
    spg = input_mm.get('second_per_grid')          # video only; float | None

    h_obj = hashlib.sha256()
    h_obj.update(struct.pack('<B', 1 if is_video else 0))
    h_obj.update(struct.pack('<3i', t, h, w))
    h_obj.update(struct.pack('<B', 0 if spg is None else 1))
    if spg is not None:
        h_obj.update(struct.pack('<d', float(spg)))
    # Reinterpret the raw storage as uint8 so the digest is dtype-agnostic and
    # works for bfloat16 (numpy cannot consume bfloat16 directly). Same dtype +
    # same values -> same bytes; the dtype is constant per engine instance.
    h_obj.update(pv.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h_obj.digest()                          # 32 bytes; never all-zero


def _resolve_fingerprint(input_mm: dict) -> bytes:
    """Use a pre-placed fingerprint if present (future pre-preprocess generator,
    or a test forcing empty/dormant); otherwise derive it from the ViT inputs.

    `is not None` (not `or`) so an explicit b'' stays empty rather than falling
    through to compute -- the empty-fingerprint sentinel must be preserved
    (empty never compares equal -> image-span reuse stays dormant).
    """
    fp = input_mm.get('fingerprint')
    return fp if fp is not None else _image_fingerprint(input_mm)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_lmdeploy/test_vl/test_image_fingerprint.py -v`
Expected: `12 passed` (all tests green; pure CPU, no GPU).

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/models/qwen3_5.py tests/test_lmdeploy/test_vl/test_image_fingerprint.py
git commit -m "feat(qwen3_5): add image fingerprint generator helpers"
```

---

## Task 2: Wire the helpers into the converter

**Files:**
- Modify: `lmdeploy/turbomind/models/qwen3_5.py` (`Qwen3_5VisionModel.to_turbomind_multimodal`, around lines 403-412)

- [ ] **Step 1: Replace the stand-in read with the resolver**

In `Qwen3_5VisionModel.to_turbomind_multimodal` (`lmdeploy/turbomind/models/qwen3_5.py`), the `Qwen3_5VitItem` constructor call currently passes `fingerprint=input_mm.get('fingerprint', b'')`. Resolve the fingerprint just before building the item. The current block (lines 403-412):

```python
            token_begin, token_end = self._offset_pair(input_mm['offset'])
            items.append(
                _tm.multimodal.Qwen3_5VitItem(
                    modality=tm_modality,
                    data=data,
                    token_begin=token_begin,
                    token_end=token_end,
                    grid_thw=grid_thw,
                    fingerprint=input_mm.get('fingerprint', b''),
                ))
```

becomes:

```python
            token_begin, token_end = self._offset_pair(input_mm['offset'])
            fingerprint = _resolve_fingerprint(input_mm)
            items.append(
                _tm.multimodal.Qwen3_5VitItem(
                    modality=tm_modality,
                    data=data,
                    token_begin=token_begin,
                    token_end=token_end,
                    grid_thw=grid_thw,
                    fingerprint=fingerprint,
                ))
```

- [ ] **Step 2: Import / syntax check**

Run: `python -c "from lmdeploy.turbomind.models.qwen3_5 import Qwen3_5VisionModel, _image_fingerprint, _resolve_fingerprint; print('ok')"`
Expected: prints `ok` (no import error; `_turbomind` loads on CPU).

- [ ] **Step 3: Re-run the helper unit tests (regression)**

Run: `pytest tests/test_lmdeploy/test_vl/test_image_fingerprint.py -v`
Expected: `12 passed`.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/models/qwen3_5.py
git commit -m "feat(qwen3_5): compute image fingerprint in to_turbomind_multimodal"
```

---

## Task 3: Update the VLM prefix-cache harness

**Files:**
- Modify: `scripts/vlm_prefix_cache_check.py`

The real generator now computes fingerprints, so the stand-in `install_fingerprint_patch` (which injected `sha256(pixels)`) is obsolete. `reuse`/`distinct` exercise the real generator with no injection. `dormant` is driven by a test-only `install_dormant_patch` that pre-places `fingerprint = b''` (the `is-not-None` hook keeps it empty).

- [ ] **Step 1: Remove the now-unused `hashlib` import**

In `scripts/vlm_prefix_cache_check.py`, delete the `import hashlib` line (line 27). It was used only by the stand-in patch being removed.

- [ ] **Step 2: Replace `install_fingerprint_patch` with `install_dormant_patch`**

Delete the entire `install_fingerprint_patch` function (lines 56-75) and replace it with:

```python
def install_dormant_patch() -> None:
    """Force empty fingerprints so image-span reuse stays dormant.

    Pre-places `fingerprint = b''` on each item dict before the converter reads
    it; the converter's `is-not-None` hook treats an explicit b'' as 'use as-is',
    so the digest stays empty (empty never compares equal -> no image-span match).
    Test scaffolding only -- the real generator (phase A) always populates a
    real digest, so this just re-creates the pre-PR dormant state for the
    negative-control scenario.
    """
    from lmdeploy.turbomind.models.qwen3_5 import Qwen3_5VisionModel
    if getattr(Qwen3_5VisionModel, '_fp_dormant_patched', False):
        return
    _orig = Qwen3_5VisionModel.to_turbomind_multimodal

    def _patched(self, multimodal):
        for mm in multimodal:
            mm['fingerprint'] = b''
        return _orig(self, multimodal)

    Qwen3_5VisionModel.to_turbomind_multimodal = _patched
    Qwen3_5VisionModel._fp_dormant_patched = True
```

- [ ] **Step 3: Switch the run() injection logic to dormant-only**

In `run(args)` (around lines 124-126), replace:

```python
    inject = args.scenario in ('reuse', 'distinct')
    if inject:
        install_fingerprint_patch()
```

with:

```python
    force_empty = args.scenario == 'dormant'
    if force_empty:
        install_dormant_patch()
```

- [ ] **Step 4: Update the report line**

In `run(args)`, replace:

```python
    print(f'=== scenario: {args.scenario} (inject_fingerprint={inject}) ===')
```

with:

```python
    print(f'=== scenario: {args.scenario} (force_empty_fingerprint={force_empty}) ===')
```

- [ ] **Step 5: Update the module docstring**

Replace the scenario block in the top docstring (lines 6-15) with:

```
  reuse    same image + same prompt twice, real generator computes the digest.
           Expect: warm request reuses the image span -> only the cold image is
           encoded (images_encoded == 1); warm text == cold text.
  distinct two DIFFERENT images of equal token length, real generator computes
           the digest. Expect: NO false hit -> both images are encoded
           (images_encoded == 2); both outputs non-empty.
  dormant  same image twice, fingerprints forced empty via a test-only patch
           (b'' pre-placed on each item; the is-not-None hook keeps it empty).
           Expect: image reuse stays dormant -> both images re-encoded
           (images_encoded == 2); outputs equal (recompute is deterministic).
```

- [ ] **Step 6: Syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/vlm_prefix_cache_check.py').read())"`
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/vlm_prefix_cache_check.py
git commit -m "test(scripts): use real fingerprint generator in VLM prefix-cache harness"
```

---

## Task 4: End-to-end verification (GPU, outside sandbox)

**Files:** none (verification only).

All commands here touch CUDA and MUST run outside the sandbox (`required_permissions: ["all"]`). The sandbox has no NVIDIA driver. First confirm an empty GPU.

- [ ] **Step 1: Check for an empty GPU**

Run: `nvidia-smi`
Expected: identify an H200 (140+ GB VRAM) with ~0% utilization. A 27B fp16 model needs ~54 GB and fits comfortably on a single H200, so run **TP1 on one GPU** (`--tp 1 --gpus 0`). Do NOT use TP2 -- the whole model + KV cache fits on one card.

- [ ] **Step 2: Scenario `reuse` (real generator, image KV reuse + ViT skip)**

Run:
```bash
python scripts/vlm_prefix_cache_check.py \
  --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --scenario reuse
```
Expected: `RESULT: OK`. Specifically: `images_encoded == 1` (warm image reused -> only the cold image was encoded), warm text == cold text, both responses coherent (>=128 tokens, on-topic; gibberish = bug).

- [ ] **Step 3: Scenario `distinct` (real generator, no false hit)**

Run:
```bash
python scripts/vlm_prefix_cache_check.py \
  --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --scenario distinct
```
Expected: `RESULT: OK`. `images_encoded == 2` (the second, different image is re-encoded -- proves different pixel content -> different digest -> no false hit).

- [ ] **Step 4: Scenario `dormant` (forced empty fingerprint -> no image reuse)**

Run:
```bash
python scripts/vlm_prefix_cache_check.py \
  --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --scenario dormant
```
Expected: `RESULT: OK`. `images_encoded == 2` (empty fingerprints never match, so the image is always re-encoded), outputs coherent and equal (recompute is deterministic).

- [ ] **Step 5: Text-path regression (model script AS IS)**

Run:
```bash
python scripts/test_turbomind_model.py \
  --model-id Qwen/Qwen3.5-27B \
  --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 \
  --enable-prefix-caching \
  --max-new-tokens 128
```
Expected: a coherent, on-topic response of >=128 tokens (gibberish = bug). Guards the shared `PrefixTrie`/`Scheduler` paths against regressions.

- [ ] **Step 6: Record results**

Confirm Steps 2-5 all passed. If all green, the feature works end-to-end: image KV + ViT are reused only when the real fingerprints match (reuse), different images of equal token length never false-hit (distinct), and forcing empty fingerprints re-creates the dormant pre-PR behavior (dormant).

### Debugging Loop (if any scenario fails)

Iterate until fixed (do not stop with active bugs):

1. Re-run the failing scenario and read the printed `log:` file fully (it has the full engine INFO trace).
2. If `reuse` shows `images_encoded != 1` (warm image not reused): the fingerprint the converter computed on the warm run differs from the cold run's, OR the fold/compare disagrees. Check that `to_turbomind_multimodal` calls `_resolve_fingerprint` (Task 2) and that `_image_fingerprint` is deterministic for identical `pixel_values`/`grid_thw`/`modality`/`second_per_grid` (re-run the Task 1 unit test). Then check `matched`/`resume source` lines -- if matching halts before the image, the scheduler fold/compare (consumer PR, already landed) is not seeing the digest.
3. If `distinct` shows `images_encoded != 2` (false hit): different images produced the same digest -- verify `_image_fingerprint` includes the pixel bytes (Task 1 Step 4) and that `make_image(1)`/`make_image(2)` actually differ in content (they do -- different RNG seed).
4. If `dormant` shows `images_encoded != 2` (image reused despite forced empty): the `is-not-None` hook is not preserving `b''` -- re-check `_resolve_fingerprint` (Task 1 Step 4) and `install_dormant_patch` (Task 3 Step 2). Empty fingerprints must stay empty so `Fingerprint::operator==` (empty never equal) blocks the match.
5. If gibberish output on the warm run: KV reuse is returning wrong blocks -- this is a consumer-side regression, not a generator bug; check the consumer PR's first-block fingerprint fold. Rebuild (`cd build && ninja`) and repeat.

---

## Self-review notes (author checklist, already applied)

- **Spec coverage:** spec Section 2 (identity inputs) -> Task 1 helper hashes exactly `(modality, grid_thw, second_per_grid, pixel_values)`; spec Section 4 (serialization) -> Task 1 Step 4; spec Section 5 (`is-not-None` phasing hook) -> Task 1 Step 4 `_resolve_fingerprint` + Task 2; spec Section 6 (data flow unchanged) -> no engine task; spec Section 7 (bfloat16, empty sentinel, `None` spg, rollback via revert) -> Task 1 tests + Task 3 dormant; spec Section 8 (testing) -> Tasks 1, 3, 4.
- **Type/name consistency:** `_image_fingerprint` and `_resolve_fingerprint` are used identically in Task 1 (definition + tests), Task 2 (converter call), Task 3 (dormant patch pre-places `fingerprint = b''` read by the same hook). The converter reads `input_mm['pixel_values']`/`['pixel_values_videos']`, `['image_grid_thw']`/`['video_grid_thw']`, `['modality']`, `['offset']`, and `['second_per_grid']` (video) -- matching the keys `preprocess_utils.get_expanded_mm_items` produces.
- **Verified against live code:** `qwen3_5.py` already imports `torch` and `from lmdeploy.vl.constants import Modality`; the stdlib block is `import math`/`import re`/`from typing import TYPE_CHECKING, Any` (so `hashlib` goes before `math`, `struct` between `re` and `from typing`); `to_turbomind_multimodal` is at line 387 with the `fingerprint=input_mm.get('fingerprint', b'')` argument at line 411; the helpers go immediately before `class Qwen3_5VisionModel:` at line 331 (after `_split_packed_vision_qkv`, lines 325-328); `grid_thw` handling mirrors the converter's `_grid_thw` (`.flatten().tolist()` for tensors, `list(...)` otherwise) so a 2-D `[N,3]` tensor does not crash the helper; `import _turbomind` succeeds on CPU (probed), so `from lmdeploy.turbomind.models.qwen3_5 import _image_fingerprint, _resolve_fingerprint` works for the unit test; the new test file carries the OpenMMLab copyright header matching sibling `test_preprocess_utils.py`; the harness already uses `images_encoded = sum(images_batched)` as the cache-hit oracle and `cache_prompt_boundary=True`/`cache_generation_boundary=True` for Qwen3.5's hybrid attention (unchanged by this plan); `install_fingerprint_patch` was the only `hashlib` user in the harness; the consumer side is already wired (`src/turbomind/models/qwen3_5vit/qwen3_5vit.cc` pushes `MultiModalSpan{interval, item.fingerprint}` into `multimodal_spans`), and `fingerprint` appears only at `qwen3_5.py:411` in all of `lmdeploy/`, so this plan's scope (Python generator + harness + test) is complete.
