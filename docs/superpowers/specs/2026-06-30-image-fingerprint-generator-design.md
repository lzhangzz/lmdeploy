# Design: Image fingerprint generator for TurboMind VLM prefix caching

- Date: 2026-06-30
- Status: Proposed design, pre-implementation
- Scope: The Python-side **fingerprint generator** for the native Qwen3.5 C++ ViT
  VLM path. It produces the per-image SHA-256 digest that the already-landed
  fingerprint *consumer* (`Qwen3_5VitItem.fingerprint` -> `Sequence::multimodal_spans`
  -> scheduler fold/compare, per
  `docs/superpowers/specs/2026-06-29-vlm-prefix-caching-turbomind-design.md`)
  currently consumes as an empty sentinel. This PR un-dormants image-span caching.
  No C++ engine, scheduler, prefix-trie, binding, or contract change is in scope --
  the consumer side is complete; only the generator is missing.

## 1. Goal

Produce a real per-image SHA-256 fingerprint so the existing consumer machinery
reuses image-span KV (and skips the ViT forward) across requests that share an
image. The fingerprint must be a **correct cache-identity key**: two requests get
the same fingerprint iff the ViT would produce identical embeddings **and** the
cached LM KV for the image-token span is identical. A false hit (same fingerprint,
different KV) corrupts output; a false miss (different fingerprint, same KV) only
loses reuse.

## 2. Identity analysis -- what varies per request

For a fixed raw image, the Qwen3.5 ViT receives `(pixel_values, grid_thw)` (plus
the modality selecting image-vs-video tokenization). Tracing the preprocessor
(`lmdeploy/vl/model/base.py::preprocess`, `lmdeploy/vl/model/preprocess_utils.py`):

- `min_pixels` / `max_pixels` -- the size override from per-request
  `mm_processor_kwargs['image']` (`get_override_size`, `preprocess_utils.py:32`).
  Used transiently to build `images_kwargs['size']` (`base.py:157-164`), then
  **discarded** -- it does not survive onto the `input_mm` item dict. Its *effect*
  (the `smart_resize` target) is baked into `pixel_values` and `grid_thw`.
- modality (IMAGE / VIDEO) -- per-request; disambiguates same-pixels image vs
  single-frame video.
- video frame-sampling (`fps`, `num_frames`, `video_metadata`) -- per-request, but
  its effect is baked into the sampled frames (`pixel_values_videos`).
- `second_per_grid` (video) -- per-request, from video metadata. **Does not change
  the ViT input** (pixels unchanged) but drives mRoPE temporal positions for the
  image-token span, so it changes the **LM KV stored in the prefix**. A pixel-only
  hash false-hits across different `second_per_grid`.
- per-model constants (`patch_size`, `temporal_patch_size`, `spatial_merge_size`,
  default `shortest_edge`/`longest_edge`, normalization mean/std, processor
  version) -- fixed at engine init, constant across all requests. The prefix cache
  is in-memory and dies with the process, so a model swap implies a new process and
  an empty cache; constants cannot cause a stale hit. Omitted from the hash.

So the minimal correct set of identity inputs is: the ViT-forward tensor
(`pixel_values`), the spatial layout (`grid_thw`), the modality, and `second_per_grid`
(video). `min_pixels`/`max_pixels` and frame-sampling are captured *via their
effect* on `pixel_values`/`grid_thw`; the per-model constants are constant and
omitted.

## 3. Phasing decision (A now, B later)

A future refactor will **skip preprocessing entirely** when the image-span KV is
directly reusable. That requires the fingerprint to be computable *before* the HF
image processor runs -- otherwise preprocessing has already been paid for, defeating
the skip.

- **Phase A (this PR): post-preprocess.** Compute the fingerprint from the actual
  ViT inputs (`pixel_values`, `grid_thw`, `modality`, `second_per_grid`), all
  present on the `input_mm` dict at converter time. Minimal plumbing; activates
  image + video span caching immediately. Cannot support skip-preprocessing (the
  hash input does not exist until the processor runs).
- **Phase B (future PR): pre-preprocess.** Compute the fingerprint from the raw
  image content hash + `grid_thw` derived via `smart_resize(W, H, min_pixels,
  max_pixels, patch, merge)` (pure arithmetic on raw dimensions + knobs, no pixel
  ops), so it is available before the HF processor. Enables skip-preprocessing for
  images. Video stays post-preprocess in that PR (`second_per_grid` and the sampled
  frames are produced by the HF video processor; replicating them pre-preprocess is
  fragile).

The engine treats the digest as opaque and the cache is in-memory, so the A->B
swap is a localized generator move with no migration. The converter is structured
to make that swap a no-converter-change (Section 5).

## 4. Fingerprint content & canonical serialization

`SHA-256` over a canonical, fixed-width-framed byte layout of exactly the
ViT-forward inputs plus the one mRoPE scalar:

```
fingerprint = SHA256( modality_byte              # 1 byte: 0=IMAGE, 1=VIDEO
                    | grid_thw                   # 3 x int32 LE: (t, h, w)
                    | second_per_grid_present    # 1 byte: 0 (image / None) or 1
                    | [second_per_grid]          # 1 x float64 LE, only if present
                    | pixel_values_bytes )       # pv.contiguous().cpu().numpy().tobytes()
```

Concrete generator -- a Qwen3.5-specific module helper in
`lmdeploy/turbomind/models/qwen3_5.py`:

```python
import hashlib
import struct

from lmdeploy.vl.constants import Modality


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
    t, h, w = (int(x) for x in (gthw.tolist() if hasattr(gthw, 'tolist') else gthw))
    spg = input_mm.get('second_per_grid')          # video only; float | None

    h_obj = hashlib.sha256()
    h_obj.update(struct.pack('<B', 1 if is_video else 0))
    h_obj.update(struct.pack('<3i', t, h, w))
    h_obj.update(struct.pack('<B', 0 if spg is None else 1))
    if spg is not None:
        h_obj.update(struct.pack('<d', float(spg)))
    h_obj.update(pv.contiguous().cpu().numpy().tobytes())
    return h_obj.digest()                          # 32 bytes; never all-zero
```

Why each field:

- `pixel_values` bytes -- the patch tensor the ViT consumes; captures the raw
  image content, the effect of `min_pixels`/`max_pixels` (a change that alters the
  resize changes the bytes), and video frame sampling (changes the sampled frames).
- `grid_thw` -- the spatial layout; required because `pixel_values` shape alone
  does not fully determine `(t, h, w)` (e.g. same `num_patches`, different
  `(t, h, w)` split).
- `modality` -- disambiguates an image vs a single-frame video that could share
  pixel bytes.
- `second_per_grid` (video) -- drives mRoPE temporal positions for the image-token
  span, so it changes the LM KV stored in the prefix without changing the ViT
  input. Folding it in prevents false hits across different `second_per_grid`.
- The pixel tensor's `dtype` is intentionally **not** hashed: `mm_feature_dtype` is
  fixed at engine init (`set_mm_feature_dtype`), so it is constant across all
  requests and cannot affect fingerprint equality -- consistent with the "hash only
  things that vary per request" rule of Section 2.

Fixed-width framing (no length ambiguity) means e.g. `grid_thw=(1,28,28)` and
`(1,2,828)` can never collide. SHA-256 never yields the all-zero digest, so the
reserved "empty" sentinel (empty never compares equal, including to itself) is
never produced here.

## 5. Generator location & the A->B phasing hook

The fingerprint is computed in the converter,
`Qwen3_5VisionModel.to_turbomind_multimodal` (`lmdeploy/turbomind/models/qwen3_5.py:387`),
with a **pre-placed-override** so the future B refactor needs no converter change:

```python
def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
    items = []
    for input_mm in multimodal:
        modality = input_mm.get('modality')
        if modality == Modality.IMAGE or modality == Modality.IMAGE.value:
            data = self._tm_tensor(input_mm['pixel_values'])
            grid_thw = self._grid_thw(input_mm['image_grid_thw'])
            tm_modality = _tm.multimodal.Modality.IMAGE
        elif modality == Modality.VIDEO or modality == Modality.VIDEO.value:
            data = self._tm_tensor(input_mm['pixel_values_videos'])
            grid_thw = self._grid_thw(input_mm['video_grid_thw'])
            tm_modality = _tm.multimodal.Modality.VIDEO
        else:
            raise ValueError(f'Qwen3.5 TurboMind does not support modality {modality!r}')

        token_begin, token_end = self._offset_pair(input_mm['offset'])
        # If a fingerprint was pre-placed (future pre-preprocess generator, or a
        # test forcing empty/dormant), use it as-is; otherwise derive it from the
        # ViT inputs (phase A). `is not None` (not `or`) so an explicit b'' stays
        # empty rather than falling through to compute.
        fingerprint = input_mm.get('fingerprint')
        if fingerprint is None:
            fingerprint = _image_fingerprint(input_mm)
        items.append(
            _tm.multimodal.Qwen3_5VitItem(
                modality=tm_modality,
                data=data,
                token_begin=token_begin,
                token_end=token_end,
                grid_thw=grid_thw,
                fingerprint=fingerprint,
            ))
    return _tm.multimodal.Qwen3_5VitInput(items)
```

- **Phase A (this PR):** nothing pre-places `fingerprint`, so `get` returns `None`
  and `_image_fingerprint(input_mm)` computes it post-preprocess. The existing
  `input_mm.get('fingerprint', b'')` stand-in is replaced by this compute-if-absent.
- **Phase B (future):** the preprocessor side computes the fingerprint before the
  HF image processor runs and pre-places it on the item dict; the converter takes
  the pre-placed branch and `_image_fingerprint` goes unused. No converter change,
  no engine change.
- `_image_fingerprint` is Qwen3.5-specific, so it lives in the Qwen3.5 TurboMind
  model module (not the shared `preprocess_utils.py`).

## 6. Data flow (unchanged downstream)

The digest flows through the already-landed consumer plumbing -- no engine,
scheduler, prefix-trie, binding, or contract change in this PR:

```
_image_fingerprint(input_mm) -> 32-byte SHA-256
  -> Qwen3_5VitItem.fingerprint (py::bytes, via bind.cpp's 0/32-byte validator)
  -> [kAdd] Sequence.multimodal_spans[] = {interval, Fingerprint}
  -> [Accept] MatchPrompt/CreateMissingBlocks/PublishGeneration fold the
     start-fingerprint into PrefixKey at the image's first block; exact compare
     in PrefixTrie::Find
  -> [Resume/Schedule] valid matched image blocks raise resume_len/history_len
  -> [Qwen3_5Vit::Setup] window-intersection filter skips the ViT for images
     fully below history_len (images_batched=0)
```

The only new code is `_image_fingerprint` and the three-line
`fingerprint = input_mm.get('fingerprint'); if fingerprint is None: fingerprint =
_image_fingerprint(input_mm)` hook in the converter.

## 7. Edge cases & error handling

- **Empty sentinel:** the generator always returns a real 32-byte digest (never
  all-zero), so for the Qwen3.5 native path fingerprints are never empty --
  image-span caching is always active (the intended un-dormanting). The empty
  fingerprint code path (`Fingerprint::operator==` false when either side empty)
  stays for safety: non-Qwen3.5 paths and any future modality that opts out. Its
  semantics are locked by the existing C++ unit test
  (`src/turbomind/engine/test_prefix_trie.cc`).
- **`second_per_grid` is `None`:** serialized as `present=0` with no value bytes.
  For images it is always absent; for videos it may be `None` if the processor did
  not produce it -- handled identically. The `modality` byte distinguishes
  "image, no spg" from "video, spg=None" so they cannot collide.
- **`grid_thw` as tensor or list:** handled via `hasattr(gthw, 'tolist')`.
- **Tensor not contiguous / on GPU:** `.contiguous().cpu()` before `.numpy()`; one
  host sync per image (same as the existing test stand-in). Acceptable for prefill;
  this is the cost of phase A and disappears for images in phase B (which hashes a
  precomputed content hash, not the pixel tensor).
- **Collisions:** SHA-256 over the full pixel content plus structured fields is
  practically zero. The exact `PrefixTrie::Find` compare on the digest already
  guards the hash-bucket step.
- **Rollback:** no kill-switch / env var. Revert the commit to return to the
  dormant pre-PR behavior. The empty-fingerprint semantics stay covered by the
  C++ unit test, so the dormant runtime state remains safe if it is ever reached
  (e.g. by a partial revert or a non-Qwen3.5 path).

## 8. Testing

- **Update `scripts/vlm_prefix_cache_check.py`** (already landed): remove the
  `install_fingerprint_patch` monkeypatch -- the real generator now computes
  fingerprints, so:
  - `reuse`: identical image + grid -> identical digest -> warm run logs
    `images_batched=0` (ViT skip) and warm text == cold text.
  - `distinct`: two different images of equal token length -> different pixel
    bytes -> different digests -> no false hit (no `images_batched=0`).
  - `dormant`: dropped as a default-runtime scenario (the generator always
    populates fingerprints, so the dormant state cannot occur for Qwen3.5). The
    empty-fingerprint *semantics* remain covered by the C++ unit test
    (`test_prefix_trie.cc`). If a runtime dormant check is still wanted, drive it
    with a test-only monkeypatch that pre-places `fingerprint = b''` on each item
    dict -- the converter's `is not None` hook (Section 5) treats an explicit `b''`
    as "use as-is", so the digest stays empty and image reuse stays dormant (test
    scaffolding, not product surface).
- **C++ unit test** (`src/turbomind/engine/test_prefix_trie.cc`, already landed)
  keeps covering `Fingerprint` equality semantics (empty != empty, distinct !=,
  identical ==) -- no change.
- **New Python unit test** for `_image_fingerprint` determinism/equality: same
  inputs -> same 32 bytes; change pixel content / `grid_thw` / `second_per_grid` /
  modality each -> different bytes; `None` vs float `second_per_grid` differ; output
  is 32 bytes and never all-zero. Pure-CPU, no GPU.
- **Model-level regression** via `scripts/test_turbomind_model.py` (AS IS) guards
  the shared text path; the VLM harness above guards the image path (GPU, outside
  the sandbox; check `nvidia-smi` for an empty GPU first; verify the response every
  time -- >=128 tokens, coherent, on-topic; gibberish = bug).

## 9. Out of scope

- **Skip-preprocessing (phase B)** -- the pre-preprocess generator + the engine-side
  early trie lookup that skips the HF image processor and ViT on a full image-span
  hit. Separate future PR (Section 10).
- Legacy Python-embedding VLM path, audio modality, PyTorch backend.
- Any C++ engine / scheduler / prefix-trie / binding / contract change -- the
  consumer side is complete (`contracts.prefix-identity`, `contracts.prefix-prepare`
  in `src/turbomind/engine/README.md` already cover fingerprint folding).

## 10. Forward-looking: skip-preprocessing refactor (phase B, future PR)

- Precompute a format-independent **raw image content hash** at `load_image` /
  `ImageMediaIO` time (SHA-256 of the decoded RGB array) and stash it on the image
  object, so the per-request fingerprint costs nothing.
- Compute `grid_thw` **pre-preprocess** via `smart_resize(W, H, min_pixels,
  max_pixels, patch, merge)` -- pure arithmetic on raw dimensions + knobs, no pixel
  ops.
- Assemble the image fingerprint pre-preprocess:
  `SHA256(image_content_hash, grid_thw, modality)`, pre-place it on the item dict,
  and the converter's override branch (Section 5) picks it up -- no converter
  change.
- Add the engine-side **early trie lookup** that, on a full image-span hit, skips
  the HF image processor and the ViT forward entirely (reusing cached KV). This is
  the new engine logic; the fingerprint generator is just its enabler.
- **Video stays post-preprocess** in that PR too: `second_per_grid` and the sampled
  frames are produced by the HF video processor, so replicating them pre-preprocess
  is fragile. Video caching still activates (post-preprocess fingerprint);
  skip-preprocessing is image-first.

## Data flow summary

```
phase A (this PR):
  [preprocess] HF processor -> pixel_values, grid_thw, second_per_grid on item dict
  [to_turbomind_multimodal] _image_fingerprint(input_mm) -> 32-byte SHA-256
    (or a pre-placed fingerprint if present)
  -> Qwen3_5VitItem.fingerprint -> Sequence.multimodal_spans -> scheduler fold/compare
  -> ViT skip on a full image-span cache hit

phase B (future):
  [load_image] image_content_hash stashed on the image
  [preprocess, pre-processor] grid_thw = smart_resize(W,H,min/max_pixels,patch,merge)
  [preprocess, pre-processor] fp = SHA256(image_content_hash, grid_thw, modality)
    pre-placed on the item dict
  [to_turbomind_multimodal] uses the pre-placed fingerprint (no compute)
  -> early trie lookup skips the HF image processor + ViT on a full image-span hit
```
