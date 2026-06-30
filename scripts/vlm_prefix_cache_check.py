#!/usr/bin/env python3
"""Manual harness: verify TurboMind native-Qwen3.5 ViT prefix caching for images.

Scenarios (greedy decoding, sequential requests in one pipeline so the 2nd sees
the 1st's published blocks):

  reuse    same image + same prompt twice, stand-in fingerprint injected.
           Expect: warm request reuses the image span -> ViT logs images_batched=0
           (skip); warm text == cold text.
  distinct two DIFFERENT images of equal token length, fingerprints injected.
           Expect: NO false hit -> the 2nd image is re-encoded (no ViT line with
           images_batched=0); both outputs non-empty.
  dormant  same image twice, NO fingerprint injected (empty fp).
           Expect: image reuse stays dormant -> 2nd image re-encoded (no
           images_batched=0 line); outputs non-empty and equal (recompute).

Run OUTSIDE the sandbox (needs a GPU):

  python scripts/vlm_prefix_cache_check.py \
      --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
      --tp 1 --gpus 0 --scenario reuse
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import sys
import tempfile


def _set_hf_cache(cache_dir: str) -> None:
    import huggingface_hub.constants as hf_constants
    hf_constants.HF_HUB_OFFLINE = 1
    hf_constants.HF_HUB_CACHE = cache_dir


def make_image(seed: int, size=(448, 448)):
    """Deterministic RGB image; same seed -> identical pixels, different seed ->
    different content. Fixed size -> identical Qwen grid_thw (equal token len)."""
    import random
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    img = Image.new('RGB', size, (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    d = ImageDraw.Draw(img)
    for _ in range(10):
        x0, y0 = rng.randint(0, size[0] - 1), rng.randint(0, size[1] - 1)
        x1, y1 = rng.randint(0, size[0] - 1), rng.randint(0, size[1] - 1)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=color)
    return img


def install_fingerprint_patch() -> None:
    """Inject sha256(pixels) at the converter input dict; exercises the real
    `input_mm.get('fingerprint')` path. Mirrors the future generator."""
    from lmdeploy.turbomind.models.qwen3_5 import Qwen3_5VisionModel
    if getattr(Qwen3_5VisionModel, '_fp_patched', False):
        return
    _orig = Qwen3_5VisionModel.to_turbomind_multimodal

    def _patched(self, multimodal):
        import torch
        for mm in multimodal:
            pv = mm.get('pixel_values', mm.get('pixel_values_videos'))
            # Reinterpret the raw tensor bytes as uint8 so the digest is dtype-agnostic
            # (numpy cannot consume bfloat16 directly). Deterministic per pixel content.
            raw = pv.contiguous().cpu().view(torch.uint8).numpy().tobytes()
            mm['fingerprint'] = hashlib.sha256(raw).digest()
        return _orig(self, multimodal)

    Qwen3_5VisionModel.to_turbomind_multimodal = _patched
    Qwen3_5VisionModel._fp_patched = True


@contextlib.contextmanager
def capture_low_level_output():
    """Redirect fds 1 & 2 to a temp file so C++ TM_LOG output is captured."""
    f = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    saved_out, saved_err = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    try:
        yield f.name
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        f.close()


VIT_RE = re.compile(r'Qwen3\.5 ViT setup: mm_seqs=(\d+) images_batched=(\d+) images_skipped=(\d+) patches=(\d+)')
MATCH_RE = re.compile(r'matched \[0,(\d+)\) \((\d+) blk')
RESUME_RE = re.compile(r'resume \[0,(\d+)\).*source=(\w+)')


def parse_log(path: str):
    vit, matched, resume = [], [], []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            if (m := VIT_RE.search(line)):
                vit.append(tuple(int(x) for x in m.groups()))
            if (m := MATCH_RE.search(line)):
                matched.append(tuple(int(x) for x in m.groups()))
            if (m := RESUME_RE.search(line)):
                resume.append((int(m.group(1)), m.group(2)))
    return vit, matched, resume


def run(args) -> int:
    _set_hf_cache(args.cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # The C++ TurboMind logger reads its level from TM_LOG_LEVEL; pipeline(...) only
    # setdefault()s it, so set it explicitly to guarantee the INFO lines we assert on.
    os.environ['TM_LOG_LEVEL'] = 'INFO'

    inject = args.scenario in ('reuse', 'distinct')
    if inject:
        install_fingerprint_patch()

    if args.scenario == 'distinct':
        images = [make_image(1), make_image(2)]  # different content, equal size
    else:
        img = make_image(1)
        images = [img, img]  # same image twice

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    engine_config = TurbomindEngineConfig(
        tp=args.tp,
        session_len=8192,
        cache_max_entry_count=0.5,
        enable_prefix_caching=True,
        enable_metrics=False,
        # Qwen3.5 is a hybrid linear/full-attention model: linear-attention layers
        # carry recurrent state, so cross-request prefix resume (and the resulting
        # ViT-skip) requires a boundary checkpoint to be published/restored.
        cache_prompt_boundary=True,
        cache_generation_boundary=True,
    )
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    prompt = args.prompt

    texts = []
    with capture_low_level_output() as log_path:
        with pipeline(args.model_id, backend_config=engine_config, log_level='INFO',
                      trust_remote_code=True) as pipe:
            for image in images:
                out = pipe((prompt, image), gen_config=gen_config)
                texts.append(out.text if hasattr(out, 'text') else str(out))

    vit, matched, resume = parse_log(log_path)

    # --- report (now that fds are restored) ---
    print(f'=== scenario: {args.scenario} (inject_fingerprint={inject}) ===')
    print(f'log: {log_path}')
    for i, t in enumerate(texts):
        print(f'--- response {i} ({len(t)} chars) ---')
        print(t.strip()[:400])
    print(f'ViT setup lines (mm_seqs, images_batched, images_skipped, patches): {vit}')
    print(f'matched (M, blk): {matched}')
    print(f'resume (history, source): {resume}')

    # --- assertions ---
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(('PASS: ' if cond else 'FAIL: ') + msg)
        ok = ok and cond

    check(all(t.strip() for t in texts), 'both responses are non-empty')
    check(len(vit) >= 1, 'ViT setup was logged at least once')
    # Total images actually run through the ViT across the whole run. This is the
    # robust cache-hit signal: a reused image contributes 0, a (re)encoded image
    # contributes 1. The per-line `images_batched==0` heuristic is NOT reliable
    # because a single request's prefill can emit a follow-up multimodal pass with
    # images_batched=0 even for a freshly-encoded image (chunked/boundary prefill).
    images_encoded = sum(b for (_, b, _, _) in vit)
    # Two requests, one image each.
    print(f'images_encoded (sum of images_batched) = {images_encoded}')

    if args.scenario == 'reuse':
        check(images_encoded == 1,
              f'warm image reused: only the cold image was encoded (images_encoded={images_encoded}, want 1)')
        check(texts[0].strip() == texts[1].strip(), 'warm text == cold text (greedy oracle)')
    elif args.scenario == 'distinct':
        check(images_encoded == 2,
              f'no false hit: both distinct images were encoded (images_encoded={images_encoded}, want 2)')
    elif args.scenario == 'dormant':
        check(images_encoded == 2,
              f'image reuse dormant: both images re-encoded (images_encoded={images_encoded}, want 2)')
        check(texts[0].strip() == texts[1].strip(), 'recompute is deterministic (texts equal)')

    print('RESULT:', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


def main(argv) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-id', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--scenario', choices=['reuse', 'distinct', 'dormant'], required=True)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--prompt', default='Describe this image in detail.')
    return run(p.parse_args(argv[1:]))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
