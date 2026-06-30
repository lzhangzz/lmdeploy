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
