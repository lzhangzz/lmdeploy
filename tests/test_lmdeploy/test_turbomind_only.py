# Copyright (c) OpenMMLab. All rights reserved.
import json
import subprocess
import sys

import pytest

import lmdeploy
from lmdeploy.archs import check_backend, check_backend_config, check_turbomind_support
from lmdeploy.messages import TurbomindEngineConfig


@pytest.mark.parametrize('backend', [None, 'turbomind'])
def test_check_backend_accepts_turbomind(backend):
    check_backend(backend)


@pytest.mark.parametrize('backend', ['pytorch', 'auto', ''])
def test_check_backend_rejects_other_backends(backend):
    with pytest.raises(ValueError, match='TurboMind is the only inference backend'):
        check_backend(backend)


def test_check_backend_config_defaults_to_turbomind():
    assert isinstance(check_backend_config(None), TurbomindEngineConfig)
    config = TurbomindEngineConfig(tp=1)
    assert check_backend_config(config) is config


def test_check_backend_config_rejects_other_types():
    with pytest.raises(TypeError, match='TurbomindEngineConfig'):
        check_backend_config(dict(tp=1))


def _write_config(tmp_path, architecture, model_type):
    (tmp_path / 'config.json').write_text(json.dumps({'architectures': [architecture], 'model_type': model_type}))
    return str(tmp_path)


def test_check_turbomind_support_rejects_unsupported_model(tmp_path):
    model_path = _write_config(tmp_path, 'GPT2LMHeadModel', 'gpt2')
    with pytest.raises(ValueError, match='not supported by the TurboMind engine'):
        check_turbomind_support(model_path)


def test_check_turbomind_support_requires_native_extension(tmp_path, monkeypatch):
    from lmdeploy import turbomind

    monkeypatch.setattr(turbomind, 'is_available', lambda: False)
    model_path = _write_config(tmp_path, 'Qwen3ForCausalLM', 'qwen3')
    with pytest.raises(RuntimeError, match='native extension is unavailable'):
        check_turbomind_support(model_path)


def test_public_exports_are_turbomind_only():
    assert hasattr(lmdeploy, 'TurbomindEngineConfig')
    for name in ('PytorchEngineConfig', 'KVTransferConfig', 'VisionConfig'):
        assert not hasattr(lmdeploy, name)
        assert name not in lmdeploy.__all__


def test_vision_config_argument_is_rejected():
    from lmdeploy.serve.core.async_engine import AsyncEngine
    with pytest.raises(TypeError, match='vision_config was removed'):
        AsyncEngine('unused-model', vision_config=dict(max_batch_size=8))


def _run_cli(*args):
    return subprocess.run([sys.executable, '-c', 'from lmdeploy.cli import run; run()', *args],
                          capture_output=True,
                          text=True,
                          timeout=120)


def test_cli_rejects_pytorch_backend():
    result = _run_cli('serve', 'api_server', 'unused-model', '--backend', 'pytorch')
    assert result.returncode == 2
    assert 'the PyTorch engine has been removed' in result.stderr


@pytest.mark.parametrize('option', [
    ['--adapters', 'a=b'],
    ['--eager-mode'],
    ['--logprobs-mode', 'raw_logprobs'],
    ['--enable-return-routed-experts'],
    ['--quant-policy', '42'],
    ['--vision-max-batch-size', '2'],
])
def test_cli_rejects_removed_options(option):
    result = _run_cli('serve', 'api_server', 'unused-model', *option)
    assert result.returncode == 2
