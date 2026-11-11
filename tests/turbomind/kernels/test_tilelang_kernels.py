"""Model-driven TileLang kernel provisioning: cache, provider and the GDN
builder's spec.

These tests need neither a GPU nor a TileLang compile; exporters are faked.
"""
import multiprocessing
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.turbomind.kernels.tilelang import cache, flashqla
from lmdeploy.turbomind.kernels.tilelang.flashqla import GdnSpec
from lmdeploy.turbomind.kernels.tilelang.provider import KernelProvider

SPEC = GdnSpec(hv=16, hq=16, head_dim=128, input_dtype=torch.bfloat16, state_dtype=torch.float32)


def _fake_export(log: Path, symbol: str, path: Path, delay: float = 0.0):
    with open(log, 'a') as file:
        file.write(symbol + '\n')
    time.sleep(delay)
    path.write_bytes(b'module ' + symbol.encode())


def _fake_kernels(log, symbols, delay=0.0):
    return {symbol: partial(_fake_export, log, symbol, delay=delay) for symbol in symbols}


def _builds(log: Path):
    return log.read_text().split() if log.exists() else []


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv('LMDEPLOY_KERNEL_CACHE', str(tmp_path / 'explicit'))
    assert cache.cache_root() == tmp_path / 'explicit'
    monkeypatch.delenv('LMDEPLOY_KERNEL_CACHE')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    assert cache.cache_root() == tmp_path / 'xdg' / 'lmdeploy' / 'tilelang'


def test_cache_key_stability_and_invalidation(monkeypatch, tmp_path):
    monkeypatch.setenv('LMDEPLOY_KERNEL_CACHE', str(tmp_path))
    source = tmp_path / 'factory.py'
    source.write_text('kernel = 1\n')
    toolchain = {'target': 'sm_120a', 'tilelang': '0.1.15', 'compiler': '/usr/bin/g++-15'}
    first = cache.cache_directory('family', toolchain, [source])
    assert first == cache.cache_directory('family', dict(reversed(list(toolchain.items()))), [source])
    assert first.parent == tmp_path and first.name.startswith('family-')
    for key, value in [('tilelang', '0.1.16'), ('compiler', '/usr/bin/g++-14'), ('target', 'sm_100a')]:
        assert cache.cache_directory('family', {**toolchain, key: value}, [source]) != first
    source.write_text('kernel = 2\n')
    assert cache.cache_directory('family', toolchain, [source]) != first


def test_ensure_builds_only_missing(tmp_path):
    directory, log = tmp_path / 'kernels', tmp_path / 'log'
    assert cache.ensure(directory, _fake_kernels(log, ['a', 'b'])) == {'a': directory / 'a.so', 'b': directory / 'b.so'}
    assert sorted(_builds(log)) == ['a', 'b']
    cache.ensure(directory, _fake_kernels(log, ['a', 'b', 'c']))
    assert sorted(_builds(log)) == ['a', 'b', 'c']
    cache.ensure(directory, _fake_kernels(log, ['a', 'b', 'c']))
    assert sorted(_builds(log)) == ['a', 'b', 'c']
    assert sorted(p.name for p in directory.glob('*.so')) == ['a.so', 'b.so', 'c.so']


def _ensure_worker(directory, log):
    cache.ensure(Path(directory), _fake_kernels(Path(log), ['x', 'y'], delay=0.5))


def test_concurrent_ensure_builds_each_symbol_once(tmp_path):
    directory, log = tmp_path / 'kernels', tmp_path / 'log'
    context = multiprocessing.get_context('spawn')
    workers = [context.Process(target=_ensure_worker, args=(str(directory), str(log))) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert sorted(_builds(log)) == ['x', 'y']


def test_failed_or_stale_build_leaves_no_visible_module(tmp_path):
    directory, log = tmp_path / 'kernels', tmp_path / 'log'
    # Leftovers of a killed builder: a staging dir with a partial module, and its lock file.
    (directory / '.tmp-dead').mkdir(parents=True)
    (directory / '.tmp-dead' / 'a.so').write_bytes(b'partial')
    (directory / '.lock').touch()

    def broken(path):
        path.write_bytes(b'half')
        raise RuntimeError('compile failed')

    with pytest.raises(RuntimeError, match='compile failed'):
        cache.ensure(directory, {'a': broken})
    assert not (directory / 'a.so').exists()
    cache.ensure(directory, _fake_kernels(log, ['a']))
    assert _builds(log) == ['a'] and (directory / 'a.so').read_bytes() == b'module a'


# ---------------------------------------------------------------------------
# FlashQLA family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('value,expected', [(None, False), ('native', False), ('flashqla', True)])
def test_requested(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('TM_GDR_BACKEND', raising=False)
    else:
        monkeypatch.setenv('TM_GDR_BACKEND', value)
    assert flashqla.requested() is expected


def test_invalid_backend(monkeypatch):
    monkeypatch.setenv('TM_GDR_BACKEND', 'invalid')
    with pytest.raises(ValueError, match='TM_GDR_BACKEND must be native or flashqla, got invalid'):
        KernelProvider(torch.float32, 120).require(SPEC)


def test_symbols():
    assert flashqla.symbols(SPEC) == [
        'flashqla_v2_cumsum_h16', 'flashqla_v2_kkt_h16_q16', 'flashqla_v2_fused_h16_q16_f32'
    ]
    bf16 = GdnSpec(16, 4, 128, torch.bfloat16, torch.bfloat16)
    assert flashqla.symbols(bf16)[-1] == 'flashqla_v2_fused_h16_q4_bf16'


@pytest.mark.parametrize(
    'arch,changes,reason',
    [
        (120, {}, None),
        (100, {}, 'requires sm120'),
        (120, {'input_dtype': torch.float16}, 'requires bfloat16 inputs'),
        (120, {'head_dim': 64}, 'requires head_dim 128'),
        (120, {'state_dtype': torch.float16}, 'requires float32 or bfloat16 recurrent state'),
        (120, {'hq': 3}, 'requires Hv divisible by Hq'),
    ],
)
def test_unsupported_reason(arch, changes, reason):
    spec = GdnSpec(**{**SPEC.__dict__, **changes})
    actual = flashqla.unsupported_reason(spec, arch)
    assert actual is None if reason is None else reason in actual


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


@pytest.fixture
def requested(monkeypatch, tmp_path):
    """Opt in, and replace compiling, runtime loading and C++ registration with
    recorders."""
    monkeypatch.setenv('TM_GDR_BACKEND', 'flashqla')
    calls = SimpleNamespace(ensure=[], runtime=0, registered=[])

    def ensure(spec):
        calls.ensure.append(spec)
        return {'k1': tmp_path / 'k1.so', 'k2': tmp_path / 'k2.so'}

    def load_runtime():
        calls.runtime += 1

    monkeypatch.setattr(flashqla, 'ensure', ensure)
    monkeypatch.setattr(flashqla, 'available', lambda: True)
    monkeypatch.setattr(flashqla, 'register', lambda paths: calls.registered.extend(paths.items()))
    monkeypatch.setattr('lmdeploy.turbomind.kernels.tilelang.runtime.load_runtime', load_runtime)
    return calls


def test_provider_builds_registers_and_memoizes(requested, tmp_path):
    provider = KernelProvider(torch.float32, 120)
    paths = {'k1': str(tmp_path / 'k1.so'), 'k2': str(tmp_path / 'k2.so')}
    assert provider.require(SPEC) == paths
    assert provider.require(SPEC) == paths
    assert requested.ensure == [SPEC] and requested.runtime == 1
    assert requested.registered == list(paths.items())


def test_provider_declines_unsupported(requested):
    assert KernelProvider(torch.float32, 90).require(SPEC) is None
    assert requested.ensure == [] and requested.runtime == 0 and requested.registered == []


def test_provider_builds_nothing_without_tilelang_support(requested, monkeypatch):
    # The extension was built with ENABLE_TILELANG=OFF: no registration binding, so nothing to build for.
    monkeypatch.setattr(flashqla, 'available', lambda: False)
    assert KernelProvider(torch.float32, 120).require(SPEC) is None
    assert requested.ensure == [] and requested.runtime == 0


def test_provider_raises_when_requested_but_tilelang_missing(monkeypatch):
    monkeypatch.setenv('TM_GDR_BACKEND', 'flashqla')
    monkeypatch.setenv('CXX', sys.executable)  # any executable; the version probe is not reached

    def version(name):
        raise flashqla.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(flashqla.metadata, 'version', version)
    monkeypatch.setattr(cache, 'compiler_identity', lambda compiler: 'fake')
    monkeypatch.setattr(flashqla, 'available', lambda: True)
    with pytest.raises(ImportError, match='FlashQLA kernels were requested but tilelang is not installed'):
        KernelProvider(torch.float32, 120).require(SPEC)


def test_not_requested_never_imports_tilelang():
    code = (
        'import sys, torch\n'
        'from lmdeploy.turbomind.kernels.tilelang.provider import KernelProvider\n'
        'from lmdeploy.turbomind.kernels.tilelang.flashqla import GdnSpec\n'
        'spec = GdnSpec(16, 16, 128, torch.bfloat16, torch.float32)\n'
        'assert KernelProvider(torch.float32, 120).require(spec) is None\n'
        "assert 'tilelang' not in sys.modules, 'tilelang imported'\n"
    )
    env = {key: value for key, value in os.environ.items() if key != 'TM_GDR_BACKEND'}
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# DeltaNetBuilder states the spec
# ---------------------------------------------------------------------------


class _RecordingProvider:

    def __init__(self, directory):
        self.state_dtype = torch.bfloat16
        self.directory = directory
        self.specs = []

    def require(self, spec):
        self.specs.append(spec)
        return self.directory


@pytest.mark.parametrize('tp', [1, 2])
def test_deltanet_builder_states_post_tp_spec(tp):
    from lmdeploy.turbomind import _tm
    from lmdeploy.turbomind.builders._base import ParallelGroup
    from lmdeploy.turbomind.builders.deltanet import DeltaNetBuilder

    config = _tm.DeltaNetConfig()
    config.num_k_heads, config.num_v_heads = 16, 32
    config.key_head_dim = config.value_head_dim = 128
    provider = _RecordingProvider({'k': '/k.so'})
    ctx = SimpleNamespace(
        active_mask=(True, ) * tp, data_type=_tm.DataType.TYPE_BF16, dtype=torch.bfloat16, kernels=provider
    )
    DeltaNetBuilder(config, ctx, tp=ParallelGroup(tp, list(range(tp))))
    assert provider.specs == [GdnSpec(32 // tp, 16 // tp, 128, torch.bfloat16, torch.bfloat16)]


def test_deltanet_builder_without_provider():
    from lmdeploy.turbomind import _tm
    from lmdeploy.turbomind.builders._base import ParallelGroup
    from lmdeploy.turbomind.builders.deltanet import DeltaNetBuilder

    config = _tm.DeltaNetConfig()
    ctx = SimpleNamespace(active_mask=(True, ), data_type=_tm.DataType.TYPE_BF16, dtype=torch.bfloat16, kernels=None)
    DeltaNetBuilder(config, ctx, tp=ParallelGroup(1, None))  # no provider: nothing to require
