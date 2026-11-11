# Copyright (c) OpenMMLab. All rights reserved.
"""FlashQLA ordinary-prefill kernels for Gated DeltaNet layers.

The GDN layer builder states a :class:`GdnSpec`; this module decides
whether the kernels are requested and buildable, names them, exports them, and
registers them with FlashQLA's own C++ kernel registry
(``_tm.add_flashqla_kernel``).
Runtime policy (CP level, ``TM_GDR_FORCE_LEGACY``, speculative rows) is decided
by the C++ ``Match`` only.

The TileLang kernel factories live in modules named after their FlashQLA
sources (``cumsum``, ``kkt_solve``, ``fused_fwd``) and are imported only when a
kernel is actually built, so importing this package never imports TileLang.

Symbol names carry the positional ABI (``v2``) and every specialization
parameter; C++ computes the same names
(``src/turbomind/kernels/linear_attn/kernel/flashqla/host.cc``).
"""
import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import metadata
from pathlib import Path

import torch

from .. import cache

NAME = 'flashqla_gdn_prefill'
TARGET = 'sm_120a'  # plain sm_120 does not enable the fused register-reconfiguration instructions
VERSIONS = {'tilelang': '0.1.15', 'apache-tvm-ffi': '0.1.11'}
SOURCES = tuple(Path(__file__).with_name(name) for name in ('__init__.py', 'cumsum.py', 'kkt_solve.py', 'fused_fwd.py'))
_STATE_CODES = {torch.float32: 'f32', torch.bfloat16: 'bf16'}


@dataclass(frozen=True)
class GdnSpec:
    """Kernel requirements of one GDN layer, after tensor-parallel sharding."""
    hv: int
    hq: int
    head_dim: int
    input_dtype: torch.dtype
    state_dtype: torch.dtype

    @property
    def family(self):
        return sys.modules[__name__]


def requested() -> bool:
    """Explicit opt-in: ``TM_GDR_BACKEND=flashqla``.

    The only parser of the variable.
    """
    value = os.environ.get('TM_GDR_BACKEND', 'native')
    if value not in ('native', 'flashqla'):
        raise ValueError(f'TM_GDR_BACKEND must be native or flashqla, got {value}')
    return value == 'flashqla'


def available() -> bool:
    """Whether this TurboMind build can run the kernels (``ENABLE_TILELANG=ON``
    adds the registration binding)."""
    from lmdeploy.turbomind import _tm
    return hasattr(_tm, 'add_flashqla_kernel')


def register(paths: dict[str, str]):
    """Register ``symbol -> path`` with FlashQLA's C++ kernel registry."""
    from lmdeploy.turbomind import _tm
    for symbol, path in paths.items():
        _tm.add_flashqla_kernel(symbol, path)


def unsupported_reason(spec: GdnSpec, arch: int) -> str | None:
    """Why this family cannot build ``spec`` for ``arch`` (e.g. 120 for sm120),
    or None."""
    if arch != 120:
        return f'requires sm120, device is sm{arch}'
    if spec.input_dtype != torch.bfloat16:
        return f'requires bfloat16 inputs, got {spec.input_dtype}'
    if spec.head_dim != 128:
        return f'requires head_dim 128, got {spec.head_dim}'
    if spec.state_dtype not in _STATE_CODES:
        return f'requires float32 or bfloat16 recurrent state, got {spec.state_dtype}'
    if spec.hq <= 0 or spec.hv <= 0 or spec.hv % spec.hq:
        return f'requires Hv divisible by Hq, got Hv={spec.hv} Hq={spec.hq}'
    return None


def _factories(spec: GdnSpec) -> dict[str, tuple[str, str, tuple]]:
    """``symbol -> (module, factory, args)``."""
    hv, hq = spec.hv, spec.hq
    state = _STATE_CODES[spec.state_dtype]
    state_name = 'float32' if spec.state_dtype == torch.float32 else 'bfloat16'
    return {
        f'flashqla_v2_cumsum_h{hv}': ('cumsum', 'cumsum_prefill', (hv, )),
        f'flashqla_v2_kkt_h{hv}_q{hq}': ('kkt_solve', 'kkt_prefill', (hv, hq)),
        f'flashqla_v2_fused_h{hv}_q{hq}_{state}': ('fused_fwd', 'fused_prefill', (hv, hq, state_name)),
    }


def symbols(spec: GdnSpec) -> list[str]:
    return list(_factories(spec))


def _check_versions():
    import tilelang
    import tvm_ffi
    installed = {'tilelang': tilelang.__version__, 'apache-tvm-ffi': tvm_ffi.__version__}
    if installed != VERSIONS:
        raise RuntimeError(f'FlashQLA kernels require {VERSIONS}, installed {installed}')


def _export(module: str, factory: str, args: tuple, symbol: str, compiler: str, path: Path):
    import tilelang

    _check_versions()
    tilelang.disable_cache()
    factories = importlib.import_module(f'.{module}', __name__)
    function = getattr(factories, factory).get_tir(*args).with_attr('global_symbol', symbol)
    kernel = tilelang.compile(
        function,
        out_idx=[],
        execution_backend='tvm_ffi',
        target={'kind': 'cuda', 'arch': TARGET},
        compile_flags=[f'-ccbin={compiler}'],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    kernel.export_library(str(path))


def kernels(spec: GdnSpec, compiler: str) -> dict[str, Callable[[Path], None]]:
    """``symbol -> exporter(path)`` for every kernel ``spec`` needs."""
    return {
        symbol: partial(_export, module, factory, args, symbol, compiler)
        for symbol, (module, factory, args) in _factories(spec).items()
    }


def ensure(spec: GdnSpec) -> dict[str, Path]:
    """``symbol -> module path`` for every kernel ``spec`` needs, building
    missing ones.

    Raises if TileLang is unavailable or a build fails.
    """
    compiler = cache.host_compiler()
    try:
        versions = {name: metadata.version(name) for name in VERSIONS}
    except metadata.PackageNotFoundError as error:
        raise ImportError(f'FlashQLA kernels were requested but {error.name} is not installed') from error
    toolchain = {
        'target': TARGET,
        'compiler': compiler,
        'compiler_version': cache.compiler_identity(compiler),
        **versions,
    }
    directory = cache.cache_directory(NAME, toolchain, SOURCES)
    return cache.ensure(directory, kernels(spec, compiler))
