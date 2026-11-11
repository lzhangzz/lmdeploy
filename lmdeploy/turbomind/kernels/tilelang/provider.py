# Copyright (c) OpenMMLab. All rights reserved.
"""Per-model kernel provider handed to layer builders through the build
context.

Mirrors how builders query ``ctx.gemm`` for linear weights: a builder states
what its layer needs, the provider decides whether and how to supply it.
"""
import torch

from lmdeploy.utils import get_logger

from . import runtime

logger = get_logger('lmdeploy')


class KernelProvider:
    """Builds the kernels layer builders require and has each kernel family
    register them with its own C++ kernel registry.

    A family module provides ``requested()``, ``unsupported_reason(spec,
    arch)``, ``available()``, ``ensure(spec) -> {symbol: path}`` and
    ``register(paths)``.

    Args:
        state_dtype: resolved recurrent-state dtype of the engine.
        arch: compute capability of the target GPUs, ``major * 10 + minor``.
    """

    def __init__(self, state_dtype: torch.dtype, arch: int):
        self.state_dtype = state_dtype
        self.arch = arch
        self._resolved = {}

    def require(self, spec) -> dict[str, str] | None:
        """``symbol -> path`` of the kernels for ``spec``, now registered with
        C++, or None when they are not requested or cannot be provided (the
        layer then runs native).

        Raises if the kernels were requested and are buildable but cannot be provided. Memoized per spec, so the many
        layers of a model cost one lookup.
        """
        if spec not in self._resolved:
            self._resolved[spec] = self._resolve(spec)
        return self._resolved[spec]

    def _resolve(self, spec) -> dict[str, str] | None:
        family = spec.family
        if not family.requested():
            return None
        if reason := family.unsupported_reason(spec, self.arch):
            logger.info(f'{family.NAME}: not building kernels for {spec}: {reason}; the layer runs native')
            return None
        if not family.available():
            logger.warning(f'{family.NAME}: requested, but this TurboMind build has no TileLang support '
                           '(ENABLE_TILELANG=OFF); the layer runs native')
            return None
        paths = {symbol: str(path) for symbol, path in family.ensure(spec).items()}
        runtime.load_runtime()
        family.register(paths)
        return paths
