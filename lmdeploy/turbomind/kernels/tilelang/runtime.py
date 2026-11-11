# Copyright (c) OpenMMLab. All rights reserved.
"""Process-wide loading of the TileLang runtime for C++ callers.

Exported kernels need ``libtvm_runtime.so`` (module loader) and ``libtilelang.so``
(TMA descriptor creation, ``__tvm_tensormap_create_tiled``) loaded with
``RTLD_GLOBAL``. ``import tilelang`` is not relied on: it loads ``libtilelang.so``
without ``RTLD_GLOBAL`` and skips it under ``SKIP_LOADING_TILELANG_SO``.
"""
import ctypes
import threading
from importlib.util import find_spec
from pathlib import Path

_LIBRARIES = ('libtvm_runtime.so', 'libtilelang.so')
_lock = threading.Lock()
_handles: dict[Path, ctypes.CDLL] = {}


def tilelang_lib_dir() -> Path:
    """Native library directory of the installed TileLang package, found
    without importing it."""
    spec = find_spec('tilelang')
    if spec is None or not spec.submodule_search_locations:
        raise ImportError('TileLang is not installed')
    return Path(next(iter(spec.submodule_search_locations))) / 'lib'


def load_runtime(lib_dir: str | Path | None = None) -> None:
    """Load the TileLang runtime libraries with ``RTLD_GLOBAL``.

    Idempotent.
    """
    directory = Path(lib_dir) if lib_dir is not None else tilelang_lib_dir()
    with _lock:
        for name in _LIBRARIES:
            path = (directory / name).resolve()
            if path in _handles:
                continue
            if not path.is_file():
                raise FileNotFoundError(f'TileLang runtime library not found: {path}')
            _handles[path] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
