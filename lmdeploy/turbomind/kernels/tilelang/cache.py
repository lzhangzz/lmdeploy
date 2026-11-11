# Copyright (c) OpenMMLab. All rights reserved.
"""Content-addressed on-disk cache of exported TileLang kernels.

Layout: ``<root>/<family>-<toolchain hash>/<symbol>.so``. The hash covers the
family's toolchain description (target arch, package versions, host compiler)
and the bytes of the family's source files, so editing a kernel factory or
changing the toolchain starts a new directory. Specialization parameters live in
the symbol names, so every model built with the same toolchain shares one
directory.

Publishing is per file (write to a temp dir, then ``os.replace``), so a partially
written ``.so`` is never visible; concurrent builders serialize on the
directory's lock file.
"""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from pathlib import Path

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def cache_root() -> Path:
    """``$LMDEPLOY_KERNEL_CACHE``, else ``$XDG_CACHE_HOME/lmdeploy/tilelang``,
    else ``~/.cache/lmdeploy/tilelang``."""
    if root := os.environ.get('LMDEPLOY_KERNEL_CACHE'):
        return Path(root)
    base = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(base) / 'lmdeploy' / 'tilelang'


def host_compiler() -> str:
    """NVCC host compiler: ``$CXX``, else ``c++``, resolved to an absolute
    path."""
    name = os.environ.get('CXX') or 'c++'
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f'Host compiler for TileLang kernels not found: {name} (set CXX)')
    return path


@cache
def compiler_identity(compiler: str) -> str:
    """First line of ``<compiler> --version``."""
    result = subprocess.run([compiler, '--version'], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0].strip() if result.stdout else ''


def cache_directory(family: str, toolchain: Mapping[str, str], sources: Iterable[Path]) -> Path:
    digest = hashlib.sha256(json.dumps(dict(toolchain), sort_keys=True).encode())
    for source in sources:
        digest.update(Path(source).name.encode() + b'\0')
        digest.update(Path(source).read_bytes())
    return cache_root() / f'{family}-{digest.hexdigest()[:16]}'


def ensure(directory: Path, kernels: Mapping[str, Callable[[Path], None]]) -> dict[str, Path]:
    """Make ``<directory>/<symbol>.so`` exist for every symbol in ``kernels``
    and return ``symbol -> path``.

    ``kernels[symbol](path)`` must write the exported module to ``path``. Only
    missing symbols are built.
    """
    directory = Path(directory)
    paths = {symbol: directory / f'{symbol}.so' for symbol in kernels}

    def missing():
        return [symbol for symbol, path in paths.items() if not path.is_file()]

    if not missing():
        return paths
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if todo := missing():
            logger.info(f'Building TileLang kernels (first use only) into {directory}: {", ".join(todo)}')
            with tempfile.TemporaryDirectory(dir=directory, prefix='.tmp-') as staging:
                for symbol in todo:
                    staged = Path(staging) / f'{symbol}.so'
                    kernels[symbol](staged)
                    os.replace(staged, paths[symbol])
    return paths
