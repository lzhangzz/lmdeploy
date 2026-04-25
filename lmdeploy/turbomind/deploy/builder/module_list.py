# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm

from ._base import Builder, BuiltModule

ModuleListConfig = _tm.ModuleListConfig


class ModuleListBuilder(Builder):
    """Builder for ModuleList containers."""

    def __setitem__(self, index: int, value):
        if self._built:
            raise RuntimeError(
                f"{type(self).__name__} is built; cannot set index {index}")
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}[{index}]: call .build() first")
        assert isinstance(value, BuiltModule), (
            f"{type(self).__name__}[{index}] requires a BuiltModule")
        self._pending_children[str(index)] = value.handles
