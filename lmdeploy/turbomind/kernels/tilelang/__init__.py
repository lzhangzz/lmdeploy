# Copyright (c) OpenMMLab. All rights reserved.
"""Exported TileLang kernels for TurboMind.

A layer builder states the kernel spec it needs and asks the build context's
:class:`~.provider.KernelProvider` for it. The provider compiles missing kernels
into a content-addressed cache (:mod:`.cache`), loads the TileLang runtime
(:mod:`.runtime`), and has the kernel family register each module with the
family's own C++ registry (a ``tilelang::Library`` instance,
``src/turbomind/kernels/tilelang/library.h``).

Nothing in this package imports TileLang unless a kernel is actually requested.
"""
