# Copyright (c) OpenMMLab. All rights reserved.
"""Distributor: distributes module creation and weight commits across all GPUs."""
from __future__ import annotations

from .load_context import commit_linear, commit_tensor


class _noop:
    """No-op context manager for when no context guard is available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class Distributor:
    """Wraps N GPU handles for a single logical module.

    Distributes create_child / commit_linear / commit_tensor
    across all GPUs with bound TP configuration.
    """

    def __init__(self, handles, contexts=None, tp=1, ranks=None):
        self._handles = handles
        self._contexts = contexts or [None] * len(handles)
        self._tp = tp
        self._ranks = ranks

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx):
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    def create_child(self, name, config, tp=None, ranks=None):
        """Create a typed module child on ALL GPUs.

        Calls ``config.for_rank(rank).to_cpp()`` per GPU.
        Returns a new Distributor scoped to the created children,
        with tp/ranks rebound if provided (otherwise inherited).
        """
        new_tp = tp if tp is not None else self._tp
        new_ranks = ranks if ranks is not None else self._ranks
        children = []
        for i, handle in enumerate(self._handles):
            with self._contexts[i] or _noop():
                rank = new_ranks[i] if new_ranks and new_tp > 1 else 0
                child = handle.create_child(name, config.for_rank(rank).to_cpp())
                children.append(child)
        return Distributor(children, self._contexts, tp=new_tp, ranks=new_ranks)

    def commit_linear(self, name, linear, split_side=None, model_dtype=None):
        """Commit a Linear bundle to all GPUs.

        If split_side is given, uses bound tp/ranks for sharding.
        If split_side is None, broadcasts (tp=1).
        """
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            with self._contexts[i] or _noop():
                rank = self._rank_for(i) if tp > 1 else 0
                commit_linear(handle, linear, name,
                              split_side=split_side, split_num=tp,
                              rank=rank, model_dtype=model_dtype)

    def commit_tensor(self, name, tensor, split_side=None):
        """Commit a raw tensor to all GPUs."""
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            with self._contexts[i] or _noop():
                rank = self._rank_for(i) if tp > 1 else 0
                commit_tensor(handle, tensor, name,
                              split_side=split_side, split_num=tp,
                              rank=rank)
