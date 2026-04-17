# Copyright (c) OpenMMLab. All rights reserved.
import torch

import _turbomind as _tm

from ._base import Builder, _torch_dtype_to_cpp, _cast_shard_for_tm


def make_norm_config(*, dim, data_type):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    return cfg


class NormBuilder(Builder):
    """Builder for a single norm weight module."""

    def set_weight(self, tensor: torch.Tensor | None):
        """Commit the norm weight tensor to all GPU handles."""
        self._ensure_handles()
        if tensor is None:
            return
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                shard = tensor
                if not shard.is_cuda:
                    shard = shard.cuda(0).contiguous()
                elif not shard.is_contiguous():
                    shard = shard.contiguous()
                cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
                dst = handle.param('weight').alloc(list(shard.shape), cpp_dtype)
                shard = _cast_shard_for_tm(shard, dst)
                dst.copy_from(shard)
