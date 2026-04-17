# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm

from ._base import Builder, SplitSide


def make_linear_config(*, input_dim, output_dim, data_type, has_bias=False):
    cfg = _tm.LinearConfig()
    cfg.input_dim = input_dim
    cfg.output_dim = output_dim
    cfg.data_type = data_type
    cfg.has_bias = has_bias
    return cfg


class LinearBuilder(Builder):
    """Builder for standalone linear layers (embeddings, lm_head).

    Wraps a C++ LinearWeight module. Use ``set_weight()`` to commit
    the weight tensor.
    """

    def set_weight(self, tensor, split_side=None):
        """Commit the weight tensor to all GPU handles."""
        self._commit_tensor('weight', tensor, split_side)
