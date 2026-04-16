# Copyright (c) OpenMMLab. All rights reserved.
from ._base import Builder, SplitSide


class LinearBuilder(Builder):
    """Builder for standalone linear layers (embeddings, lm_head).

    Wraps a C++ LinearWeight module. Use ``set_weight()`` to commit
    the weight tensor.
    """

    def set_weight(self, tensor, split_side=None):
        """Commit the weight tensor to all GPU handles."""
        self._commit_tensor('weight', tensor, split_side)
