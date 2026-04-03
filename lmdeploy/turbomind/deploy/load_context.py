# Copyright (c) OpenMMLab. All rights reserved.
"""Composable loading primitives for building the C++ module tree from Python."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .linear import Linear

if TYPE_CHECKING:
    from .config import ModelConfig


def _cpp_dtype(dtype_str: str) -> int:
    """Convert a model-config data_type string to C++ DataType enum value as int.

    Returns a plain int (not the enum object) so it can be passed in
    ModuleConfig dicts via pybind11 without requiring enum → int64_t cast.
    """
    import _turbomind as _tm
    return {
        'float32':  _tm.DataType.TYPE_FP32,
        'float16':  _tm.DataType.TYPE_FP16,
        'bfloat16': _tm.DataType.TYPE_BF16,
    }[dtype_str].value


def _act_type_id(act_str: str) -> int:
    """Convert activation_type string to C++ ActivationType enum value."""
    return {'silu': 0, 'gpt-oss': 1}.get(act_str, 0)


class LoadContext:
    """Wraps a C++ Module handle and provides composable loading primitives.

    Each LoadContext is rooted at one C++ Module.  The ``create`` method returns
    a new LoadContext rooted at the created child module.  ``load_linear`` and
    ``load_tensor`` handle the full create + commit lifecycle.
    """

    def __init__(self, handle, tp_config: dict,
                 model_config: 'ModelConfig | None' = None):
        """
        Args:
            handle: C++ Module handle (pybind11 object).
            tp_config: Dict with keys: tp_size, rank, head_dim,
                       rope_dim, permute_qk, repeat_kv, attn_output_gate,
                       kv_head_num.
            model_config: The Python ``ModelConfig`` for the model being loaded.
        """
        self._handle = handle
        self._tp_config = tp_config
        self._model_config = model_config

    @property
    def model_config(self) -> 'ModelConfig':
        assert self._model_config is not None, 'model_config not set'
        return self._model_config

    @property
    def cpp_dtype(self):
        """C++ DataType enum for the model's compute dtype."""
        return _cpp_dtype(self.model_config.data_type)

    @property
    def tp_size(self) -> int:
        return self._tp_config['tp_size']

    @property
    def rank(self) -> int:
        return self._tp_config['rank']

    @property
    def head_dim(self) -> int:
        return self._tp_config.get('head_dim', 0)

    @property
    def rope_dim(self) -> int:
        return self._tp_config.get('rope_dim', 0)

    @property
    def repeat_kv(self) -> int:
        return self._tp_config.get('repeat_kv', 0)

    @property
    def attn_output_gate(self) -> bool:
        return self._tp_config.get('attn_output_gate', False)

    @property
    def kv_head_num(self) -> int:
        return self._tp_config.get('kv_head_num', 0)

    def create(self, name: str, module_type: str, **config) -> 'LoadContext':
        """Create a child module via the C++ registry.

        Returns a new LoadContext rooted at the created module.
        """
        child = self._handle.create_child(name, module_type, config)
        return LoadContext(child, self._tp_config, self._model_config)

    def child(self, name: str) -> 'LoadContext':
        """Return a LoadContext for an existing child (no creation)."""
        handle = self._handle.get(name)
        return LoadContext(handle, self._tp_config, self._model_config)

    def load_linear(self, name: str, linear: Linear,
                    tp_rule: str | None = None):
        """Create a LinearWeight child and commit weight data.

        Handles TP splitting, quantization packing, and dtype casting.
        The child is created via create_child, then weights are committed
        directly to the child handle.
        """
        from .module import SplitSide as _SplitSide
        from .module import _infer_cpp_linear_dtype

        # Infer C++ dtype and group_size from the Linear
        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if group_size == 0:
            group_size = max(1, 128)

        # Extract dimensions from weight tensor shape (Linear doesn't store dims)
        weight = linear.tensors.get('weight') or linear.tensors.get('qweight')
        input_dim = weight.shape[0] if weight is not None else 0
        output_dim = weight.shape[-1] if weight is not None else 0

        # Create the LinearWeight child
        child_handle = self._handle.create_child(
            name, 'LinearWeight', {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'data_type': cpp_dtype,
                'has_bias': 'bias' in linear.tensors,
            })

        # Commit the weight data directly to the child handle.
        tp_side = _SplitSide[tp_rule] if tp_rule else None
        split_num = self.tp_size if tp_side else 1
        self._commit_linear_to_handle(child_handle, linear, cpp_dtype,
                                      group_size, tp_side, split_num,
                                      self.rank)

    @staticmethod
    def _commit_linear_to_handle(handle, linear: Linear, cpp_dtype, group_size,
                                 split_side, split_num: int, rank: int):
        """Commit Linear tensor data to a pre-created C++ module handle.

        This is the stripped-down version of ``commit_linear`` that works
        with a pre-created handle instead of using ``module.get(name)``.
        """
        from .module import _cast_shard_for_tm, _SPLIT_SIDE_TO_DIM, SplitSide

        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        packer = linear.weight_format.packer if linear.weight_format else None

        def _kind_order(item):
            k, _ = item
            return (0, k) if k in ('weight', 'qweight') else (1, k)

        for kind, tensor in sorted(linear.tensors.items(), key=_kind_order):
            if packer is not None:
                tensor = packer(tensor, kind)

            tensor_split_dim = split_dim
            if kind == 'bias' and split_side == SplitSide.INPUT:
                tensor_split_dim = None

            if tensor_split_dim is not None and split_num > 1:
                split_size = tensor.shape[tensor_split_dim] // split_num
                shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
            else:
                shard = tensor

            shard = shard.cuda().contiguous()
            dst = handle.alloc(kind, cpp_dtype, group_size)
            if dst:
                shard = _cast_shard_for_tm(shard, dst)
                if dst.byte_size != shard.nbytes and dst.byte_size > shard.nbytes:
                    pad_dim = tensor_split_dim if tensor_split_dim is not None else -1
                    if pad_dim < 0:
                        pad_dim = shard.dim() + pad_dim
                    outer = shard.numel() // shard.shape[pad_dim]
                    extra = (dst.byte_size - shard.nbytes) // (outer * shard.element_size())
                    new_shape = list(shard.shape)
                    new_shape[pad_dim] += extra
                    padded = torch.zeros(new_shape,
                                         dtype=shard.dtype,
                                         device=shard.device)
                    idx = [slice(None)] * shard.dim()
                    idx[pad_dim] = slice(0, shard.shape[pad_dim])
                    padded[tuple(idx)].copy_(shard)
                    shard = padded
                dst.copy_from(shard)

    def load_tensor(self, name: str, tensor: torch.Tensor,
                    module_type: str = 'NormWeight',
                    module_config: dict | None = None,
                    tp_rule: str | None = None):
        """Create a module child and commit tensor data.

        Args:
            name: Child module name.
            tensor: Weight tensor to commit.
            module_type: C++ module type to create (e.g., ``"NormWeight"``).
            module_config: Config dict for module creation.
            tp_rule: ``"output"`` or ``"input"`` for TP split, None for
                broadcast.
        """
        from .module import SplitSide as _SplitSide
        from .module import commit_tensor

        config = module_config or {}
        child_handle = self._handle.create_child(name, module_type, config)

        tp_side = _SplitSide[tp_rule] if tp_rule else None
        commit_tensor(child_handle, tensor, 'weight',
                             split_side=tp_side,
                             split_num=self.tp_size if tp_side else 1,
                             rank=self.rank)
