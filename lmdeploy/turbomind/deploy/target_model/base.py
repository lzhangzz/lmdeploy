# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from abc import ABC
from collections.abc import Sequence

import torch
import tqdm
import yaml
from mmengine import Registry

from ..config import AttentionConfig, LoraConfig, ModelConfig, TurbomindModelConfig, config_from_dict, config_to_dict
from ..source_model.base import BaseInputModel

OUTPUT_MODELS = Registry('target model', locations=['lmdeploy.turbomind.deploy.target_model.base'])


def tprint(*args, **kwargs):
    to_file = kwargs.pop('to_file', False)
    if not to_file:
        return
    from io import StringIO
    s = StringIO()
    print(*args, **kwargs, file=s, end='')
    tqdm.tqdm.write(s.getvalue())


def _compute_dtype(data_type: str) -> torch.dtype:
    """Map the model's compute data_type string to a torch dtype."""
    return torch.bfloat16 if data_type == 'bfloat16' else torch.float16


def _pad_inter_size(inter_size: int, group_size: int, tp: int):
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


class BaseOutputModel(ABC):
    """Base output model."""

    def __init__(self, input_model: BaseInputModel, cfg: TurbomindModelConfig, model_cls, out_dir: str = ''):
        super().__init__()
        self.input_model = input_model
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.attn_tp_size = self.model_config.attn_tp_size
        self.attn_cp_size = self.model_config.attn_cp_size
        self.mlp_tp_size = self.model_config.mlp_tp_size
        self.out_dir = out_dir
        self.to_file = True if out_dir else False
        self.model_comm = None
        self.gpu_count = 0

        # get `model_info` at first, which will be updated to `self.model_config` and `self.attention_config`
        self.input_model_info = self.input_model.model_info()
        self.input_model_info = self.single_to_list(self.input_model_info, keys=['inter_size', 'expert_num'])
        self.permute_qk = self.input_model_info.get('permute_qk', True)
        self.update_model_config()
        for i, v in enumerate(self.model_config.inter_size):
            self.model_config.inter_size[i] = _pad_inter_size(v, self.model_config.group_size, self.mlp_tp_size)
        if self.model_config.expert_num:
            self.model_config.expert_inter_size = _pad_inter_size(self.model_config.expert_inter_size,
                                                                  self.model_config.group_size, self.mlp_tp_size)

        # head_num is divisble by tp but kv_head_num is not
        # and tp is divisble by kv_head_num
        assert self.model_config.head_num % self.attn_tp_size == 0
        self.repeat_kv = 0
        if (self.attn_tp_size > self.model_config.kv_head_num
                and self.attn_tp_size % self.model_config.kv_head_num == 0):
            self.repeat_kv = (self.attn_tp_size // self.model_config.kv_head_num)
            self.model_config.kv_head_num = self.attn_tp_size

        self.model_config.verify()
        assert self.model_config.kv_head_num % self.attn_tp_size == 0

        # print(self.model_config)

        self.update_attention_config()
        self.update_lora_config()
        # ! Dependency on `self`
        self.model = model_cls(self)

    def single_to_list(self, config: dict, keys):
        num_layer = int(config['num_layer'])
        for k in keys:
            v = config.get(k, None)
            if v is not None and not isinstance(v, Sequence):
                config[k] = [v] * num_layer
        return config

    def update_model_config(self):
        """Update `self.model_config` according to the input_model's
        `model_info`"""
        final_cfg = config_to_dict(self.model_config)
        final_cfg.update(self.input_model_info)
        if 'embedding_size' not in self.input_model_info.keys():
            final_cfg.update(embedding_size=self.input_model_info['vocab_size'])

        self.model_config = config_from_dict(ModelConfig, final_cfg)

    def update_attention_config(self):
        """Update attention config according to input model's model info."""
        final_cfg = config_to_dict(self.attention_config)
        final_cfg.update(self.input_model_info)
        self.attention_config = config_from_dict(AttentionConfig, final_cfg)

    def update_lora_config(self):
        """Update lora config according to input model's model info."""
        final_cfg = config_to_dict(self.lora_config)
        final_cfg.update(self.input_model_info)
        self.lora_config = config_from_dict(LoraConfig, final_cfg)

    def export_config(self) -> None:
        """Export turbomind config."""
        if self.to_file:
            config_path = osp.join(self.out_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.safe_dump(self.tm_config.to_dict(), f)

    def allocate_weight(self, param_name: str, cpp_dtype, group_size: int) -> None:
        """Allocate a linear weight on the C++ side via allocate_weight."""
        if self.model_comm is None:
            return
        for i in range(self.gpu_count):
            try:
                self.model_comm.allocate_weight(i, param_name, cpp_dtype, group_size)
            except RuntimeError:
                pass

    def export_weight(self, param: torch.Tensor, name: str) -> None:
        """Export turbomind weight."""

        _CASTABLE = {torch.float32, torch.float16, torch.bfloat16}

        def _cast(tensor: torch.Tensor) -> torch.Tensor:
            """Cast standard float types to the model's compute dtype.

            Non-float types (int32, uint8) and special float types (float8)
            are left unchanged — their dtype is determined by the checkpoint
            format and handled by the C++ side.
            """
            if tensor.dtype in _CASTABLE:
                return tensor.to(_compute_dtype(self.model_config.data_type))
            return tensor

        def _tofile(tensor, path):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.view(torch.half)
            tensor.contiguous().cpu().numpy().tofile(path)

        if self.to_file:
            param = _cast(param)
            tprint(name, param.shape)
            _tofile(param, osp.join(self.out_dir, name))
        elif self.model_comm is not None:
            try:
                import _turbomind as _tm
            except ImportError:
                _tm = None

            torch_tensor = param if param.is_contiguous() else param.contiguous()
            torch_tensor = torch_tensor.cuda()
            torch_tensor = _cast(torch_tensor)

            def _reconcile_dtype(tm_tensor, src_tensor):
                """Match Python tensor dtype to C++ tensor dtype."""
                if _tm is not None:
                    if tm_tensor.type == _tm.DataType.TYPE_FP32 and src_tensor.dtype in [
                            torch.float16, torch.bfloat16
                    ]:
                        return src_tensor.float()
                    elif tm_tensor.type == _tm.DataType.TYPE_FP16 and src_tensor.dtype == torch.float32:
                        return src_tensor.half()
                return src_tensor

            for i in range(self.gpu_count):
                try:
                    tm_tensor = self.model_comm.get_parameter(i, name)
                    tm_tensor.copy_from(_reconcile_dtype(tm_tensor, torch_tensor))
                except RuntimeError:
                    continue
        else:
            tprint('skip export', name, param.shape)

    def save_split(self, tensor: torch.Tensor, name: str, split_dim=None, split_num=1, copy=False) -> None:
        """Save split.

        - 2D input
            shape must be (input_dims, output_dims)
        - 1D input (bias)
            shape must be (output_dims)
            split is skipped when split_dim == 0
        """

        if copy or (tensor.dim() == 1 and split_dim == 0):
            split_dim = None
            copy = True

        if split_dim is not None:
            tprint(f'*** splitting {name}, shape={tensor.shape}, '
                   f'split_dim={split_dim}, split_num={split_num}',
                   to_file=self.to_file)
            if tensor.shape[split_dim] % split_num != 0:
                raise RuntimeError(f'{name}: shape={list(tensor.shape)}, split_num={split_num}')
            split_size = tensor.shape[split_dim] // split_num
            splits = torch.split(tensor, split_size, dim=split_dim)
            for i, split in enumerate(splits):
                prefix, ext = osp.splitext(name)
                self.export_weight(split, f'{prefix}.{i}{ext}')
        elif copy:
            tprint(f'### copying {name}, shape={tensor.shape}', to_file=self.to_file)
            copies = [tensor] * split_num
            for i, copy in enumerate(copies):
                prefix, ext = osp.splitext(name)
                self.export_weight(copy, f'{prefix}.{i}{ext}')
        else:
            self.export_weight(tensor, name)

    def export(self) -> None:
        """Export to turbomind model format."""
        num_layer = self.model_config.num_layer
        from tqdm import tqdm
        pbar = tqdm(total=num_layer, desc='Convert to turbomind format', leave=self.to_file)
        self.export_config()
        for i, reader in self.input_model.readers():
            if self.model(i, reader):
                pbar.update(1)
        pbar.close()

    def export_iter(self):
        self.export_config()
        for i, reader in self.input_model.readers():
            self.model(i, reader)
            yield i

    @property
    def tm_config(self):
        return TurbomindModelConfig(model_config=self.model_config,
                                    attention_config=self.attention_config,
                                    lora_config=self.lora_config)
