# Copyright (c) OpenMMLab. All rights reserved.

from abc import ABC
from collections.abc import Sequence

from mmengine import Registry

from ..config import (AttentionConfig, LoraConfig, ModelConfig, TurbomindModelConfig,
                      config_from_dict, config_to_dict)
from ..source_model.base import BaseInputModel

OUTPUT_MODELS = Registry('target model', locations=['lmdeploy.turbomind.deploy.target_model.base'])


def _pad_inter_size(inter_size: int, group_size: int, tp: int):
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


class BaseOutputModel(ABC):
    """Base output model."""

    @classmethod
    def finalize_config(cls, input_model, cfg):
        """Finalize cfg by merging input model info. Mutates cfg in-place.

        Returns repeat_kv (int).
        """
        mc = cfg.model_config
        attn_tp = mc.attn_tp_size
        mlp_tp = mc.mlp_tp_size

        # Get and normalize input model info
        info = input_model.model_info()
        num_layer = int(info['num_layer'])
        for k in ['inter_size', 'expert_num']:
            v = info.get(k)
            if v is not None and not isinstance(v, Sequence):
                info[k] = [v] * num_layer

        # Merge into model_config
        final_cfg = config_to_dict(mc)
        final_cfg.update(info)
        if 'embedding_size' not in info:
            final_cfg['embedding_size'] = info['vocab_size']
        cfg.model_config = config_from_dict(ModelConfig, final_cfg)
        mc = cfg.model_config

        # Pad inter_size / expert_inter_size
        for i, v in enumerate(mc.inter_size):
            mc.inter_size[i] = _pad_inter_size(v, mc.group_size, mlp_tp)
        if mc.expert_num:
            mc.expert_inter_size = _pad_inter_size(
                mc.expert_inter_size, mc.group_size, mlp_tp)

        # Handle repeat_kv
        assert mc.head_num % attn_tp == 0
        repeat_kv = 0
        if attn_tp > mc.kv_head_num and attn_tp % mc.kv_head_num == 0:
            repeat_kv = attn_tp // mc.kv_head_num
            mc.kv_head_num = attn_tp
        mc.verify()
        assert mc.kv_head_num % attn_tp == 0

        # Merge into attention_config and lora_config
        for config_attr in ('attention_config', 'lora_config'):
            orig = getattr(cfg, config_attr)
            cls_type = type(orig)
            merged = config_to_dict(orig)
            merged.update(info)
            setattr(cfg, config_attr, config_from_dict(cls_type, merged))

        return repeat_kv

    def __init__(self, input_model, cfg, model_cls,
                 model_comm, gpu_count, *, repeat_kv):
        super().__init__()
        self.input_model = input_model
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.attn_tp_size = cfg.model_config.attn_tp_size
        self.attn_cp_size = cfg.model_config.attn_cp_size
        self.mlp_tp_size = cfg.model_config.mlp_tp_size
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.repeat_kv = repeat_kv

        self.model = model_cls(self)

    def root(self, index: int):
        """Return the C++ ``Module`` root for GPU *index*."""
        return self.model_comm.root(index)

    def tp_ranks(self, index: int):
        """Return ``(attn_tp_rank, mlp_tp_rank)`` for GPU *index*."""
        return (self.model_comm.attn_tp_rank(index),
                self.model_comm.mlp_tp_rank(index))

    def export(self) -> None:
        """Export to turbomind model format."""
        num_layer = self.model_config.num_layer
        from tqdm import tqdm
        pbar = tqdm(total=num_layer, desc='Convert to turbomind format', leave=False)
        for i, reader in self.input_model.readers():
            if self.model(i, reader):
                pbar.update(1)
        pbar.close()

    def export_iter(self):
        for i, reader in self.input_model.readers():
            self.model(i, reader)
            yield i

    @property
    def tm_config(self):
        return TurbomindModelConfig(model_config=self.model_config,
                                    attention_config=self.attention_config,
                                    lora_config=self.lora_config)
