# Copyright (c) OpenMMLab. All rights reserved.
"""BaseOutputModel — drives the spec through TextModelLoader + export."""
from __future__ import annotations

from abc import ABC

from mmengine import Registry

from ..config import (AttentionConfig, LoraConfig, ModelConfig,
                      TurbomindModelConfig)

OUTPUT_MODELS = Registry('target model',
                         locations=['lmdeploy.turbomind.deploy.target_model.base'])


class BaseOutputModel(ABC):
    """Base output model. Drives a TextModelSpec through loading + commit."""

    @classmethod
    def finalize_config(cls, spec, cfg: TurbomindModelConfig):
        """Assemble the YAML wire-format config from spec + pre-seeded fields.

        The spec has already been constructed with a resolved engine_config
        (dtype, model_format, session_len, tp sizes) plus group_size. The
        only fields that the converter set directly onto ``cfg`` without
        a corresponding engine_config field are ``model_arch``,
        ``chat_template``, and ``model_name`` (pure metadata).

        We generate ``produced`` from the spec, copy those three metadata
        fields from ``cfg`` onto it, then install ``produced`` back onto
        ``cfg``.
        """
        produced = spec.to_legacy_config()
        preserved = ('model_arch', 'chat_template', 'model_name')
        for name in preserved:
            val = getattr(cfg.model_config, name, None)
            if val not in (None, '', 0):
                setattr(produced.model_config, name, val)
        produced.model_config.verify()
        cfg.model_config     = produced.model_config
        cfg.attention_config = produced.attention_config
        cfg.lora_config      = produced.lora_config

    def __init__(self, spec, cfg, model_comm, gpu_count, model_path):
        from ..text_model_loader import TextModelLoader
        self.spec = spec
        self.tm_config = cfg
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.attn_tp_size = cfg.model_config.attn_tp_size
        self.attn_cp_size = cfg.model_config.attn_cp_size
        self.mlp_tp_size = cfg.model_config.mlp_tp_size
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        # model_path is writable by update_params (Queue takes over).
        self.model_path = model_path

        # Bind runtime handles onto the spec. TextModelLoader pulls
        # contexts/root_handles/ranks from model_comm.
        self.model = TextModelLoader(self)

    # ------------------------------------------------------------------
    # GPU-topology helpers (used by TextModelLoader)
    # ------------------------------------------------------------------

    def root(self, index: int):
        return self.model_comm.root(index)

    def context(self, index: int):
        return self.model_comm.context(index)

    def tp_ranks(self, index: int):
        return (self.model_comm.attn_tp_rank(index),
                self.model_comm.mlp_tp_rank(index))

    # ------------------------------------------------------------------
    # Export drivers
    # ------------------------------------------------------------------

    def export(self) -> None:
        from tqdm import tqdm
        import torch
        from ..loader import create_loader
        pbar = tqdm(total=1, desc='Convert to turbomind format', leave=False)
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()
        pbar.update(1)
        pbar.close()

    def export_iter(self):
        import torch
        from ..loader import create_loader
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        yield -1
        # Runs on StopIteration; preserves old readers() behavior.
        torch.cuda.empty_cache()
