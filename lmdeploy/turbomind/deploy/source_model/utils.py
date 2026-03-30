# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for source model input classes."""
from __future__ import annotations

import math

from lmdeploy.archs import get_model_arch

from ..config import RopeParam


def load_model_config(model_path: str) -> dict:
    """Load and normalise the HuggingFace model config to a plain dict.

    Handles nested configs (text_config, llm_config) and transformers
    AutoConfig objects that expose a to_dict() method.
    """
    _, model_config = get_model_arch(model_path)
    if hasattr(model_config, 'text_config'):
        model_config = model_config.text_config
    elif hasattr(model_config, 'llm_config'):
        model_config = model_config.llm_config
    if hasattr(model_config, 'to_dict'):
        return model_config.to_dict()
    return model_config


def parse_rope_param(cfg: dict, head_dim: int) -> tuple[RopeParam, int]:
    """Parse RoPE configuration from a model config dict.

    Returns:
        rope_param: populated RopeParam instance
        max_position_embeddings: int (0 if not present in config)
    """
    if 'rope_parameters' in cfg:
        # transformers v5.0.0 aggregates rope settings into rope_parameters
        rope_scaling = cfg['rope_parameters']
        rope_theta = float(rope_scaling.get('rope_theta', 10000.0))
    else:
        rope_theta = float(cfg.get('rope_theta', 10000.0))
        rope_scaling = cfg.get('rope_scaling', None)

    max_position_embeddings = int(cfg.get('max_position_embeddings', 0))
    rope_param = RopeParam(type='default', base=rope_theta, dim=head_dim)

    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get('rope_type', '') or rope_scaling.get('type', '')
        if rope_scaling.get('mrope_section') is not None:
            rope_type = 'mrope'
        scaling_factor = rope_scaling.get('factor', 0.0)

        if rope_type == 'default':
            pass
        elif rope_type == 'dynamic':
            rope_param.type = 'dynamic'
            rope_param.factor = scaling_factor
            rope_param.max_position_embeddings = max_position_embeddings
        elif rope_type == 'linear':
            rope_param.type = 'linear'
            rope_param.factor = scaling_factor
        elif rope_type == 'llama3':
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings', 0)
            rope_param.type = 'llama3'
            rope_param.factor = scaling_factor
            rope_param.low_freq_factor = low_freq_factor
            rope_param.high_freq_factor = high_freq_factor
            rope_param.original_max_position_embeddings = original_max_position_embeddings
        elif rope_type == 'yarn':
            attention_factor = rope_scaling.get('attention_factor', None)
            if attention_factor is None:
                attention_factor = 0.1 * math.log(scaling_factor) + 1.0
            beta_fast = rope_scaling.get('beta_fast', 32.0)
            beta_slow = rope_scaling.get('beta_slow', 1.0)
            rope_param.type = 'yarn'
            if 'original_max_position_embeddings' in rope_scaling:
                original_max_position_embeddings = rope_scaling['original_max_position_embeddings']
                scaling_factor = max_position_embeddings / original_max_position_embeddings
            else:
                original_max_position_embeddings = max_position_embeddings
            rope_param.factor = scaling_factor
            rope_param.max_position_embeddings = original_max_position_embeddings
            rope_param.attention_factor = attention_factor
            rope_param.beta_fast = beta_fast
            rope_param.beta_slow = beta_slow
        elif rope_type == 'mrope':
            mrope_section = rope_scaling.get('mrope_section')
            rope_param.type = 'mrope'
            rope_param.mrope_section = mrope_section
        else:
            raise RuntimeError(f'Unsupported rope type: {rope_type}')

    return rope_param, max_position_embeddings


def get_yarn_params(rope_scaling: dict) -> tuple[float, float]:
    """Compute DeepSeek2/MLA YaRN attention scale factors.

    Returns:
        attention_factor: mscale ratio used for attention scaling
        softmax_scale: pre-softmax scale (non-zero only when mscale_all_dim > 0)
    """
    scaling_factor = float(rope_scaling['factor'])
    mscale = rope_scaling['mscale']
    mscale_all_dim = rope_scaling['mscale_all_dim']

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    attention_factor = float(
        yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim))

    softmax_scale = 0.0
    if mscale_all_dim:
        scale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        softmax_scale = scale * scale

    return attention_factor, softmax_scale
