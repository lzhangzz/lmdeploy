# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for source model input classes."""
from __future__ import annotations

import math

import torch

from lmdeploy.archs import get_model_arch

from ..config import RopeParam
from ..kind_map import TRIVIAL_FORMAT


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


_ROPE_TYPE_MAP = {
    'default': 1,
    'linear': 2,
    'dynamic': 3,
    'yarn': 4,
    'llama3': 5,
    'mrope': 6,
}


def rope_type_to_int(type_str: str) -> int:
    return _ROPE_TYPE_MAP[type_str]


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


def reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Reorder rotary embedding layout for TurboMind's RoPE kernel."""
    if rope_dim < head_dim:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), head_num, head_dim)
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
        return x.reshape(orig_shape)
    else:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        return x.view(-1, head_num, 2, head_dim // 2).transpose(2, 3).reshape(x.shape)


def _dequant_linear(linear) -> 'Linear':
    """Dequantize a quantized Linear to trivial when the format provides dequant.

    Local copy to avoid circular import from builder/_base.py.
    """
    fmt = linear.weight_format
    if fmt is None or fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors)
    from ..linear import Linear
    return Linear(tensors=new_tensors, weight_format=TRIVIAL_FORMAT, data_format=None)


def reorder_rotary_emb_linear(linear, head_dim: int, rope_dim: int):
    """Apply RoPE permutation to all tensors in a Linear.

    Quantization-aware:
    - If quantized and block_out % head_dim != 0, dequantizes first
      (permuting within a head would cross block boundaries).
    - For weight/bias: element-level RoPE permutation.
    - For scales/zeros when block_out % head_dim == 0: block-level channel
      shuffling. Each head maps to (block_out / head_dim) complete blocks,
      so we apply the same interleave pattern at block granularity.
    - For scales/zeros when dequantized: skipped (trivial format has none).
    """
    from ..linear import Linear

    wfmt = linear.weight_format
    block_out = (wfmt.block_out or 0) if wfmt is not None else 0

    # If blocks don't align with heads, dequant first
    if block_out and block_out % head_dim != 0:
        linear = _dequant_linear(linear)
        block_out = 0

    new_tensors = {}
    for kind, tensor in linear.tensors.items():
        if kind in ("scales", "zeros") and block_out > 0:
            # Block-level shuffle: apply RoPE at block granularity.
            # scales/zeros have shape [in_blocks, n_heads * blocks_per_head].
            # reorder_rotary_emb handles this: head_num = last_dim // blocks_per_head.
            blocks_per_head = block_out // head_dim
            rope_dim_blocks = rope_dim * blocks_per_head // head_dim
            new_tensors[kind] = reorder_rotary_emb(tensor, blocks_per_head, rope_dim_blocks)
        elif tensor.size(-1) % head_dim == 0:
            new_tensors[kind] = reorder_rotary_emb(tensor, head_dim, rope_dim)
        else:
            new_tensors[kind] = tensor

    return Linear(tensors=new_tensors, weight_format=linear.weight_format,
                  data_format=linear.data_format)


# --- TP padding helpers (moved from target_model/base.py) -----------------

def _pad_inter_size(inter_size: int, group_size: int, tp: int) -> int:
    """Pad inter_size so it is divisible by group_size * tp.

    Moved from target_model/base.py where it lived as a module-level helper
    inside finalize_config. Same formula.
    """
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


def _pad_kv_head(kv_head_num: int, attn_tp: int) -> int:
    """Pad kv_head_num up to attn_tp when attn_tp is a multiple of kv_head_num.

    Matches the rule in finalize_config:
      if attn_tp > kv_head_num and attn_tp % kv_head_num == 0:
          kv_head_num = attn_tp
    """
    if attn_tp > kv_head_num and attn_tp % kv_head_num == 0:
        return attn_tp
    return kv_head_num


# --- Layer-prefix detection ------------------------------------------------

def detect_layer_prefix(params: dict | None, cfg: dict) -> tuple[str, str, str]:
    """Return (layer_prefix, embed_key, norm_key) for a HF checkpoint.

    Models that wrap the decoder in a ``language_model`` submodule (Molmo,
    some multimodal variants, Qwen3.5 when packaged as a multimodal root)
    store weights under ``model.language_model.*``. Plain decoder models
    use ``model.*``.

    If ``params`` is None (spec hasn't loaded weights yet), fall back to the
    standard ``model.*`` layout. Specs that need early disambiguation can
    override this during their own parsing.
    """
    if params is not None and any(
            k.startswith('model.language_model.') for k in params):
        return ('model.language_model.layers',
                'model.language_model.embed_tokens.weight',
                'model.language_model.norm.weight')
    return ('model.layers',
            'model.embed_tokens.weight',
            'model.norm.weight')
