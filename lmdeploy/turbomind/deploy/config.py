# Copyright (c) OpenMMLab. All rights reserved.
import json
from dataclasses import asdict

# use pydantic.dataclasses.dataclass to check data type
from pydantic.dataclasses import dataclass


def config_to_dict(config):
    """Export config to a dict."""
    assert isinstance(config, (AttentionConfig, LoraConfig)), \
        f'A dataclass is expected, but got {type(config)}'

    return asdict(config)


@dataclass
class RopeParam:
    type: str
    base: float
    dim: int
    factor: float = 1.0
    max_position_embeddings: int = None
    attention_factor: float = 1.0
    beta_fast: float = 32
    beta_slow: float = 1
    low_freq_factor: float = None
    high_freq_factor: float = None
    original_max_position_embeddings: int = None
    mrope_section: list[int] = None


@dataclass
class AttentionConfig:
    softmax_scale: float = 0
    cache_block_seq_len: int = 64
    use_logn_attn: int = 0
    max_position_embeddings: int = 0
    rope_param: RopeParam = None


@dataclass
class LoraConfig:
    lora_policy: str = ''
    lora_r: int = 0
    lora_scale: float = 0.0
    lora_max_wo_r: int = 0
    lora_rank_pattern: str = ''
    lora_scale_pattern: str = ''


@dataclass
class TurbomindModelConfig:
    """Config for turbomind model."""
    attention_config: AttentionConfig = None
    lora_config: LoraConfig = None
    model_arch: str = ''
    chat_template: str = ''
    model_name: str = ''
    data_type: str = ''
    model_format: str = 'hf'
    session_len: int = 0
    group_size: int = 0
    attn_tp_size: int = 1
    attn_cp_size: int = 1
    mlp_tp_size: int = 1

    def to_dict(self):
        """Export to a dict."""
        result = {}
        if self.attention_config is not None:
            result['attention_config'] = config_to_dict(self.attention_config)
        if self.lora_config is not None:
            result['lora_config'] = config_to_dict(self.lora_config)
        return result

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)
