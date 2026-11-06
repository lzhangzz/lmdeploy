# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def config_from_pretrained(pretrained_model_name_or_path: str, **kwargs):
    try:
        return AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except Exception as error:
        logger.debug(f'AutoConfig.from_pretrained failed: {error}, falling back to PretrainedConfig.')

    # Some models neither provide an auto_map nor have a native Transformers
    # config; fall back to the generic config.
    from transformers import PretrainedConfig
    return PretrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
