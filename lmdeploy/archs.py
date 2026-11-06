# Copyright (c) OpenMMLab. All rights reserved.
from .messages import TurbomindEngineConfig
from .utils import get_logger

logger = get_logger('lmdeploy')


def check_backend(backend: str | None) -> None:
    """Reject inference backends other than TurboMind."""
    if backend not in (None, 'turbomind'):
        raise ValueError(f'Unsupported backend {backend!r}. TurboMind is the only inference backend; '
                         'the PyTorch engine has been removed.')


def check_backend_config(backend_config: TurbomindEngineConfig | None) -> TurbomindEngineConfig:
    """Return a TurboMind engine config, rejecting other config types."""
    if backend_config is None:
        return TurbomindEngineConfig()
    if not isinstance(backend_config, TurbomindEngineConfig):
        raise TypeError(f'backend_config must be a TurbomindEngineConfig, got {type(backend_config).__name__}. '
                        'The PyTorch engine and PytorchEngineConfig have been removed.')
    return backend_config


def check_turbomind_support(model_path: str, trust_remote_code: bool = False) -> None:
    """Raise a clear error unless TurboMind can run the model.

    Args:
        model_path (str): the local path of a model.
        trust_remote_code (bool): whether to trust remote code from model repositories.
    """
    from lmdeploy import turbomind

    if not turbomind.is_available():
        raise RuntimeError('The TurboMind native extension is unavailable, so LMDeploy cannot run inference. '
                           'Install a CUDA build of LMDeploy or build it from source.') from turbomind._import_error

    from lmdeploy.turbomind.supported_models import is_supported
    if not is_supported(model_path, trust_remote_code=trust_remote_code):
        raise ValueError(f'{model_path!r} is not supported by the TurboMind engine. See the supported models list '
                         'in the LMDeploy documentation.')


def check_vl_llm(config: dict) -> bool:
    """Check if the model is a vl model from model config."""
    arch = config['architectures'][0]
    supported_archs = set([
        'InternVLChatModel', 'Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration',
        'Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration', 'Qwen3_5ForConditionalGeneration',
        'Qwen3_5MoeForConditionalGeneration', 'InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration',
        'InternS2MobiusForConditionalGeneration', 'InternS2MobiusForCausalLM',
        'InternS2PreviewForConditionalGeneration', 'InternS2PreviewForCausalLM',
    ])
    return arch in supported_archs


def get_task(model_path: str,
             trust_remote_code: bool = False,
             backend_config: TurbomindEngineConfig | None = None):
    """Get pipeline type and pipeline class from model config."""
    from lmdeploy.serve.core import AsyncEngine

    if backend_config and backend_config.language_model_only:
        return 'llm', AsyncEngine
    _, config = get_model_arch(model_path, trust_remote_code=trust_remote_code)
    if check_vl_llm(config.to_dict()):
        from lmdeploy.serve.core import VLAsyncEngine
        return 'vlm', VLAsyncEngine

    # default task, pipeline_class
    return 'llm', AsyncEngine


def get_model_arch(model_path: str, trust_remote_code: bool = False):
    """Get a model's architecture and configuration.

    Args:
        model_path(str): the model path
    """
    from lmdeploy.hf_configs import config_from_pretrained

    cfg = config_from_pretrained(model_path, trust_remote_code=trust_remote_code)

    _cfg = cfg.to_dict()
    if _cfg.get('architectures', None):
        arch = _cfg['architectures'][0]
    elif _cfg.get('auto_map', None) and 'AutoModelForCausalLM' in _cfg['auto_map']:
        arch = _cfg['auto_map']['AutoModelForCausalLM'].split('.')[-1]
    elif _cfg.get('language_config', None) and _cfg['language_config'].get(
            'auto_map', None) and 'AutoModelForCausalLM' in _cfg['language_config']['auto_map']:
        arch = _cfg['language_config']['auto_map']['AutoModelForCausalLM'].split('.')[-1]
    else:
        raise RuntimeError(f'Could not find model architecture from config: {_cfg}')
    return arch, cfg


def search_nested_config(config, key):
    """Recursively searches for the value associated with the given key in a
    nested configuration of a model."""
    if isinstance(config, dict):
        for k, v in config.items():
            if k == key:
                return v
            if isinstance(v, (dict, list)):
                result = search_nested_config(v, key)
                if result is not None:
                    return result
    elif isinstance(config, list):
        for item in config:
            result = search_nested_config(item, key)
            if result is not None:
                return result
    return None
