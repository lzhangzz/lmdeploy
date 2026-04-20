# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from ...utils import _get_and_verify_max_len, is_bf16_supported
from ..supported_models import SUPPORTED_ARCHS
from .config import TurbomindModelConfig
from .source_model.base import INPUT_MODELS
from .source_model.utils import load_model_config
from .target_model.base import BaseOutputModel

SUPPORTED_FORMATS = ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4', None]
logger = get_logger('lmdeploy')


def _deep_merge(base: dict, override: dict, path: str = '') -> dict:
    """Recursively merge override into base, mutating base in-place."""
    for k, v in override.items():
        key_path = f'{path}.{k}' if path else k
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v, key_path)
        else:
            if k not in base:
                logger.warning(f'hf_overrides key "{key_path}" not found in config, applying anyway')
            base[k] = v
    return base

_DEFAULT_GROUP_SIZES = {
    'awq': 128,
    'gptq': 128,
    'compressed-tensors': 128,
    'fp8': 128,
    'mxfp4': 32,
}

_SUPPORTED_GROUP_SIZES = {
    'awq': frozenset({128}),
    'gptq': frozenset({128}),
    'compressed-tensors': frozenset({32, 128}),
    'fp8': frozenset({128}),
    'mxfp4': frozenset({32}),
}


def _validate_quant_group_size(model_format: str | None, group_size: int | None) -> int | None:
    """Normalize and validate quantized group sizes.

    The low-level int4 kernels can be shared across formats, but we only expose the format/group-size combinations that
    are verified end to end.
    """
    if group_size in (None, 0):
        group_size = _DEFAULT_GROUP_SIZES.get(model_format, group_size)

    supported_group_sizes = _SUPPORTED_GROUP_SIZES.get(model_format)
    if supported_group_sizes is not None and group_size not in supported_group_sizes:
        supported = ', '.join(map(str, sorted(supported_group_sizes)))
        raise ValueError(f'Unsupported group_size={group_size} for model_format="{model_format}". '
                         f'Supported group_size values: {supported}.')

    return group_size


def get_spec_registered_name(model_path: str, model_format: str):
    """Get the registered name of a model. The name will be used to access the
    INPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4']
    """
    arch = get_model_arch(model_path)[0]
    register_name = SUPPORTED_ARCHS[arch]
    return register_name


def get_output_model_registered_name_and_config(model_path: str, model_format: str, dtype: str, group_size: int):
    """Get the registered name of the turbomind model and its configuration
    according to the input model path, format and user-input config. The name
    will be used to access the OUTPUT_MODELS registry.

    Args:
        model_path (str): the path of the input model
        model_format (str): the format of the model, which can be one of
            ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4']
        dtype (str): the data type of the model's weights and activations
        group_size (int): the quantization group size used by grouped formats
    """
    register_name = 'tm'

    has_bf16 = is_bf16_supported()

    model_arch, model_config = get_model_arch(model_path)

    # infer dtype from device and model config
    if dtype == 'auto':
        # pick dtype by device as default
        dtype = 'bfloat16' if has_bf16 else 'float16'
        # dtype from model (prefer `dtype` over deprecated `torch_dtype`)
        torch_dtype = getattr(model_config, 'dtype', None)
        if torch_dtype is None:
            torch_dtype = getattr(model_config, 'torch_dtype', None)
        TORCH_DTYPE_MAP = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}
        dtype = TORCH_DTYPE_MAP.get(torch_dtype, dtype)

    if dtype == 'bfloat16' and not has_bf16:
        logger.warning('data type fallback to float16 since '
                       'torch.cuda.is_bf16_supported is False')
        dtype = 'float16'

    config = TurbomindModelConfig.from_dict()

    session_len = _get_and_verify_max_len(model_config, None)

    group_size = _validate_quant_group_size(model_format, group_size)

    if model_format in ['awq', 'gptq', 'compressed-tensors']:
        dtype = 'float16'
        if model_format == 'compressed-tensors':
            model_format = 'awq'

    config.model_config.model_arch = model_arch
    config.model_config.data_type = dtype
    config.model_config.model_format = model_format
    config.model_config.group_size = group_size
    config.model_config.session_len = session_len

    return register_name, config


def get_tm_config(model_path,
                  model_name,
                  chat_template_name,
                  engine_config: TurbomindEngineConfig,
                  group_size: int = None):
    """Compute finalized TurbomindModelConfig and the TextModelSpec.

    Returns:
        tuple: (spec, tm_cfg, model_path)
    """
    _, cfg = get_model_arch(model_path)
    quant_config = search_nested_config(cfg.to_dict(), 'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None or engine_config.model_format == quant_method, (
            f'mismatched quant method: user input "{engine_config.model_format}" '
            f'vs model quant_config "{quant_method}"')
        assert not group_size or group_size == _group_size, (f'mismatched quant group size: user input "{group_size}" '
                                                             f'vs model quant_config "{_group_size}"')

        if quant_method == 'awq':
            assert version == 'gemm', f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and quant_config.get(
                'sym', True), f'unsupported quant config: {quant_config}'
        elif quant_method == 'fp8':
            pass
        elif quant_method == 'mxfp4':
            _group_size = 32
        elif quant_method == 'compressed-tensors':
            _format = quant_config['config_groups']['group_0']['format']
            assert _format == 'pack-quantized', ('compressed-tennsors only supports pack-quantized format, '
                                                 f'but got {_format}')
            _weights = quant_config['config_groups']['group_0']['weights']
            _group_size = _weights['group_size']
            _num_bits = _weights['num_bits']
            _type = _weights['type']
            assert _num_bits == 4 and _type == 'int', ('pack-quantized requires 4-bit int, '
                                                       f'but got {_num_bits}-bit {_type}')
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    group_size = _validate_quant_group_size(engine_config.model_format, group_size)

    # Default to 'hf' for unquantized checkpoints. Without this, a None
    # model_format flows through to ModelConfig.verify() and fails. The old
    # pipeline tolerated None because config_from_dict filtered None values;
    # the new spec.to_legacy_config() assigns unconditionally.
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    spec_name = get_spec_registered_name(model_path, engine_config.model_format)

    output_model_name, tm_cfg = get_output_model_registered_name_and_config(model_path=model_path,
                                                                            model_format=engine_config.model_format,
                                                                            dtype=engine_config.dtype,
                                                                            group_size=group_size)

    engine_config.dtype = tm_cfg.model_config.data_type
    engine_config.model_format = tm_cfg.model_config.model_format
    if engine_config.session_len is None:
        engine_config.session_len = tm_cfg.model_config.session_len
    if engine_config.attn_tp_size is None:
        engine_config.attn_tp_size = 1
    if engine_config.attn_cp_size is None:
        engine_config.attn_cp_size = 1
    if engine_config.mlp_tp_size is None:
        engine_config.mlp_tp_size = 1

    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name
    tm_cfg.model_config.attn_tp_size = engine_config.attn_tp_size
    tm_cfg.model_config.attn_cp_size = engine_config.attn_cp_size
    tm_cfg.model_config.mlp_tp_size = engine_config.mlp_tp_size

    hf_cfg = load_model_config(model_path)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)

    BaseOutputModel.finalize_config(spec, tm_cfg)

    return spec, tm_cfg, model_path
