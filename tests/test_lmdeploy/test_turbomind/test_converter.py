# yapf: disable
from lmdeploy import TurbomindEngineConfig
from lmdeploy.turbomind import update_parallel_config
from lmdeploy.turbomind.deploy.converter import (
    get_input_model_registered_name,
    get_output_model_registered_name_and_config,
)
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS

# yapf: enable


def test_torch_dtype_fallback():
    """torch_dtype is deprecated in transformers v5+; dtype should be
    preferred.

    This test ensures get_output_model_registered_name_and_config still works
    for models whose config exposes either `dtype` or `torch_dtype`.
    """
    _, config = get_output_model_registered_name_and_config(
        'internlm/internlm2-chat-7b',
        model_format='hf',
        dtype='auto',
        group_size=0,
    )
    assert config.model_config.data_type in ('float16', 'bfloat16')


def test_ffn_reader_kind_none():
    """FFN readers must handle kind=None (returns filter list, not tensors).

    This is the probe call from Ffn.apply() to discover parameter keys before loading actual tensor data. A missing
    guard causes KeyError with 'None' in the key string (regression test for InternLM2Reader._ffn bug).
    """
    import re

    from lmdeploy.turbomind.deploy.source_model.internlm2 import InternLM2Reader
    from lmdeploy.turbomind.deploy.source_model.llama import LlamaReader

    # Create minimal readers with fake params that match ffn patterns
    fake_params = {
        'model.layers.0.mlp.gate_proj.weight': None,
        'model.layers.0.mlp.down_proj.weight': None,
        'model.layers.0.mlp.up_proj.weight': None,
        'model.layers.0.feed_forward.w1.weight': None,
        'model.layers.0.feed_forward.w2.weight': None,
        'model.layers.0.feed_forward.w3.weight': None,
    }

    # LlamaReader with kind=None should return filtered key list
    reader = LlamaReader.__new__(LlamaReader)
    reader.params = dict(fake_params)
    reader.ffn_pattern = r'mlp'
    result = reader._ffn(0, None)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(k, str) for k in result)
    assert all(re.search(r'mlp', k) for k in result)

    # InternLM2Reader with kind=None should also return filtered key list
    reader2 = InternLM2Reader.__new__(InternLM2Reader)
    reader2.params = dict(fake_params)
    reader2.fp8_quant = None
    reader2.ffn_pattern = r'feed_forward'
    result2 = reader2._ffn(0, None)
    assert isinstance(result2, list)
    assert len(result2) > 0
    assert all(isinstance(k, str) for k in result2)
    assert all(re.search(r'feed_forward', k) for k in result2)


def test_registered_models():
    for model, model_format, group_size, register_name in [
        ('Qwen/Qwen3-8B', 'hf', 0, 'tm'),
        ('Qwen/Qwen3-30B-A3B', 'hf', 0, 'tm'),
    ]:
        input_name = get_input_model_registered_name(model, model_format=model_format)
        assert input_name in list(INPUT_MODELS.module_dict.keys())

        output_name, config = get_output_model_registered_name_and_config(model,
                                                                          model_format=model_format,
                                                                          dtype='auto',
                                                                          group_size=0)
        assert output_name == register_name
        assert config.model_config.group_size == group_size
        assert config.session_len > 0
        assert config.model_config.model_arch is not None


def test_update_from_engine_config():
    import copy
    _, _config = get_output_model_registered_name_and_config('internlm/internlm2-chat-7b',
                                                             model_format='hf',
                                                             dtype='auto',
                                                             group_size=0)
    config = copy.deepcopy(_config)
    config.update_from_engine_config(None)
    assert (config == _config)

    config = copy.deepcopy(_config)
    engine_config = TurbomindEngineConfig()
    update_parallel_config(engine_config)
    config.update_from_engine_config(engine_config)
    assert config.model_config.attn_tp_size == 1
    assert config.session_len == 32768

    config = copy.deepcopy(_config)
    engine_config = TurbomindEngineConfig(model_format='hf',
                                          tp=2,
                                          device_num=2,
                                          session_len=4000,
                                          max_batch_size=100,
                                          cache_max_entry_count=0.5,
                                          quant_policy=8,
                                          rope_scaling_factor=3.0,
                                          use_logn_attn=True,
                                          max_prefill_iters=64,
                                          num_tokens_per_iter=256)
    update_parallel_config(engine_config)
    config.update_from_engine_config(engine_config)

    assert (config.model_config.attn_tp_size == engine_config.attn_tp_size)
    assert (config.session_len == engine_config.session_len)
    assert (config.attention_config.rope_param.type == 'dynamic')
    assert (config.attention_config.rope_param.factor == engine_config.rope_scaling_factor)
    assert (config.attention_config.use_logn_attn == engine_config.use_logn_attn)


def test_dtype():
    testsets = [('auto', 'bfloat16'), ('float16', 'float16'), ('bfloat16', 'bfloat16')]
    for specified_dtype, expected_dtype in testsets:
        _, _config = get_output_model_registered_name_and_config('Qwen/Qwen3-8B',
                                                                 model_format='hf',
                                                                 dtype=specified_dtype,
                                                                 group_size=0)
        assert _config.model_config.data_type == expected_dtype
    for specified_dtype in ['auto', 'float16', 'bfloat16']:
        _, _config = get_output_model_registered_name_and_config('Qwen/Qwen3-8B-AWQ',
                                                                 model_format='awq',
                                                                 dtype=specified_dtype,
                                                                 group_size=128)
        assert _config.model_config.data_type == 'float16'
