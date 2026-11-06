import pytest
from transformers import AutoConfig

from lmdeploy.utils import _get_and_verify_max_len, override_hf_config


def test_get_and_verify_max_len():
    # with PretrainedConfig
    config = AutoConfig.from_pretrained('OpenGVLab/InternVL-Chat-V1-5-AWQ', trust_remote_code=True)
    assert (_get_and_verify_max_len(config, None) == 32768)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
    assert (_get_and_verify_max_len(config, 102400) == 102400)

    # with PretrainedConfig
    config = AutoConfig.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
    assert (_get_and_verify_max_len(config, None) == 32768)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
    assert (_get_and_verify_max_len(config, 102400) == 102400)


def test_get_chat_template_builtin_name():
    """A builtin chat-template name must resolve without crashing.

    Regression for #4533: get_chat_template imported DEPRECATED_CHAT_TEMPLATE_NAMES / REMOVED_CHAT_TEMPLATE_NAMES from
    lmdeploy.model, which were removed in #4252, so any non-file --chat-template value raised ImportError before
    reaching the registry check.
    """
    from lmdeploy.cli.utils import get_chat_template
    cfg = get_chat_template('llama2')
    assert cfg is not None
    assert cfg.model_name == 'llama2'


def test_get_chat_template_unregistered_name():
    """An unregistered name is rejected by the registry check, not by an
    ImportError."""
    from lmdeploy.cli.utils import get_chat_template
    with pytest.raises(AssertionError):
        get_chat_template('not-a-registered-template')


def test_get_chat_template_none():
    """Passing no chat template returns None."""
    from lmdeploy.cli.utils import get_chat_template
    assert get_chat_template(None) is None


def test_override_hf_config_nested():
    from transformers.models.llava import LlavaConfig
    hf_config = LlavaConfig()

    assert hf_config.model_type == 'llava'
    override_hf_config(hf_config, dict(model_type='llava_custom'))
    assert hf_config.model_type == 'llava_custom'

    assert hf_config.text_config.model_type == 'llama'
    assert hf_config.text_config.rope_parameters['rope_type'] == 'default'
    override_hf_config(hf_config, dict(text_config=dict(rope_parameters=dict(rope_type='yarn'))))
    assert hf_config.text_config.model_type == 'llama'
    assert hf_config.text_config.rope_parameters['rope_type'] == 'yarn'

    override_hf_config(hf_config,
                       dict(model_type='llava_custom2', text_config=dict(rope_parameters=dict(rope_type='yarn2'))))
    assert hf_config.model_type == 'llava_custom2'
    assert hf_config.text_config.model_type == 'llama'
    assert hf_config.text_config.rope_parameters['rope_type'] == 'yarn2'


def test_override_hf_config_dict():
    cfg = dict(a=1, nested=dict(x=1, y=2))
    override_hf_config(cfg, dict(a=2, nested=dict(y=3, z=4), new=dict(k=1)))
    assert cfg == dict(a=2, nested=dict(x=1, y=3, z=4), new=dict(k=1))
