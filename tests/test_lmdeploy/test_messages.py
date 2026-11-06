
import pytest

from lmdeploy import GenerationConfig, Tokenizer
from lmdeploy.messages import Response
from lmdeploy.utils import get_hf_gen_cfg


@pytest.mark.parametrize('field, value', [
    ('logprob_start_len', 0),
    ('repetition_ngram_size', 2),
    ('repetition_ngram_threshold', 2),
    ('return_routed_experts', True),
    ('with_cache', True),
    ('preserve_cache', True),
    ('migration_request', None),
    ('logits_processors', []),
])
def test_generation_config_rejects_removed_engine_fields(field, value):
    with pytest.raises(TypeError):
        GenerationConfig(**{field: value})


@pytest.mark.parametrize('carrier', [[], [{2: -0.5}]])
def test_response_extend_preserves_logprob_carrier(carrier):
    response = Response('', 0, 3)
    assert not hasattr(response, 'input_logprobs')
    response.extend(Response('', 0, 3, logprobs=carrier))
    assert response.logprobs == carrier


def test_engine_generation_config():
    tokenizer = Tokenizer('internlm/internlm2-chat-7b', trust_remote_code=True)
    config = GenerationConfig(n=3, stop_words=['<|im_end|>'])
    stop_token_ids = tokenizer.encode('<|im_end|>', add_bos=False)
    config.convert_stop_bad_words_to_ids(tokenizer)
    assert stop_token_ids == config.stop_token_ids
    assert isinstance(config.stop_token_ids, list) and \
        isinstance(config.stop_token_ids[0], int)


@pytest.mark.parametrize('model_path', [
    'deepseek-ai/DeepSeek-V3',
    'Qwen/Qwen2.5-32B-Instruct',
    'internlm/internlm3-8b-instruct',
])
def test_update_from_hf_gen_cfg(model_path):
    tokenizer = Tokenizer(model_path, trust_remote_code=True)
    model_cfg = get_hf_gen_cfg(model_path)

    generation_config = GenerationConfig()
    generation_config.update_from_hf_gen_cfg(model_cfg, tokenizer.eos_token_id)
    assert generation_config.stop_token_ids is not None
