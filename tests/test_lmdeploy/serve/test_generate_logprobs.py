# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import APIRouter

from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.serve.openai.endpoints.generate import (
    _create_token_logprobs,
    _create_top_logprobs,
    check_request,
    register,
)
from lmdeploy.serve.openai.protocol import GenerateReqInput


def _make_generate(context):
    router = APIRouter()
    register(router, context)
    for route in router.routes:
        if getattr(route, 'path', None) == '/generate':
            return route.endpoint
    raise AssertionError('/generate route not registered')


class _SessionManager:

    def has(self, session_id):
        return False

    def get(self, *args, **kwargs):
        raise AssertionError('empty scoring response must not create a session')

    def remove(self, session):
        pass


class _ServerContext:

    def __init__(self, config, speculative_config=None, async_engine=None):
        self._session_manager = _SessionManager()
        self.async_engine = async_engine or SimpleNamespace(
            backend_config=config,
            session_mgr=self._session_manager,
            speculative_config=speculative_config)
        self.default_gen_config = {}

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    @property
    def session_manager(self):
        return self._session_manager

    def create_session(self, session_id):
        return SimpleNamespace()


def test_generate_logprob_validation():
    context = _ServerContext(TurbomindEngineConfig())

    assert check_request(
        GenerateReqInput(prompt='Paris is the capital of',
                         max_tokens=2,
                         return_logprob=True), context) == ''
    assert check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         top_logprobs_num=2), context) == ''
    assert 'positive integer' in check_request(
        GenerateReqInput(input_ids=[1, 2], max_tokens=0), context)
    assert 'top_logprobs_num requires return_logprob=True' in check_request(
        GenerateReqInput(input_ids=[1, 2], top_logprobs_num=2), context)
    assert 'negative integer' in check_request(
        GenerateReqInput(input_ids=[1, 2],
                         return_logprob=True,
                         top_logprobs_num=-1), context)
    assert 'exactly one of prompt or input_ids' in check_request(
        GenerateReqInput(prompt='hi', input_ids=[1]), context)


def test_generate_logprob_formatting_preserves_requested_empty():
    assert _create_token_logprobs(None, None) is None
    assert _create_token_logprobs([], []) is None
    assert _create_token_logprobs(
        [4], [{4: -0.75, 8: -2.0}]) == [(-0.75, 4)]
    with pytest.raises(ValueError):
        _create_token_logprobs([4], [])

    assert _create_token_logprobs(
        [5], [{5: -0.25}]) == [(-0.25, 5)]
    assert _create_top_logprobs(None, None, 2) is None
    assert _create_top_logprobs([], [], 2) is None
    assert _create_top_logprobs([4], [{4: -0.75}], 0) is None
    assert _create_top_logprobs(
        [4], [{4: -0.75, 8: -2.0, 9: -3.0}], 3) == [[(-0.75, 4), (-2.0, 8), (-3.0, 9)]]
    assert _create_top_logprobs(
        [4], [{4: -0.75, 8: -2.0, 9: -3.0, 10: -4.0}], 3) == [[(-2.0, 8), (-3.0, 9),
                                                                 (-4.0, 10)]]
    with pytest.raises(ValueError):
        _create_top_logprobs([4], [], 2)


class _RawRequest:

    async def is_disconnected(self):
        return False


class _DecodeEngine:

    def __init__(self, expected_logprobs=1):
        self.backend_config = TurbomindEngineConfig()
        self.session_mgr = _SessionManager()
        self.epoch = 1
        self.expected_logprobs = expected_logprobs

    async def preprocess(self, **kwargs):
        assert kwargs['input_ids'] == [1, 2]
        assert kwargs['gen_config'].max_new_tokens == 2
        assert kwargs['gen_config'].logprobs == self.expected_logprobs
        return SimpleNamespace(input_token_len=2)

    async def generate(self, prepared_request, stream_response=True):
        assert prepared_request.input_token_len == 2
        assert stream_response is True
        yield SimpleNamespace(
            response='x',
            token_ids=[7],
            logprobs=[{7: -0.125, 8: -2.0}],
            finish_reason='length',
            input_token_len=2,
            generate_token_len=1,
        )


def test_generate_keeps_decode_logprobs():
    engine = _DecodeEngine()
    context = _ServerContext(engine.backend_config, async_engine=engine)
    generate = _make_generate(context)
    request = GenerateReqInput(input_ids=[1, 2],
                               max_tokens=2,
                               return_logprob=True)
    result = asyncio.run(generate(request, raw_request=_RawRequest()))

    assert result.text == 'x'
    assert result.output_ids == [7]
    assert result.meta_info.output_token_logprobs == [(-0.125, 7)]
    assert result.meta_info.output_top_logprobs is None


def test_generate_keeps_decode_top_logprobs():
    engine = _DecodeEngine(expected_logprobs=2)
    context = _ServerContext(engine.backend_config, async_engine=engine)
    generate = _make_generate(context)
    request = GenerateReqInput(input_ids=[1, 2],
                               max_tokens=2,
                               return_logprob=True,
                               top_logprobs_num=2)
    result = asyncio.run(generate(request, raw_request=_RawRequest()))

    assert result.meta_info.output_token_logprobs == [(-0.125, 7)]
    assert result.meta_info.output_top_logprobs == [[(-0.125, 7), (-2.0, 8)]]
