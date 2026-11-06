# Copyright (c) OpenMMLab. All rights reserved.
"""Request validation for the ``/v1/chat/completions`` endpoint."""
from __future__ import annotations

from lmdeploy.serve.openai.protocol import ChatCompletionRequest

# Upper bound for `n` (number of choices). Each choice is a separate
# engine.generate() call on the fan-out path, so cap to protect resources.
_MAX_FANOUT_N = 128


def check_request(request: ChatCompletionRequest,
                  server_context,
                  json_request: dict | None = None) -> str:
    """Validate chat-completion options and fan-out compatibility."""
    session_manager = server_context.session_manager

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    # check sampling settings
    if request.n is not None and request.n <= 0:
        return f'The n {request.n!r} must be a positive int.'
    # n > 1 is implemented as server-side fan-out (N independent engine
    # generate() calls). Cap it to prevent unbounded resource use.
    if request.n is not None and request.n > _MAX_FANOUT_N:
        return (f'The n {request.n!r} exceeds the maximum supported '
                f'choices ({_MAX_FANOUT_N}).')
    if request.n is not None and request.n > 1 and request.session_id not in (
            None, -1):
        return 'n > 1 cannot be used with an explicit session_id.'
    if request.logit_bias:
        return 'logit_bias is not supported by the TurboMind engine.'
    if request.top_p is not None and not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k is not None and request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'
    if request.min_p is not None and not (0 <= request.min_p <= 1):
        return f'The min_p {request.min_p!r} must be in [0, 1].'
    if request.max_completion_tokens is not None and request.max_completion_tokens <= 0:
        return f'The max_completion_tokens {request.max_completion_tokens!r} must be a positive integer.'
    if 'max_tokens' in request.model_fields_set:
        max_tokens = request.model_dump(include={'max_tokens'})['max_tokens']
        if max_tokens is not None and max_tokens <= 0:
            return f'The max_tokens {max_tokens!r} must be a positive integer.'
    if request.min_new_tokens is not None and request.min_new_tokens < 0:
        return f'The min_new_tokens {request.min_new_tokens!r} cannot be a negative integer.'
    if request.top_logprobs is not None and request.top_logprobs > 20:
        return f'The top_logprobs {request.top_logprobs!r} must be in [0, 20].'

    # Validate input_ids and image_data constraints.
    # messages has higher priority. input_ids and image_data are only used when
    # messages is empty. image_data requires input_ids.
    messages_empty = len(request.messages) == 0
    if not messages_empty:
        # messages is active — input_ids and image_data must not be set
        if request.input_ids is not None:
            return 'input_ids cannot be used when messages is non-empty. messages takes priority.'
        if request.image_data is not None:
            return 'image_data cannot be used when messages is non-empty. messages takes priority.'
    else:
        # messages is empty — input_ids and image_data are the active inputs
        if request.image_data is not None and request.input_ids is None:
            return 'image_data requires input_ids to be set when messages is empty.'
        if request.input_ids is None:
            return 'messages must not be empty unless input_ids is set.'
        if len(request.input_ids) == 0:
            return 'The input_ids must not be an empty list.'

    parser_cls = server_context.response_parser_cls
    if request.tool_choice == 'required' and not request.tools:
        return '`tool_choice="required"` requires at least one tool.'

    if request.tool_choice != 'none' and request.tools:
        if parser_cls is None or parser_cls.tool_parser_cls is None:
            return 'Please launch the api_server with --tool-call-parser if you want to use tools.'

    return ''
