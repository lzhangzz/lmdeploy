# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.openai.errors import create_error_response
from lmdeploy.serve.openai.protocol import (
    EmbeddingsRequest,
    EncodeRequest,
    EncodeResponse,
    PPLRequest,
    PPLResponse,
)
from lmdeploy.serve.utils.server_utils import validate_json_request


def register(router: APIRouter, server_context) -> None:

    @router.post('/v1/embeddings', tags=['unsupported'])
    async def create_embeddings(request: EmbeddingsRequest,
                                raw_request: Request = None):
        """Creates embeddings for the text."""
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     'Unsupported by turbomind.')

    @router.post('/v1/encode', dependencies=[Depends(validate_json_request)])
    async def encode(request: EncodeRequest, raw_request: Request = None):
        """Encode prompts.

        The request should be a JSON object with the following fields:

        - **input**: the prompt to be encoded. In str or list[str] format.
        - **do_preprocess**: whether do preprocess or not. Default to False.
        - **add_bos**: Whether to add a BOS token when encoding. Default to True.
        """

        def encode(prompt: str, do_preprocess: bool, add_bos: bool):
            if do_preprocess:
                prompt = server_context.async_engine.chat_template.get_prompt(
                    prompt)
            input_ids = server_context.async_engine.tokenizer.encode(
                prompt, add_bos=add_bos)
            return input_ids

        if isinstance(request.input, str):
            encoded = encode(request.input, request.do_preprocess,
                             request.add_bos)
            return EncodeResponse(input_ids=encoded, length=len(encoded))
        else:
            encoded, length = [], []
            for prompt in request.input:
                ids = encode(prompt, request.do_preprocess, request.add_bos)
                encoded.append(ids)
                length.append(len(ids))
            return EncodeResponse(input_ids=encoded, length=length)

    @router.post('/get_ppl', dependencies=[Depends(validate_json_request)])
    async def get_ppl(request: PPLRequest, raw_request: Request = None):
        """Get the perplexity (mean cross-entropy loss) of the input prompt.

        The request should be a JSON object with the following fields:

        - **input** (str | list[int]): the input to score, either raw text or token
          ids. Text is tokenized with ``tokenizer.encode`` (no chat template is
          applied).
        """
        async_engine = server_context.async_engine

        request_input = request.input

        # pydantic already validated `input` as `str | list[int]`; text ->
        # tokenizer.encode, otherwise the token ids are used as-is
        if isinstance(request_input, str):
            input_ids = async_engine.tokenizer.encode(request_input)
        else:
            input_ids = request_input
        if not input_ids:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         'Input must not be empty.')

        try:
            ppl = await async_engine.async_get_ppl(input_ids)
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

        response = PPLResponse(ppl=ppl)
        return response.model_dump()
