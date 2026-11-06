# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import is_bf16_supported
from lmdeploy.vl.model.builder import load_vl_model


def _get_hf_config_mm_feature_dtype(hf_config) -> torch.dtype | None:
    """Get multimodal feature dtype from the original Transformers config."""

    def _to_torch_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            return getattr(torch, dtype.removeprefix('torch.'), None)
        return None

    if hf_config is None:
        return None

    configs = [hf_config, getattr(hf_config, 'text_config', None), getattr(hf_config, 'llm_config', None)]

    for config in configs:
        if config is None:
            continue
        for attr_name in ('dtype', 'torch_dtype'):
            dtype = _to_torch_dtype(getattr(config, attr_name, None))
            if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
                return dtype
    return None


def _resolve_mm_feature_dtype(hf_config, backend_config) -> torch.dtype | None:
    """Resolve multimodal feature dtype.

    HF config provides the default. An explicit backend dtype overrides it,
    while ``auto`` keeps the HF-derived dtype.
    """
    dtype = _get_hf_config_mm_feature_dtype(hf_config)
    backend_dtype = getattr(backend_config, 'dtype', 'auto')
    if backend_dtype != 'auto':
        dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }[backend_dtype]
    if dtype == torch.bfloat16 and not is_bf16_supported():
        dtype = torch.float16
    return dtype


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    """Raise exception on finish."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        raise e


class ImageEncoder:
    """Image encoder."""

    def __init__(
        self,
        model_path: str,
        backend_config: TurbomindEngineConfig | None = None,
        trust_remote_code: bool = False,
    ):
        self.model = load_vl_model(model_path,
                                   'turbomind',
                                   backend_config=backend_config,
                                   trust_remote_code=trust_remote_code)
        self.mm_feature_dtype = _resolve_mm_feature_dtype(
            getattr(self.model, 'hf_config', None), backend_config)
        if self.model is not None and hasattr(self.model, 'set_mm_feature_dtype'):
            self.model.set_mm_feature_dtype(self.mm_feature_dtype)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._uses_new_preprocess = self._is_new_preprocess_api(self.model)
        torch.cuda.empty_cache()

    @staticmethod
    def _is_new_preprocess_api(model) -> bool:
        """New-style preprocess takes `input_prompt` + `mm_processor_kwargs`,
        legacy takes only `messages`."""
        if model is None:
            return False
        sig = inspect.signature(model.preprocess).parameters
        return 'input_prompt' in sig and 'mm_processor_kwargs' in sig

    async def preprocess(self,
                         messages: list[dict],
                         input_prompt: str | list[int] | None = None,
                         mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Preprocess multimodal data in the messages."""
        if self._uses_new_preprocess:
            future = asyncio.get_event_loop().run_in_executor(
                self.executor, self.model.preprocess, messages, input_prompt, mm_processor_kwargs)
        else:
            future = asyncio.get_event_loop().run_in_executor(
                self.executor, self.model.preprocess, messages)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        return outputs

    async def wrap_for_turbomind(
        self,
        messages: list[dict],
        chat_template,
        tokenizer,
        tools: list[object] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> dict:
        """Pack preprocessed inputs for TurboMind's native vision encoder.

        Args:
            messages (list[dict]): a list of message, which is supposed to be
                the output of `preprocess`
            chat_template: the chat template
            tokenizer: the tokenizer model
            tools: the tools for the chat template
            chat_template_kwargs: additional arguments for the chat template

        Returns:
            dict: ``prompt``, ``input_ids`` and the ``multimodal`` items consumed by the engine.
        """
        if not self.model.has_input_ids(messages):
            result = self.model.to_pytorch(messages,
                                           chat_template,
                                           tokenizer,
                                           tools=tools,
                                           chat_template_kwargs=chat_template_kwargs)
        else:
            result = self.model.to_pytorch_with_input_ids(messages)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], list):
                messages[i]['preprocess'] = None
        return result
