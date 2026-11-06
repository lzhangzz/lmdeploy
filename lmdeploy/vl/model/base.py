# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from abc import ABC, abstractmethod
from itertools import groupby
from typing import Any

import torch
from mmengine import Registry
from transformers import AutoConfig, AutoTokenizer

from lmdeploy.archs import get_model_arch
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.preprocess_utils import (
    get_expanded_input_ids,
    get_expanded_mm_items,
    get_mm_items_offset,
    get_override_size,
)

VISION_MODELS = Registry('vision_model')


class VisionModel(ABC):
    """Visual model which extract image feature."""
    _arch: str | list[str] = None

    # mapping from processor output attribute names to modality types
    ATTR_NAME_TO_MODALITY = {
        # image-related attributes
        'pixel_values': Modality.IMAGE,
        'image_grid_thw': Modality.IMAGE,
        # video-related attributes
        'pixel_values_videos': Modality.VIDEO,
        'video_grid_thw': Modality.VIDEO,
        'video_second_per_grid': Modality.VIDEO,
    }

    # processor output attributes that carry the main feature tensor
    FEATURE_NAMES = [
        'pixel_values',
        'pixel_values_videos',
    ]

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 hf_config: AutoConfig = None,
                 backend: str = '',
                 trust_remote_code: bool = False):
        """init."""
        self.model_path = model_path
        self.with_llm = with_llm
        self.backend = backend
        self.mm_feature_dtype: torch.dtype | None = None
        if hf_config is None:
            _, hf_config = get_model_arch(model_path, trust_remote_code=trust_remote_code)
        self.hf_config = hf_config
        self.image_token_id = self.get_pad_token_id(model_path, hf_config, trust_remote_code=trust_remote_code) or 0

    def set_mm_feature_dtype(self, dtype: torch.dtype | None):
        """Set target dtype for floating MM feature tensors."""
        self.mm_feature_dtype = dtype

    @classmethod
    def _postprocess_mm_output(cls, output, target_dtype: torch.dtype | None):
        """Cast floating processor-output tensors to the target model dtype."""
        if not isinstance(target_dtype, torch.dtype):
            return output
        if not target_dtype.is_floating_point:
            return output
        if isinstance(output, torch.Tensor):
            if output.is_floating_point() and output.dtype != target_dtype:
                return output.to(dtype=target_dtype)
            return output
        if isinstance(output, dict):
            return {key: cls._postprocess_mm_output(value, target_dtype) for key, value in output.items()}
        if isinstance(output, list):
            return [cls._postprocess_mm_output(value, target_dtype) for value in output]
        return output

    def get_pad_token_id(self, model_path, hf_config, trust_remote_code: bool = False):
        """Get pad_token_id from hf_config or tokenizer."""
        pad_token_id = getattr(hf_config, 'pad_token_id', None)
        if pad_token_id is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
                pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            except Exception as e:
                print(e)
                pass
        return pad_token_id

    @abstractmethod
    def build_preprocessor(self, trust_remote_code: bool = False):
        """Build the preprocessor.

        NOTE: When the derived class implements this method, try not to
        introduce the upper stream model repo as a thirdparty package
        """
        raise NotImplementedError()

    def build_model(self, trust_remote_code: bool = False):
        """Load the whole VLM model.

        Only used when `with_llm=True`.
        """
        raise NotImplementedError()

    def preprocess(self,
                   messages: list[dict],
                   input_prompt: str | list[int],
                   mm_processor_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Preprocess multimodal data and return a dict with ``input_ids`` and
        multimodal features.

        New-style models inherit this implementation. Legacy models override with `def preprocess(self, messages)`.
        """

        mm_items = self.collect_multimodal_items(messages)

        raw_images, raw_videos, video_metadatas = [], [], []
        for modality, data, params in mm_items:
            if modality == Modality.IMAGE:
                raw_images.append(data)
            elif modality == Modality.VIDEO:
                raw_videos.append(data)
                video_metadatas.append(params.get('video_metadata', None))
            else:
                raise ValueError(f'unsupported modality {modality}')

        # get kwargs for processor
        kwargs = {}
        images_kwargs = {}
        videos_kwargs = {}
        mm_processor_kwargs = mm_processor_kwargs or {}
        if raw_images:
            kwargs['images'] = raw_images
            image_size = get_override_size(self.processor.image_processor,
                                           mm_processor_kwargs.get('image'),
                                           modality='image')
            if image_size is not None:
                images_kwargs['size'] = image_size
        if raw_videos:
            kwargs['videos'] = raw_videos
            videos_kwargs['video_metadata'] = video_metadatas
            # perform resize in hf processor, while sample frames has been done in video loader
            videos_kwargs['do_resize'] = True
            videos_kwargs['do_sample_frames'] = False
            video_size = get_override_size(self.processor.video_processor,
                                           mm_processor_kwargs.get('video'),
                                           modality='video')
            if video_size is not None:
                videos_kwargs['size'] = video_size
        if images_kwargs:
            kwargs['images_kwargs'] = images_kwargs
        if videos_kwargs:
            kwargs['videos_kwargs'] = videos_kwargs

        # process raw items with hf processor
        input_text = input_prompt if isinstance(input_prompt, str) else ''
        processor_outputs = self.processor(
            text=[input_text],
            padding=True,
            return_tensors='pt',
            **kwargs,
        )

        # collect items from hf processor outputs and categorized by modality for lmdeploy to consume
        collected_mm_items: dict[Modality, dict[str, Any]] = {}
        for attr_name, value in processor_outputs.items():
            current_modality = self.ATTR_NAME_TO_MODALITY.get(attr_name)
            if current_modality:
                if current_modality not in collected_mm_items:
                    collected_mm_items[current_modality] = {}

                if attr_name in self.FEATURE_NAMES:
                    value = self._postprocess_mm_output(value, getattr(self, 'mm_feature_dtype', None))
                    processor_outputs[attr_name] = value
                    attr_name = 'feature'

                collected_mm_items[current_modality][attr_name] = value

        # get input_ids, expand multimodal tokens only when we receive input ids from /generate endpoint
        if isinstance(input_prompt, str):
            input_ids = processor_outputs['input_ids'].flatten()
        else:
            input_ids = get_expanded_input_ids(input_prompt, collected_mm_items, self.processor, self.mm_tokens)

        # compute offsets for all items
        for modality, item in collected_mm_items.items():
            mm_token_id = self.mm_tokens.get_token_id_by_modality(modality)
            item['offset'] = get_mm_items_offset(input_ids=input_ids, mm_token_id=mm_token_id)

        # expand bundled hf processor outputs into per-image/video entry for lmdeploy to consume
        expanded_mm_items = get_expanded_mm_items(collected_mm_items, self.mm_tokens)

        result = dict(input_ids=input_ids.tolist(), multimodal=expanded_mm_items)
        return result

    @staticmethod
    def has_input_ids(messages: list[dict]) -> bool:
        """Check whether the messages contain input_ids directly.

        This is True when the first (and only) user message content is a list
        whose first item carries a ``text`` field that is itself a list of
        token ids (i.e. the output of the ``/generate`` endpoint rather than a
        plain text prompt).

        Args:
            messages (list[dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            bool: whether the messages contain input_ids directly
        """
        user_contents = [x['content'] for x in messages if x['role'] == 'user']
        if len(user_contents) != 1:
            return False
        content = user_contents[0]
        if not isinstance(content, list) or not content:
            return False
        first_item = content[0]
        return isinstance(first_item, dict) and isinstance(first_item.get('text'), list)

    @staticmethod
    def get_input_prompt(messages: list[dict], chat_template,
                         tools: list[object] | None = None,
                         chat_template_kwargs: dict | None = None) -> str | list[int]:
        """Return the input prompt for the preprocessor.

        When the messages already carry embedded token ids (from the
        ``/generate`` endpoint), extract and return them directly.
        Otherwise, render the messages through *chat_template* to produce a
        plain-text prompt string.

        Args:
            messages: Preprocessed message list.
            chat_template: Chat template used to render a text prompt.
            tools: Optional tools forwarded to chat template rendering.
            chat_template_kwargs: Extra kwargs forwarded to ``messages2prompt``.
        Returns:
            A list of token ids when input_ids are embedded, otherwise a str.
        """
        if VisionModel.has_input_ids(messages):
            return messages[0]['content'][0]['text']
        return chat_template.messages2prompt(messages, tools=tools, **(chat_template_kwargs or {}))

    def to_pytorch(self, messages, chat_template, tokenizer, chat_template_kwargs=None, **kwargs):
        """Pack the preprocessing results into ``prompt``, ``input_ids`` and
        ``multimodal`` items consumed by TurboMind's native vision encoder.

        Args:
            messages(list[dict]): the output of `preprocess`
            chat_template: the chat template defined in `lmdeploy/model.py`
            tokenizer: the tokenizer model
            chat_template_kwargs: additional arguments for chat template
                processing, such as `add_vision_id` and `enable_thinking`
        """
        raise NotImplementedError()

    @staticmethod
    def collect_multimodal_items(messages):
        """Gather all multimodal items along with their respective parameters
        from the messages and compile them into a single list.

        Args:
            messages (list[dict]): a list of message
        Returns:
            list[tuple[Modality, Any, dict]]: a list of (modality, data, params) for each multimodal item
        """
        multimodal_items = []
        for message in messages:
            content = message['content']
            if not isinstance(content, list):
                continue

            for x in content:
                if not isinstance(x, dict):
                    continue

                modality = x.get('type')
                if modality is None or modality == 'text':
                    continue

                data = x.get('data')
                params = {k: v for k, v in x.items() if k not in ['type', 'data']}
                multimodal_items.append((modality, data, params))

        return multimodal_items

    @staticmethod
    def IMAGE_TOKEN_included(messages):
        """Check whether the IMAGE_TOKEN is included in the messages.

        Args:
            messages (list[dict]): a list of message
        Returns:
            bool: whether the IMAGE_TOKEN is included in the messages
        """
        for message in messages:
            role, content = message['role'], message['content']
            if role != 'user':
                continue
            if isinstance(content, str) and '<IMAGE_TOKEN>' in content:
                return True
            elif isinstance(content, list):
                content = [x['text'] for x in content if x['type'] == 'text']
                if any('<IMAGE_TOKEN>' in x for x in content):
                    return True
        return False

    def to_pytorch_with_input_ids(self, messages):
        """Pack the preprocessing results when input_ids are provided directly.

        Args:
            messages(list[dict]): the output of `preprocess`
        """
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        _input_ids = messages[0]['content'][0]['text']
        segs = []
        for k, g in groupby(_input_ids, lambda x: x == self.image_token_id):
            if not k:
                segs.append(list(g))
            else:
                segs.extend([[]] * (len(list(g)) - 1))
        if _input_ids[0] == self.image_token_id:
            segs = [[]] + segs
        if _input_ids[-1] == self.image_token_id:
            segs = segs + [[]]

        assert self.image_token_id == preps[0]['image_token_id']
        assert len(segs) == len(preps) + 1, (f'the number of image token id {self.image_token_id} is not equal '
                                             f'to input images, {len(segs) - 1} vs {len(preps)}')
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                input_ids.extend([self.image_token_id] * image_tokens)
            input_ids.extend(seg)

        return dict(prompt=None, input_ids=input_ids, multimodal=preps)

    def to_pytorch_aux(self, messages, prompt, IMAGE_TOKEN, tokenizer):
        """Auxiliary function to pack the preprocessing results.

        Args:
            messages(list[dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenizer: the tokenizer model
        """
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(preps) + 1, (f'the number of {IMAGE_TOKEN} is not equal '
                                             f'to input images, {len(segs) - 1} vs {len(preps)}')

        # calculate the image token offset for each image
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                assert self.image_token_id == preps[i - 1]['image_token_id']
                input_ids.extend([self.image_token_id] * image_tokens)
            token_ids = tokenizer.encode(seg, add_bos=(i == 0))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
        if arch and (arch == cls._arch or arch in cls._arch):
            return True
        return False

@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: str | None = None
    video_token: str | None = None

    image_token_id: int | None = None
    video_token_id: int | None = None

    def get_token_id_by_modality(self, modality: Modality) -> int | None:
        """Get token ID for a given modality."""
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
        }.get(modality)
