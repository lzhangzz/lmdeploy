# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.vl.model.base import VISION_MODELS
from lmdeploy.vl.model.qwen3 import Qwen3VLModel

_INTERN_S2_ARCHS = [
    'InternS2MobiusForConditionalGeneration',
    'InternS2MobiusForCausalLM',
    'InternS2PreviewForConditionalGeneration',
    'InternS2PreviewForCausalLM',
]


def check_transformers():
    try:
        # import the config module, not the model, so the check stays cheap
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: F401
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3_5 model."""

    _arch = [
        'Qwen3_5ForConditionalGeneration',
        'Qwen3_5MoeForConditionalGeneration',
        *_INTERN_S2_ARCHS,
    ]

    def build_preprocessor(self, trust_remote_code: bool = False):
        check_transformers()
        super().build_preprocessor(trust_remote_code=trust_remote_code)
