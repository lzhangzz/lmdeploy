# Copyright (c) OpenMMLab. All rights reserved.

from .base import OUTPUT_MODELS, BaseOutputModel


@OUTPUT_MODELS.register_module(name='tm')
class TurbomindModel(BaseOutputModel):
    """In-memory turbomind output model."""
    pass
