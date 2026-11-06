# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.archs import check_backend
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from .async_engine import AsyncEngine

logger = get_logger('lmdeploy')


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self,
                 model_path: str,
                 backend: str = 'turbomind',
                 backend_config: TurbomindEngineConfig | None = None,
                 trust_remote_code: bool = False,
                 allowed_media_domains: list[str] | None = None,
                 **kwargs) -> None:
        from lmdeploy.serve.processors import MultimodalProcessor
        from lmdeploy.vl.engine import ImageEncoder

        check_backend(backend)
        self.vl_encoder = ImageEncoder(model_path,
                                       backend_config=backend_config,
                                       trust_remote_code=trust_remote_code)
        super().__init__(model_path,
                         backend_config=backend_config,
                         trust_remote_code=trust_remote_code,
                         allowed_media_domains=allowed_media_domains,
                         **kwargs)
        # Update prompt_processor to support multimodal processing
        self.prompt_processor = MultimodalProcessor(self.tokenizer,
                                                    self.chat_template,
                                                    vl_encoder=self.vl_encoder,
                                                    allowed_media_domains=allowed_media_domains)
        if self.model_name == 'base':
            raise RuntimeError(
                'please specify chat template as guided in https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#set-chat-template'  # noqa: E501
            )

    def close(self):
        if hasattr(self, 'vl_encoder'):
            del self.vl_encoder
            super().close()
