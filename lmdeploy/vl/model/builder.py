# Copyright (c) OpenMMLab. All rights reserved.
import os

from lmdeploy.archs import get_model_arch
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger, get_model
from lmdeploy.vl.model.base import VISION_MODELS

from .internvl import InternVLVisionModel  # noqa F401
from .internvl3_hf import InternVL3VisionModel  # noqa F401
from .qwen2 import Qwen2VLModel  # noqa F401
from .qwen3 import Qwen3VLModel  # noqa F401
from .qwen3_5 import Qwen3_5Model  # noqa F401

logger = get_logger('lmdeploy')


def load_vl_model(model_path: str,
                  backend: str,
                  backend_config: TurbomindEngineConfig | None = None,
                  trust_remote_code: bool = False):
    """Load visual model.

    Args:
        model_path(str): the path or repo_id from model hub of the model
        backend(str): 'turbomind' for inference, or None outside inference
        backend_config: the config of the inference engine
    """
    if not os.path.exists(model_path):
        revision = getattr(backend_config, 'revision', None)
        download_dir = getattr(backend_config, 'download_dir', None)
        model_path = get_model(model_path, revision=revision, download_dir=download_dir)

    _, hf_config = get_model_arch(model_path, trust_remote_code=trust_remote_code)
    kwargs = dict(model_path=model_path,
                  hf_config=hf_config,
                  backend=backend,
                  trust_remote_code=trust_remote_code)

    for name, module in VISION_MODELS.module_dict.items():
        try:
            if module.match(hf_config):
                logger.info(f'matching vision model: {name}')
                model = module(**kwargs)
                # TurboMind runs the visual encoder natively, so only the
                # preprocessor is needed on the Python side.
                model.build_preprocessor(trust_remote_code=trust_remote_code)
                return model
        except Exception as e:
            logger.error(f'build vision model {name} failed, {e}')
            raise

    raise ValueError(f'unsupported vl model with config {hf_config}')
