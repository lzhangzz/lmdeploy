# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum

IMAGE_TOKEN = '<IMAGE_TOKEN>'


class Modality(Enum):
    IMAGE = 'image'
    VIDEO = 'video'

    def __eq__(self, other):
        if isinstance(other, Modality):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)
