# Author: victorg

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from typing import Sequence
from mmengine.model.base_model import BaseDataPreprocessor
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor


@MODELS.register_module()
class DetDataPreprocessor7Channel(DetDataPreprocessor):
    """7-channel variant of DetDataPreprocessor that skips the 1-or-3 channel mean/std check."""

    def __init__(self,
                 mean: Sequence[float],
                 std: Sequence[float],
                 pad_size_divisor: int = 1,
                 pad_value: float = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False,
                 batch_augments: Sequence[dict] = None):
        # 1) Initialize the absolute base class (skipping ImgDataPreprocessor):
        BaseDataPreprocessor.__init__(
            self,
            non_blocking=non_blocking,
        )

        # 2) Manually register a 7-channel mean/std (no length assertion)
        assert (mean is None) == (std is None), 'mean/std must both be None or both provided'
        if mean is not None:
            self._enable_normalize = True
            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
                False)
        else:
            self._enable_normalize = False

        # 3) Copy over ImgDataPreprocessor fields that DetDataPreprocessor expects
        #    (channel conversion flags, padding, etc.)
        self._channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

        # 4) Detection-specific extras
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor