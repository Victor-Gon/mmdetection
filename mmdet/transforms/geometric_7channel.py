# Author: victorg

import random
import mmcv

from typing import Sequence
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class Rotate7(BaseTransform):
    """Rotate a 7-channel image by a random angle in [-max_angle, max_angle]."""

    def __init__(self,
                 prob: float = 0.5,
                 max_angle: float = 30,
                 border_value: Sequence[float] = None):
        super().__init__()
        self.prob = prob
        self.max_angle = max_angle
        self.border_value = tuple(border_value) if border_value is not None else (0.0,)*7

    def transform(self, results):
        if random.random() > self.prob:
            return results

        img = results['img']
        assert img.ndim == 3 and img.shape[2] == 7, f'Rotate7 needs a 7-channel image, got {img.shape}'
        angle = random.uniform(-self.max_angle, self.max_angle)
        results['img'] = mmcv.imrotate(
            img,
            angle,
            border_value=self.border_value,
            auto_bound=False)
        return results


@TRANSFORMS.register_module()
class ShearX7(BaseTransform):
    """Shear-X a 7-channel image by a random magnitude in [-max_shear, max_shear]."""

    def __init__(self,
                 prob: float = 0.5,
                 max_shear: float = 0.3,
                 border_value: Sequence[float] = None):
        super().__init__()
        self.prob = prob
        self.max_shear = max_shear
        self.border_value = tuple(border_value) if border_value is not None else (0.0,)*7

    def transform(self, results):
        if random.random() > self.prob:
            return results

        img = results['img']
        assert img.ndim == 3 and img.shape[2] == 7, f'ShearX7 needs a 7-channel image, got {img.shape}'
        mag = random.uniform(-self.max_shear, self.max_shear)
        results['img'] = mmcv.imshear(
            img,
            mag,
            direction='horizontal',
            border_value=self.border_value)
        return results


@TRANSFORMS.register_module()
class ShearY7(BaseTransform):
    """Shear-Y a 7-channel image by a random magnitude in [-max_shear, max_shear]."""

    def __init__(self,
                 prob: float = 0.5,
                 max_shear: float = 0.3,
                 border_value: Sequence[float] = None):
        super().__init__()
        self.prob = prob
        self.max_shear = max_shear
        self.border_value = tuple(border_value) if border_value is not None else (0.0,)*7

    def transform(self, results):
        if random.random() > self.prob:
            return results

        img = results['img']
        assert img.ndim == 3 and img.shape[2] == 7, f'ShearY7 needs a 7-channel image, got {img.shape}'
        mag = random.uniform(-self.max_shear, self.max_shear)
        results['img'] = mmcv.imshear(
            img,
            mag,
            direction='vertical',
            border_value=self.border_value)
        return results


@TRANSFORMS.register_module()
class TranslateX7(BaseTransform):
    """Translate-X a 7-channel image by a random fraction of width in [-max_frac, max_frac]."""

    def __init__(self,
                 prob: float = 0.5,
                 max_frac: float = 0.2,
                 border_value: Sequence[float] = None):
        super().__init__()
        self.prob = prob
        self.max_frac = max_frac
        self.border_value = tuple(border_value) if border_value is not None else (0.0,)*7

    def transform(self, results):
        if random.random() > self.prob:
            return results

        img = results['img']
        assert img.ndim == 3 and img.shape[2] == 7, f'TranslateX7 needs a 7-channel image, got {img.shape}'
        _, w, _ = img.shape
        shift = int(random.uniform(-self.max_frac, self.max_frac) * w)
        results['img'] = mmcv.imtranslate(
            img,
            shift,
            direction='horizontal',
            border_value=self.border_value)
        return results


@TRANSFORMS.register_module()
class TranslateY7(BaseTransform):
    """Translate-Y a 7-channel image by a random fraction of height in [-max_frac, max_frac]."""

    def __init__(self,
                 prob: float = 0.5,
                 max_frac: float = 0.2,
                 border_value: Sequence[float] = None):
        super().__init__()
        self.prob = prob
        self.max_frac = max_frac
        self.border_value = tuple(border_value) if border_value is not None else (0.0,)*7

    def transform(self, results):
        if random.random() > self.prob:
            return results

        img = results['img']
        assert img.ndim == 3 and img.shape[2] == 7, f'TranslateY7 needs a 7-channel image, got {img.shape}'
        h, _, _ = img.shape
        shift = int(random.uniform(-self.max_frac, self.max_frac) * h)
        results['img'] = mmcv.imtranslate(
            img,
            shift,
            direction='vertical',
            border_value=self.border_value)
        return results
