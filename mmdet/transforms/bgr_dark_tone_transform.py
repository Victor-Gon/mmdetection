# Author: victorg

import cv2
import math
import random
import numpy as np

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class BGRTo3ChannelDarkTone(BaseTransform):
    """
    Convert BGR input to a low-light augmented BGR output.

    Pipeline:
      1) Read BGR (HxWx3) and convert to RGB uint8.
      2) With prob p (and if not super dark), apply low-light chain:
         - gamma darkening, brightness crush, per-channel scaling,
           saturation reduction, Gaussian noise.
      3) Convert back to BGR uint8 and return.

    Args:
        p (float): probability to apply augmentation if image is not super dark.
        gamma_range, brightness_range, red/green/blue_scale, saturation_range,
        noise_std_range: parameter ranges for the augmentation chain.
        dark_thresh (float): mean luminance threshold (in [0,1]) below which
            augmentation is skipped.
    """

    def __init__(
        self,
        p: float = 0.3,
        gamma_range=(1.8, 2.5),
        brightness_range=(0.25, 0.45),
        red_scale=(0.6, 0.8),
        green_scale=(0.5, 0.7),
        blue_scale=(0.4, 0.6),
        saturation_range=(0.6, 0.8),
        noise_std_range=(0.01, 0.05),
        dark_thresh: float = 0.2,
    ):
        super().__init__()
        self.p = float(p)
        self.gamma_range = tuple(gamma_range)
        self.brightness_range = tuple(brightness_range)
        self.red_scale = tuple(red_scale)
        self.green_scale = tuple(green_scale)
        self.blue_scale = tuple(blue_scale)
        self.saturation_range = tuple(saturation_range)
        self.noise_std_range = tuple(noise_std_range)
        self.dark_thresh = float(dark_thresh)

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert to uint8, accepting float in [0,1] or [0,255]."""
        if img.dtype == np.uint8:
            return img
        img_float = img.astype(np.float32)
        if np.nanmax(img_float) <= 1.5:
            img_float = img_float * 255.0
        img_float = np.clip(img_float, 0.0, 255.0)
        return (img_float + 0.5).astype(np.uint8)

    @staticmethod
    def _rgb_from_bgr_u8(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _bgr_from_rgb_u8(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _luminance_mean(rgb_f32: np.ndarray) -> float:
        lum = 0.299 * rgb_f32[..., 0] + 0.587 * rgb_f32[..., 1] + 0.114 * rgb_f32[..., 2]
        return float(lum.mean())

    def _augment_rgb(self, rgb_f32: np.ndarray) -> np.ndarray:
        """Apply the low-light chain to RGB in [0,1] float32."""
        # gamma darkening
        gamma = random.uniform(*self.gamma_range)
        rgb = np.power(np.clip(rgb_f32, 0.0, 1.0), 1.0 / gamma, dtype=np.float32)

        # brightness crush
        alpha = random.uniform(*self.brightness_range)
        rgb = rgb * alpha

        # per-channel tone shift
        rs = random.uniform(*self.red_scale)
        gs = random.uniform(*self.green_scale)
        bs = random.uniform(*self.blue_scale)
        rgb = rgb * np.array([rs, gs, bs], dtype=np.float32).reshape(1, 1, 3)

        # saturation reduction
        sat_f = random.uniform(*self.saturation_range)
        gray = rgb.mean(axis=2, keepdims=True)
        rgb = gray + (rgb - gray) * sat_f

        # Gaussian noise
        noise_std = random.uniform(*self.noise_std_range)
        if noise_std > 0:
            rgb = rgb + np.random.normal(0.0, noise_std, size=rgb.shape).astype(np.float32)

        return np.clip(rgb, 0.0, 1.0)

    @staticmethod
    def _to_u8_from_f32(rgb_f32: np.ndarray) -> np.ndarray:
        return np.clip(rgb_f32 * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def transform(self, results):
        bgr = results['img']
        bgr_u8 = self._to_uint8(bgr)
        rgb_u8 = self._rgb_from_bgr_u8(bgr_u8)
        rgb_f32 = rgb_u8.astype(np.float32) / 255.0

        do_aug = (random.random() <= self.p)
        is_super_dark = self._luminance_mean(rgb_f32) < self.dark_thresh

        if do_aug and not is_super_dark:
            rgb_f32_aug = self._augment_rgb(rgb_f32)
            rgb_u8_aug = self._to_u8_from_f32(rgb_f32_aug)
        else:
            rgb_u8_aug = rgb_u8  # no-op

        results['img'] = self._bgr_from_rgb_u8(rgb_u8_aug)
        return results