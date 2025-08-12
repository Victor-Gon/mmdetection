# Author: victorg

import cv2
import math
import random
import numpy as np

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class BGRTo7ChannelDarkToneCLAHE(BaseTransform):
    """
    Convert input to a 7-channel low-light augmented representation.

    Accepted inputs:
      - BGR uint8, HxWx3  → will be converted to RGB
      - 7-channel, HxWx7   → will take the first 3 channels as *RGB* baseline

    Augmentation (probability p) is applied *only* to the RGB triple. Then we
    recompute CLAHE channels (gray, R, G, B) from the (augmented|original) RGB.
    Output is HxWx7 uint8 ordered as:
      [aug-R, aug-G, aug-B, CLAHE-gray, CLAHE-R, CLAHE-G, CLAHE-B]
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
        # CLAHE params:
        clahe_clipLimit: float = 2.0,
        clahe_tileGridSize: tuple = (8, 8),
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

        self.clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clipLimit),
            tileGridSize=tuple(clahe_tileGridSize),
        )

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert float/other types to uint8, assuming either [0,1] or [0,255] range."""
        if img.dtype == np.uint8:
            return img
        img_float = img.astype(np.float32)
        # Heuristic: if values mostly in [0,1.5], treat as [0,1]
        if np.nanmax(img_float) <= 1.5:
            img_float = img_float * 255.0
        img_float = np.clip(img_float, 0.0, 255.0)
        return (img_float + 0.5).astype(np.uint8)

    def _extract_rgb_u8(self, img: np.ndarray) -> np.ndarray:
        """
        Returns an RGB uint8 HxWx3 from either:
          - BGR HxWx3
          - 7-ch HxWx7 (first 3 channels interpreted as RGB)
        """
        assert img.ndim == 3, f'Expected HxWxC, got shape={img.shape}'
        C = img.shape[2]
        if C == 3:
            # BGR -> RGB
            bgr_u8 = self._to_uint8(img)
            return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
        elif C == 7:
            # Take first 3 as RGB
            rgb = img[..., :3]
            return self._to_uint8(rgb)
        else:
            raise AssertionError(f'Expected 3 or 7 channels, got C={C}')

    @staticmethod
    def _luminance_mean(rgb_f32: np.ndarray) -> float:
        # rgb_f32 is HxWx3 in [0,1]
        lum = 0.299 * rgb_f32[..., 0] + 0.587 * rgb_f32[..., 1] + 0.114 * rgb_f32[..., 2]
        return float(lum.mean())

    def _augment_rgb(self, rgb_f32: np.ndarray) -> np.ndarray:
        """Apply the low-light chain to RGB in [0,1] float32."""
        # 2) gamma darkening
        gamma = random.uniform(*self.gamma_range)
        rgb = np.power(np.clip(rgb_f32, 0.0, 1.0), 1.0 / gamma, dtype=np.float32)

        # 3) brightness crush
        alpha = random.uniform(*self.brightness_range)
        rgb = rgb * alpha

        # 4) per-channel tone shift
        rs = random.uniform(*self.red_scale)
        gs = random.uniform(*self.green_scale)
        bs = random.uniform(*self.blue_scale)
        rgb = rgb * np.array([rs, gs, bs], dtype=np.float32).reshape(1, 1, 3)

        # 5) saturation reduction
        sat_f = random.uniform(*self.saturation_range)
        gray = rgb.mean(axis=2, keepdims=True)
        rgb = gray + (rgb - gray) * sat_f

        # 6) add Gaussian noise
        noise_std = random.uniform(*self.noise_std_range)
        if noise_std > 0:
            rgb = rgb + np.random.normal(0.0, noise_std, size=rgb.shape).astype(np.float32)

        return np.clip(rgb, 0.0, 1.0)

    def _clahe_from_rgb(self, rgb_u8: np.ndarray):
        """Return (gray_clahe, r_clahe, g_clahe, b_clahe) as uint8 HxW arrays."""
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        gray_c = self.clahe.apply(gray)
        r_c = self.clahe.apply(rgb_u8[..., 0])
        g_c = self.clahe.apply(rgb_u8[..., 1])
        b_c = self.clahe.apply(rgb_u8[..., 2])
        return gray_c, r_c, g_c, b_c

    @staticmethod
    def _to_u8_from_f32(rgb_f32: np.ndarray) -> np.ndarray:
        return np.clip(rgb_f32 * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def transform(self, results):
        img = results['img']

        # Always extract an RGB uint8 baseline (supports 3- or 7-channel input)
        rgb_u8 = self._extract_rgb_u8(img)
        rgb_f32 = (rgb_u8.astype(np.float32) / 255.0)

        # Decide whether to run augmentation
        do_aug = (random.random() <= self.p)
        is_super_dark = self._luminance_mean(rgb_f32) < self.dark_thresh

        if do_aug and not is_super_dark:
            rgb_f32_aug = self._augment_rgb(rgb_f32)
            rgb_u8_aug = self._to_u8_from_f32(rgb_f32_aug)
        else:
            rgb_u8_aug = rgb_u8  # no-op path

        # Recompute CLAHE channels from the (augmented or original) RGB
        gray_c, r_c, g_c, b_c = self._clahe_from_rgb(rgb_u8_aug)

        # Stack into HxWx7 uint8 ordered as:
        # [aug-R, aug-G, aug-B, CLAHE-gray, CLAHE-R, CLAHE-G, CLAHE-B]
        combined = np.dstack([
            rgb_u8_aug[..., 0],
            rgb_u8_aug[..., 1],
            rgb_u8_aug[..., 2],
            gray_c, r_c, g_c, b_c
        ]).astype(np.uint8)

        results['img'] = combined
        return results