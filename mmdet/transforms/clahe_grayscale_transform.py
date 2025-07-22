# Author: victorg

import cv2
import numpy as np

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class BGRToCLAHEGrayscaleTransform(BaseTransform):
    """Apply CLAHE on a grayscale version of the image.

    - If input is BGR/RGB (HxWx3), it will be converted to GRAY first.
    - If input is already single-channel (HxW or HxWx1), CLAHE is applied directly.
    - By default, keeps a channel dimension (HxWx1) so downstream code expecting
      `img.shape[2]` still works. Set `keep_dim=False` to return HxW.

    Args:
        clip_limit (float): CLAHE clip limit.
        tile_grid_size (tuple[int, int]): Tile size for CLAHE.
        keep_dim (bool): Keep channel dim (HxWx1). Default True.
    """

    def __init__(self,
                 clip_limit: float = 2.0,
                 tile_grid_size=(8, 8),
                 keep_dim: bool = True):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.keep_dim = keep_dim

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:  # already gray
            return img
        if img.ndim == 3 and img.shape[2] == 1:  # HxWx1
            return img.squeeze(-1)
        if img.ndim == 3 and img.shape[2] == 3:
            # assume BGR (MMDet default). Change to COLOR_RGB2GRAY if needed.
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise AssertionError(f'Unsupported image shape {img.shape}')
    
    def _after_load_image(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        gray = self._to_gray(img)

        clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
        img_eq = clahe.apply(gray)

        img_eq = img_eq.astype(np.uint8)

        if self.keep_dim:
            img_eq = img_eq[:, :, None]

        return img_eq

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self._after_load_image(img)
        return results