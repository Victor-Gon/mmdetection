# Author: victorg

import cv2
import numpy as np

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class BGRTo7ChannelWithCLAHETransform(BaseTransform):
    """Transform to convert RGB images to a 7-channel format with CLAHE applied.
    This transform applies CLAHE to both the grayscale channel and the L-channel
    of the LAB color space, and then combines these with the original RGB channels.
    The output is a numpy array with shape (H, W, 7), where the last dimension
    contains the original RGB channels, the CLAHE-enhanced grayscale channel,
    and the CLAHE-enhanced RGB channels in LAB color space.
    """
    def __init__(self):
        super().__init__()

    def _after_load_image(self, img):
        # 1) ensure we have a NumPy array
        if not isinstance(img, np.ndarray):
            img = np.array(img)  # uint8 HxWx3

        # If we've already produced a 7-channel image, do nothing
        if img.ndim == 3 and img.shape[2] == 7:
            return img

        # 2) must be 3-channel BGR
        assert img.ndim == 3 and img.shape[2] == 3, (
            f'RGBTo7ChannelWithCLAHETransform needs a 3-channel image, got {img.shape}'
        )

        # 3) BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 4) CLAHE on gray
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray_c = clahe.apply(gray)

        # 5) CLAHE on L channel of LAB
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_c = clahe.apply(l)
        lab_c = cv2.merge((l_c, a, b))
        rgb_c = cv2.cvtColor(lab_c, cv2.COLOR_LAB2RGB)

        # 6) Stack into HxWx7, uint8
        combined = np.dstack([img_rgb, gray_c, rgb_c])
        combined = combined.astype(np.uint8)
        return combined
    
    def transform(self, results):
        img = results['img']
        results['img'] = self._after_load_image(img)
        return results