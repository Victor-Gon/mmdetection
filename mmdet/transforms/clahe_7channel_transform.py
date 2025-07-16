# Author: victorg

import cv2
import numpy as np
from PIL import Image
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
        pass
    
    def _after_load_image(self, img_bgr):
        # 2) Swap to RGB for CLAHE
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 3) CLAHE on gray
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray_c = clahe.apply(gray)                   # still uint8
        # 4) CLAHE on L channel in LAB
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_c = clahe.apply(l)
        lab_c = cv2.merge((l_c, a, b))
        rgb_c = cv2.cvtColor(lab_c, cv2.COLOR_LAB2RGB)
        # 5) Stack into (H, W, 7) uint8
        combined = np.dstack([img_rgb, gray_c, rgb_c])
        return combined

    def transform(self, results):
        pil_rgb = results['img']
        results['img'] = self._after_load_image(pil_rgb)
        return results