# Author: victorg

import cv2
import numpy as np
from PIL import Image
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class RGBTo7ChannelWithCLAHETransform(BaseTransform):
    """Transform to convert RGB images to a 7-channel format with CLAHE applied.
    This transform applies CLAHE to both the grayscale channel and the L-channel
    of the LAB color space, and then combines these with the original RGB channels.
    The output is a numpy array with shape (H, W, 7), where the last dimension
    contains the original RGB channels, the CLAHE-enhanced grayscale channel,
    and the CLAHE-enhanced RGB channels in LAB color space.
    """
    def __init__(self):
        pass
    
    def _after_load_image(self, pil_rgb: Image.Image) -> np.ndarray:
        # Convert the PIL image to a numpy array (H, W, 3)
        rgb_np = np.array(pil_rgb)

        # 1) CLAHE grayscale channel (from enhanced image)
        gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_np = clahe.apply(gray_np)  # shape (H, W)

        # 2) CLAHE-enhanced RGB via LAB (from enhanced image)
        img_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        rgb_clahe_np = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2RGB)

        # 3) Stack into final (H, W, 7): [R,G,B], CLAHE-gray, [R',G',B']
        combined = np.dstack([rgb_np, clahe_np, rgb_clahe_np]) 

        return combined

    def transform(self, results):
        pil_rgb = results['img']
        results['img'] = self._after_load_image(pil_rgb)
        return results