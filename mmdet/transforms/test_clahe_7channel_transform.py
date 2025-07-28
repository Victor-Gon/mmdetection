# Author: victorg

import os
import cv2
import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class TestBGRTo7ChannelWithCLAHETransform(BaseTransform):
    """Transform to convert RGB images to a 7-channel format with CLAHE applied,
    and optionally save the visualizations to disk.
    """
    _output_count = 0
    _max_outputs = 10
    _output_dir = '/mnt/hd/victorg/workspace/mmdetection/work_dirs/victor_7channel_test/img/'

    def __init__(self):
        super().__init__()
        os.makedirs(self._output_dir, exist_ok=True)

    def _after_load_image(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.ndim == 3 and img.shape[2] == 7:
            return img

        assert img.ndim == 3 and img.shape[2] == 3, (
            f'RGBTo7ChannelWithCLAHETransform needs a 3-channel image, got {img.shape}'
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray_c = clahe.apply(gray)

        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_c = clahe.apply(l)
        lab_c = cv2.merge((l_c, a, b))
        rgb_c = cv2.cvtColor(lab_c, cv2.COLOR_LAB2RGB)

        combined = np.dstack([img_rgb, gray_c, rgb_c]).astype(np.uint8)

        # Export if limit not reached
        if self._output_count < self._max_outputs:
            index = self._output_count
            rgb = combined[:, :, 0:3]
            gray = combined[:, :, 3]
            clahe_rgb = combined[:, :, 4:7]

            cv2.imwrite(os.path.join(self._output_dir, f'{index:02d}_orig_rgb.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self._output_dir, f'{index:02d}_gray_clahe.jpg'), gray)
            cv2.imwrite(os.path.join(self._output_dir, f'{index:02d}_lab_clahe_rgb.jpg'), cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR))

            self._output_count += 1

            if self._output_count == self._max_outputs:
                print(f'[INFO] Exported {self._max_outputs} images to {self._output_dir}. You may now stop the script.')

        return combined

    def transform(self, results):
        img = results['img']
        results['img'] = self._after_load_image(img)
        return results