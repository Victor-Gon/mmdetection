# Author: victorg (single-channel variant)

import torch
import warnings

from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


@MODELS.register_module()
class ResNet1Channel(ResNet):
    """ResNet with a 1-channel stem. conv1 weights are created by averaging the
    pretrained 3-channel (BGR) weights: GRAY = mean(B, G, R)."""

    def init_weights(self):
        # 1) Let MMEngine load everything it can (will complain about conv1)
        try:
            super().init_weights()
        except Exception as e:
            msg = str(e)
            if 'conv1.weight' in msg and 'size mismatch' in msg:
                warnings.warn(
                    'Skipping pretrained conv1.weight load (3→1 mismatch). '
                    'It will be inflated manually.'
                )
            else:
                raise

        conv1 = self.conv1
        out_ch, in_ch, k, _ = conv1.weight.shape
        if in_ch != 1:
            warnings.warn(f'conv1 expects {in_ch} input channels, not 1. '
                          'Inflation skipped.')
            return

        print('Inflating conv1 to [GRAY] from BGR pretrained weights (mean of 3 channels).')

        with torch.no_grad():
            # We assume the checkpoint would have provided [:, :3, ...] if shapes matched
            # or that super().init_weights() left random weights there—so grab them anyway.
            old_w = conv1.weight.data[:, :3, :, :].clone()  # [out_ch, 3, k, k]

            # Mean across channel dim (promediate B,G,R)
            gray = old_w.mean(dim=1, keepdim=True)          # [out_ch, 1, k, k]

            conv1.weight.data.copy_(gray)