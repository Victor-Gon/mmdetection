# Author: victorg (modified)

import torch
import warnings

from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


@MODELS.register_module()
class ResNet7Channel(ResNet):

    def init_weights(self):
        """Initialize via MMEngine (ignore conv1 mismatch), then inflate conv1 to
        7 channels ordered as R, G, B, GRAY, R, G, B from a BGR pretrained stem.
        """
        # 1) Run the base init (other layers get pretrained weights)
        try:
            super().init_weights()
        except Exception as e:
            msg = str(e)
            # catch only the size-mismatch for conv1.weight
            if 'conv1.weight' in msg and 'size mismatch' in msg:
                warnings.warn(
                    'Skipping pretrained conv1.weight load (3→7 mismatch). '
                    'It will be inflated manually.'
                )
            else:
                raise

        # 2) Inflate conv1 from the 3-channel weights already in memory
        conv1 = self.conv1
        out_ch, in_ch, k, _ = conv1.weight.shape

        if in_ch != 7:
            warnings.warn(f'conv1 expects {in_ch} input channels, not 7. '
                          'Inflation skipped.')
            return

        print('Inflating conv1 to [R, G, B, GRAY, R, G, B] from BGR pretrained weights.')

        with torch.no_grad():
            # First 3 channels are BGR in the pretrained weights
            old_w = conv1.weight.data[:, :3, :, :].clone()  # shape: [out_ch, 3, k, k]
            # Reorder BGR -> RGB
            rgb = old_w[:, [2, 1, 0], :, :]  # [R, G, B]

            gray = rgb.mean(dim=1, keepdim=True)

            # Concatenate: R,G,B,GRAY,R,G,B  -> total 7
            new_w = torch.cat([rgb, gray, rgb], dim=1)  # [out_ch, 7, k, k]
            assert new_w.shape[1] == 7

            conv1.weight.data.copy_(new_w)