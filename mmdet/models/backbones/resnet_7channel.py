# Author: victorg

import torch
import warnings

from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


@MODELS.register_module()
class ResNet7Channel(ResNet):

    def init_weights(self):
        """Initialize via MMEngine (ignoring conv1 mismatch), then inflate conv1."""
        # 1) Run the base init (this applies all init_cfg, pretrained for other layers)
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
                # re-raise anything unexpected
                raise

        print('Inflating conv1 layer to 7 channels (from 3) using the mean of the first 3 channels.')

        # 2) Inflate conv1 from the 3-channel weights already in memory
        conv1 = self.conv1
        out_ch, in_ch, k, _ = conv1.weight.shape
        if in_ch == 7:
            # assume super().init_weights() loaded the 3-channel stem into [:, :3, …]
            old_w = conv1.weight.data[:, :3, :, :].clone()

            # build the new 7-channel kernel
            new_w = torch.zeros((out_ch, 7, k, k),
                                dtype=old_w.dtype,
                                device=old_w.device)
            new_w[:, 0:3, :, :] = old_w
            new_w[:, 3:4, :, :] = old_w.mean(dim=1, keepdim=True)
            new_w[:, 4:7, :, :] = old_w

            # overwrite
            conv1.weight.data.copy_(new_w)