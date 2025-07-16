# Author: victorg

import torch

from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


@MODELS.register_module()
class ResNet7Channel(ResNet):

    def init_weights(self):
        # 1) Run MMDet’s normal init/load logic
        super().init_weights()

        # 2) Inflate conv1 if it expects 7 channels
        conv1 = self.conv1
        # weight shape: [out_ch, in_ch, k, k]
        if conv1.weight.shape[1] == 7:
            # Load the 3-channel pretrained conv1 from your checkpoint
            ckpt = torch.hub.load_state_dict_from_url(
                'https://download.openmmlab.com/mmdetection/.../resnet50_caffe.pth',
                map_location='cpu', check_hash=True)
            old_w = ckpt['conv1.weight']  # [out_ch, 3, k, k]

            # Build new 7-channel weight
            out_ch, _, k, _ = old_w.shape
            new_w = torch.zeros(out_ch, 7, k, k, dtype=old_w.dtype)

            # Copy RGB → channels 0–2
            new_w[:, 0:3, :, :] = old_w

            # Channel 3: the “grayscale” weight (mean of the 3)
            gray = old_w.mean(dim=1, keepdim=True)  # [out_ch,1,k,k]
            new_w[:, 3:4, :, :] = gray

            # Channels 4–6: repeat RGB again
            new_w[:, 4:7, :, :] = old_w

            # Overwrite conv1.weight
            conv1.weight.data.copy_(new_w)

            # 3) If conv1 has a bias, copy that too
            if conv1.bias is not None and 'conv1.bias' in ckpt:
                # old bias: [out_ch]
                old_b = ckpt['conv1.bias']
                conv1.bias.data.copy_(old_b)