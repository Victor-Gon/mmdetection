import time
import warnings

import torch
import torch.nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ldcnet.model import LDCNet, ENet
from ldcnet.CoordConv import AddCoordsNp
from ldcnet.data import load_calib
from torchvision import transforms

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

img_h, img_w = 1280, 1920

# Select model
model_type = "LDCNet"
# model_type = "ENet"
# model_type = None

model_path = "ldcnet/results/ldcnet_12epochs/model_Best.pth"

# Select fusion
fusion_type = "Early"
# fusion_type = "Middle"

# [R, G, B, Dense Lidar]
mean=[123.675, 116.28, 103.53, 18.411]
std=[58.395, 57.12, 57.375, 16.634]


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if(model_type == "LDCNet"):
            self.fusion_model = LDCNet(img_h,img_w)
            self.fusion_model.load_state_dict(torch.load(model_path))
            self.fusion_model.eval()
        elif(model_type == "ENet"):
            self.fusion_model = ENet(img_h,img_w)
            self.fusion_model.load_state_dict(torch.load(model_path))
            self.fusion_model.eval()

        if(fusion_type == "Early"):
            self.convy = torch.nn.Conv2d(1, 3, 1) 
            self.convfpn = torch.nn.Conv2d(512, 256, 1)
            self.conv1x1 = torch.nn.Conv2d(4, 3, 1) 
            self.relu = torch.nn.ReLU()

        elif(fusion_type == "Middle"):
            self.relu = torch.nn.ReLU() 
            self.convy = torch.nn.Conv2d(1, 3, 1) 
        
        # RetinaNet
        self.convfpn0 = torch.nn.Conv2d(512, 256, 1)
        self.convfpn1 = torch.nn.Conv2d(512, 256, 1)   
        self.convfpn2 = torch.nn.Conv2d(512, 256, 1)
        self.convfpn3 = torch.nn.Conv2d(512, 256, 1)
        self.convfpn4 = torch.nn.Conv2d(512, 256, 1)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        z = None

        with torch.no_grad():

            if(model_type == "LDCNet"):
                img[:,3,:,:] = img[:,3,:,:] / 80.
                img_h2, img_w2 = img.shape[2], img.shape[3]
                K = load_calib()
                position = AddCoordsNp(img_h2, img_w2)
                position = position.call()

                args = {"position": to_float_tensor(position).view(-1, 2, img_h2, img_w2).to(device), "K": torch.tensor(K).view(-1, 3, 3).to(device)}
                self.fusion_model.to(device)
                batch_features = torch.tensor(img).view(-1, 4, img_h2, img_w2).to(device).float()
                img[:,3:,:,:] = self.fusion_model(batch_features, args)
        
            elif(model_type == "ENet"):
                img[:,3,:,:] = img[:,3,:,:] / 80.
                img_h2, img_w2 = img.shape[2], img.shape[3]
                self.fusion_model.to(device)
                K = load_calib()
                position = AddCoordsNp(img_h2, img_w2)
                position = position.call()

                args = {"position": to_float_tensor(position).view(-1, 2, img_h2, img_w2).to(device), "K": torch.tensor(K).view(-1, 3, 3).to(device)}
                batch_features = torch.tensor(img).view(-1, 4, img_h2, img_w2).to(device).float()
                img[:,3:,:,:] = self.fusion_model(batch_features, args)[2]


        # Normalization
        for i in range(4):
            img[:,i,:,:] = (img[:,i,:,:] - mean[i]) / std[i]

        # Early fusion
        if(fusion_type == "Early"):
            img = self.conv1x1(img)
            img = self.relu(img) 
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)

            z = x

        # Middle fusion II
        elif(fusion_type == "Middle"):
            x = self.backbone(img[:,0:3,:,:])
            y = self.relu(self.convy(img[:,3:,:,:]))
            y = self.backbone(y)

            if self.with_neck:
                x = self.neck(x)
                y = self.neck(y)

            # z = ()
            # for i in range(len(x)):
            #     # w = torch.cat((x[i], torch.mul(y[i], 1.0/(i+1))),1)
            #     w = torch.cat((x[i], y[i]),1)
            #     z = z + (self.relu(self.convfpn[i](w)),)
            
            z = ()
            w = torch.cat((x[0], y[0]),1)
            z = z + (self.relu(self.convfpn0(w)),)

            w = torch.cat((x[1], y[1]),1)
            z = z + (self.relu(self.convfpn1(w)),)

            w = torch.cat((x[2], y[2]),1)
            z = z + (self.relu(self.convfpn2(w)),)
    
            w = torch.cat((x[3], y[3]),1)
            z = z + (self.relu(self.convfpn3(w)),)

            w = torch.cat((x[4], y[4]),1)
            z = z + (self.relu(self.convfpn4(w)),)

        return z

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'

        # from time import time
        # img_, img_m = img[:2], img_metas[:2]
        # t = time()
        # x = self.extract_feat(img_)
        # print(time()-t)
        # if proposals is None:
        #     proposal_list = self.rpn_head.simple_test_rpn(x, img_m)
        #     print(time() - t)
        # else:
        #     proposal_list = proposals
        #
        # x = self.roi_head.simple_test(
        #     x, proposal_list, img_m, rescale=rescale)
        # print(time()-t)


        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        return self.roi_head.onnx_export(x, proposals, img_metas)
