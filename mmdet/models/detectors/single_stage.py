import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ldcnet.model import ENet, LDCNet
from ldcnet.CoordConv import AddCoordsNp
from ldcnet.data import load_calib
from torchvision import transforms

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

img_h, img_w = 1080, 1920

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
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if(model_type == "LDCNet"):
            self.fusion_model = LDCNet(img_h,img_w)
            self.fusion_model.load_state_dict(torch.load(model_path))
            self.fusion_model.eval()
        elif(model_type == "ENet"):
            self.fusion_model = ENet(img_h,img_w)
            self.fusion_model.load_state_dict(torch.load(model_path))
            self.fusion_model.eval()

        # Early fusion
        if(fusion_type == "Early"):
            self.conv1x1 = torch.nn.Conv2d(4, 3, 1) 
            self.relu = torch.nn.ReLU() 

        # Middle fusion
        elif(fusion_type == "Middle"):
            self.relu = torch.nn.ReLU() 
            self.convy = torch.nn.Conv2d(1, 3, 1) 
            self.convfpn = [torch.nn.Conv2d(512, 256, 1).cuda().half() for _ in range(5)]

            self.convfpn0 = torch.nn.Conv2d(512, 256, 1)
            self.convfpn1 = torch.nn.Conv2d(512, 256, 1)   
            self.convfpn2 = torch.nn.Conv2d(512, 256, 1)
            self.convfpn3 = torch.nn.Conv2d(512, 256, 1)
            self.convfpn4 = torch.nn.Conv2d(512, 256, 1)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            y = self.relu(self.convy(y))
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

        # else:
        #     x = self.backbone(img)
        #     if self.with_neck:
        #         x = self.neck(x)

        #     z = x

        return z


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
