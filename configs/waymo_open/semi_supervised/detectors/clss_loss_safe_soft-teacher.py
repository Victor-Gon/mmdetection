# Author: victorg

import copy
import torch
from torch import Tensor
from typing import Tuple

from mmdet.models.detectors.soft_teacher import SoftTeacher
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmdet.models.utils import unpack_gt_instances, filter_gt_instances
from mmdet.structures.bbox import bbox2roi, bbox_project


@MODELS.register_module()
class CLSLossSafeSoftTeacher(SoftTeacher):
    def rcnn_cls_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                          unsup_rpn_results_list: InstanceList,
                                          batch_data_samples: SampleList,
                                          batch_info: dict) -> dict:
        """Calculate classification loss from a batch of inputs and pseudo data
        samples.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            unsup_rpn_results_list (list[:obj:`InstanceData`]):
                List of region proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process.

        Returns:
            dict[str, Tensor]: A dictionary of rcnn
                classification loss components
        """
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        cls_data_samples = copy.deepcopy(batch_data_samples)
        cls_data_samples = filter_gt_instances(
            cls_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)

        outputs = unpack_gt_instances(cls_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(cls_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')
            assign_result = self.student.roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        selected_bboxes = [res.priors for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(x, rois)
        # cls_reg_targets is a tuple of labels, label_weights,
        # and bbox_targets, bbox_weights
        cls_reg_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, self.student.train_cfg.rcnn)

        selected_results_list = []
        for bboxes, data_samples, teacher_matrix, teacher_img_shape in zip(
                selected_bboxes, batch_data_samples,
                batch_info['homography_matrix'], batch_info['img_shape']):
            student_matrix = torch.tensor(
                data_samples.homography_matrix, device=teacher_matrix.device)
            homography_matrix = teacher_matrix @ student_matrix.inverse()
            projected_bboxes = bbox_project(bboxes, homography_matrix,
                                            teacher_img_shape)
            selected_results_list.append(InstanceData(bboxes=projected_bboxes))

        with torch.no_grad():
            results_list = self.teacher.roi_head.predict_bbox(
                batch_info['feat'],
                batch_info['metainfo'],
                selected_results_list,
                rcnn_test_cfg=None,
                rescale=False)
            bg_score = torch.cat(
                [results.scores[:, -1] for results in results_list])
            # cls_reg_targets[0] is labels
            neg_inds = cls_reg_targets[
                0] == self.student.roi_head.bbox_head.num_classes
            # cls_reg_targets[1] is label_weights
            cls_reg_targets[1][neg_inds] = bg_score[neg_inds].detach()

        losses = self.student.roi_head.bbox_head.loss(
            bbox_results['cls_score'], bbox_results['bbox_pred'], rois,
            *cls_reg_targets)
        # cls_reg_targets[1] is label_weights
        if 'loss_cls' in losses:
            losses['loss_cls'] = losses['loss_cls'] * len(
                cls_reg_targets[1]) / max(sum(cls_reg_targets[1]), 1.0)
        else:
            device = bbox_results['cls_score'].device
            losses['loss_cls'] = torch.tensor(0.0, device=device)

        # --- NaN filter: replace any NaN loss with zero ---
        for key, value in losses.items():
            if torch.isnan(value).any():
                losses[key] = torch.zeros_like(value)
        # ---------------------------------------------------

        return losses