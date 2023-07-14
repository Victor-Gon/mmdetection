_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    'mamoas_detection.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# We can use the pre-trained Faster-RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# Enable automatic-mixed precision to reduce computational cost
# optim_wrapper = dict(type='AmpOptimWrapper')

# Other things to adjust:
# Input resize (maybe multi-scale training/testing)
# Anchors size (https://github.com/open-mmlab/mmdetection/issues/3669)