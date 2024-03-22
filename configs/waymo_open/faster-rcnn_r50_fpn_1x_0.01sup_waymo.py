_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    'waymo_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
# data
train_dataloader = dict(batch_size=8,
                        dataset=dict(ann_file='semi_anns/instances_train2020.1@1.json'))