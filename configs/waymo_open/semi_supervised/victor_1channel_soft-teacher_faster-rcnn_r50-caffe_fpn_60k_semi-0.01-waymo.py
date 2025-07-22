# Author: victorg

_base_ = ['victor_1channel_soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-waymo.py']

# 1% waymo train2020 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2020.1@1.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2020.1@1-unlabeled.json'
train_dataloader = dict(
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 60k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=60000, val_interval=5000)

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=60000,
        by_epoch=False,
        milestones=[40000, 54000],
        gamma=0.1)
]