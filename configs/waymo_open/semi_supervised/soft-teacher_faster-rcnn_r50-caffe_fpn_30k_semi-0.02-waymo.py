_base_ = ['soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-waymo.py']

# 2% waymo train2020 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2020.1@2.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2020.1@2-unlabeled.json'
train_dataloader = dict(
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 30k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=30000, val_interval=5000)

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=30000,
        by_epoch=False,
        milestones=[20000, 26000],
        gamma=0.1)
]