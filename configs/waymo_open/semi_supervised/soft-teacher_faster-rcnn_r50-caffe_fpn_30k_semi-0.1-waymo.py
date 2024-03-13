_base_ = ['soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-waymo.py']

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