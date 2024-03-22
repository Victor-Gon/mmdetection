_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    'waymo_detection.py',
    '../_base_/default_runtime.py'
]
# model
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
# data
train_dataloader = dict(batch_size=8,
                        dataset=dict(ann_file='semi_anns/instances_train2020.1@1.json'))

# training schedule for 60k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=60000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)