# Author: victorg

_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py', '../../_base_/default_runtime.py',
    'victor_1channel_semi_waymo_detection.py', 
]


# Calculated mean and std for the Dataset
# TODO: single channel mean
mean = [
    0.3169861344600267,
    0.34263812425871415,
    0.3810613051452817,
    0.4072443514665751,
    0.3777730230166431,
    0.40322751529709666,
    0.4421570344150529
]
mean = [x*255 for x in mean] 
# TODO: single channel std
std = [
    0.17776090025526553,
    0.1857455798299951,
    0.20917705938609538,
    0.20067077711641113,
    0.19846362421099906,
    0.20318028236250318,
    0.22191031409908632
]
std = [x*255 for x in std]
# std = [1, 1, 1, 1, 1, 1, 1] 

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor7Channel',
    mean=mean,
    std=std,
    bgr_to_rgb=False,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet1Channel',
    in_channels=1,
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='open-mmlab://detectron2/resnet50_caffe'))
detector.roi_head.bbox_head.num_classes=3


model = dict(
    _delete_=True,
    type='CLSLossSafeSoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=False,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% waymo train2020 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2020.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/' \
                             'instances_train2020.1@10-unlabeled.json'
unlabeled_dataset.data_prefix = dict(img='train2020/')
train_dataloader = dict(
    batch_size=5,
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    accumulative_counts=4)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]