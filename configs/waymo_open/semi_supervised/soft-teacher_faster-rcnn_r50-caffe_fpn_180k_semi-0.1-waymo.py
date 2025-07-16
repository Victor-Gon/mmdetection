_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py', '../../_base_/default_runtime.py',
    'semi_waymo_detection.py', 
]

custom_imports = dict(
    imports=['configs.waymo_open.semi_supervised.detectors.clss_loss_safe_soft-teacher'],
    allow_failed_imports=False)

mean_bgr = [
    0.3810613051452817,
    0.34263812425871415,
    0.3169861344600267
]
mean_bgr = [x * 255 for x in mean_bgr]
std_bgr = [
    0.20917705938609538,
    0.1857455798299951,
    0.17776090025526553
]
std_bgr = [x * 255 for x in std_bgr]
std_bgr = [1.0, 1.0, 1.0]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=mean_bgr,
    std=std_bgr,
    bgr_to_rgb=False,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet',
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
        freeze_teacher=True,
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