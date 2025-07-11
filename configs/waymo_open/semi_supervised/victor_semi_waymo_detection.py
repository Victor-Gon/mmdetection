# Author: victorg

dataset_type = 'WaymoOpenDataset'
data_root = '/mnt/hd/victorg/workspace/mmdetection/.data_victor/'

backend_args = None

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

# scale = [(1333, 400), (1333, 1200)] # Coco config
scale= [(640, 960), (1280, 1920)] # Two tuples that indicate lower and upper bound of image scale for random resizing

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 1920), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 1
num_workers = 5
# There are two common semi-supervised learning settings on the coco dataset：
# (1) Divide the train2017 into labeled and unlabeled datasets
# by a fixed percentage, such as 1%, 2%, 5% and 10%.
# The format of labeled_ann_file and unlabeled_ann_file are
# instances_train2017.{fold}@{percent}.json, and
# instances_train2017.{fold}@{percent}-unlabeled.json
# `fold` is used for cross-validation, and `percent` represents
# the proportion of labeled data in the train2017.
# (2) Choose the train2017 as the labeled dataset
# and unlabeled2017 as the unlabeled dataset.
# The labeled_ann_file and unlabeled_ann_file are
# instances_train2017.json and image_info_unlabeled2017.json
# We use this configuration by default.

# Paths for new annotations
labeled_annotation_file = '/mnt/hd/victorg/workspace/mmdetection/.data_victor/annotations/labeled_10.json'
unlabeled_annotation_file = '/mnt/hd/victorg/workspace/mmdetection/.data_victor/annotations/unlabeled_90.json'
val_annotation_file = '/mnt/hd/victorg/workspace/mmdetection/.data_victor/annotations/instances_val2020_split.json'

# The data_prefixes are used for images of the corresponding splits
data_prefix_train = '/mnt/hd/waymococo_f0/train2020/'  # All images for training come from this path

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='/mnt/hd/victorg/workspace/mmdetection/.data_victor/annotations/labeled_10.json',  # Correct file
    data_prefix=dict(img='/mnt/hd/waymococo_f0/train2020/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args
)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='/mnt/hd/victorg/workspace/mmdetection/.data_victor/annotations/unlabeled_90.json',  # Correct file
    data_prefix=dict(img='/mnt/hd/waymococo_f0/train2020/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args
)

# Validation Dataset Configuration
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_annotation_file,
        data_prefix=dict(img=data_prefix_train),  # Use the same image folder for validation
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# Test Dataloader (same as val)
test_dataloader = val_dataloader

# Training Dataloader Configuration (same as before)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]),  # ratio of labeled to unlabeled data
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset])
)

# Evaluation Metrics (same as before)
val_evaluator = dict(
    type='WaymoMetric',
    ann_file=data_root + 'annotations/instances_val2020_split.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)

test_evaluator = val_evaluator