_base_ = [
    'mmrotate::_base_/datasets/visdronezsd.py',
    'mmrotate::_base_/default_runtime.py'
]
angle_version = 'le90'
lang_model_name = 'bert-base-uncased'
batch_size = 8
num_workers = 2

custom_imports = dict(
    imports=['projects.GLIP.glip'], allow_failed_imports=False)


model = dict(
    type='mmdet.GLIP',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN_DropBlock',
        plugin=dict(
            type='mmdet.DropBlock',
            drop_prob=0.3,
            block_size=3,
            warmup_iters=0),
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='RotatedATSSVLFusionHead',
        lang_model_name=lang_model_name,
        num_classes=20,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            ratios=[1.0],
            octave_base_scale=8,    #
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    language_model=dict(type='mmdet.BertModel', name=lang_model_name),
    train_cfg=dict(
        assigner=dict(
            type='RotatedATSSAssigner',
            topk=9,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
            type='mmdet.PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=300))

# dataset settings
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler'),
    dataset=dict(
        pipeline=train_pipeline,
        return_classes=True))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        pipeline=val_pipeline,
        return_classes=True))

# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=24,
    num_workers=num_workers,
    dataset=dict(
        _delete_=True,
        type='NWPU45Dataset',
        data_root="data/NWPU-RESISC45",
        ann_file='annotations/nwpu45_unlabeled_2.json',
        data_prefix=dict(img='NWPU-RESISC45/'),
        test_mode=True,
        backend_args=_base_.backend_args,
        pipeline=test_pipeline,
        return_classes=True)
        )

test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=False,
    outfile_prefix='work_dirs/glip_atss_r50_a_fpn_dyhead_visdronezsd_base_nwpu45_pseudo_labeling/nwpu45_pseudo_labeling_2')

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor= 1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20000,
        by_epoch=False,
        milestones=[16000, 18000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=dict(max_norm=35, norm_type=2))


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
    ]
