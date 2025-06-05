_base_ = [
    'mmrotate::_base_/datasets/visdronezsd.py',
    'mmrotate::_base_/default_runtime.py'
]
angle_version = 'le90'
lang_model_name = 'bert-base-uncased'
batch_size = 8
num_workers = 2

custom_imports = dict(
    imports=['projects.GroundingDINO.groundingdino'], allow_failed_imports=False)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
pretrained = 'checkpoints/swin_tiny_patch4_window7_224.pth'

model = dict(
    type='RotatedGroundingDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        boxtype2tensor=False
    ),
    language_model=dict(
        type='mmdet.BertModel',
        name=lang_model_name,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0), ###
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),  ###
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='RotatedGroundingDINOHead',   ###
        num_classes=20,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.BinaryFocalLossCost', weight=2.0),
                dict(type='RBoxL1Cost', weight=5.0, box_format='xywha'),
                dict(
                    type='GDCost',
                    loss_type='kld',
                    fun='log1p',
                    tau=1,
                    sqrt=False,
                    weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

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
    batch_size=2,
    num_workers=num_workers,
    dataset=dict(
        ann_file='ImageSets/Main/test.txt',
        # data_prefix=dict(img_path='JPEGImages-trainval'),
        pipeline=val_pipeline,
        return_classes=True)
        )

# training schedule for 20k
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
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
    ]
