_base_ = [
    'mmrotate::_base_/models/oriented-rcnn-le90_r50_fpn.py',
    'mmrotate::_base_/default_runtime.py',
    'vild_visdronezsd.py'
]

work_dir = 'work_dirs/vild_oriented-rcnn_r50_fpn_visdronezsd_step2_finetune'

custom_imports = dict(
    imports=['projects.ViLD.vild'], allow_failed_imports=False)

batch_size = 12
num_workers = 2
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
)

test_dataloader = dict(
    dataset=dict(
        ann_file='ImageSets/Main/test.txt',
    )
)

model = dict(
    type='RotatedViLD',
    data_preprocessor = dict(
        type='mmdet.DetDataPreprocessor',
        mean=[122.7709383 , 116.7460125 , 104.09373615],
        std=[68.5005327 , 66.6321579 , 70.32316305],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    visual=dict(
        type='ModifiedResNet2',
        layers=[3, 4, 6, 3],
        width=64,
        output_dim=1024,
        heads=32,
        image_size=224,
    ),
    pseudo_cfg=dict(
        semi_weight=0.5,    # semi branch
        vector_path="projects/CastDetv2/resources/remoteCLIP_embeddings_normalized.npy",
        proposal_path='work_dirs/vild_oriented-rcnn_r50_fpn_visdronezsd_step1_prepare/proposals_300',
        pseudo_nms=True,
        iou_threshold=0.6,
        pre_keep=1000,
        post_keep=300,
        initialize=False,
        mini_batch_size=128,
        filter_empty_instances=True
    ),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHeadZSD',
            num_classes=20,
            fc_cls=dict(
                    type='Projection2',
                    vector_path="projects/CastDetv2/resources/remoteCLIP_embeddings_normalized.npy",
                    is_scale=True,
                    is_grad_bg=True,
                    is_grad=False
                ),
        ),
    )
)

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=10000, val_interval=2000)
val_cfg = dict(type='ValLoop')
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
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'visual': dict(decay_mult=0., lr_mult=0.)
        },
        norm_decay_mult=0.)
    )


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])

load_from = "work_dirs/oriented-rcnn_r50-fpn_20k_visdronezsd_base-set/merged_vild_init_iter20k.pth"