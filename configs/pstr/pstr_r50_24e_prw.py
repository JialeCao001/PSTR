_base_ = [
    '../_base_/datasets/cuhk_detection.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='PSTR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='PSTRMapper',
        in_channels=[512, 1024, 2048],
        out_channels=256),
    bbox_head=dict(
        type='PSTRHead',
        num_query=100,
        num_person=483,
        queue_size=500,
        cat_weight=[0.5,1.0,1.0],
        num_classes=1,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='PstrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=3,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        num_levels=1,
                        embed_dims=256),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            num_levels=1,
                            embed_dims=256)
                    ],
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder1=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=1,
                return_intermediate=False,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='PartAttention',
                            num_levels=1,
                            num_points=4,
                            embed_dims=256),
                        dict(
                            type='PartAttention',
                            num_levels=1,
                            num_points=4,
                            embed_dims=256)
                    ],
                    feedforward_channels=256,
                    operation_order=('cross_attn', 'cross_attn')))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            iou_oim=dict(type='OIMCost', weight=0.2))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(667, 400), (1000, 600), (1333, 800), (1500, 900), (1666, 1000)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_ids'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1500, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# change the path of the datasetz
data_root = './data/PRW-v16.04.20/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'train_pid.json', # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        pipeline=train_pipeline),
    val=dict(
        ann_file=data_root + 'test_pid.json',  # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'test_pid.json',  # change the path of the annotation file
        img_prefix=data_root + 'frames/',
        proposal_file=data_root+'annotation/test/train_test/TestG50.mat',
        pipeline=test_pipeline)
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.2),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[19, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
