custom_imports = dict(
    imports=['mmdet3d_plugin.models', 'mmdet3d_plugin.datasets'],
    allow_failed_imports=False)

# WHALES setup -------------------------------------------------------------
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.2, 0.2, 8]
class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
num_classes = len(class_names)
num_views = 4

# 1080x1920 -> 640x1120 to fit memory; padded to multiple of 32 (640x1120).
img_scale = (1120, 640)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# BEVFormer hyperparameters ------------------------------------------------
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=False,
    can_bus_in_dataset=False,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PerceptionTransformer',
            num_cams=num_views,
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            can_bus_in_dataset=False,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=num_views,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoderBEVFormer',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3DBEVFormer',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1CostBEVFormer', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

# Dataset & pipelines ------------------------------------------------------
dataset_type = 'WhalesDataset'
data_root = 'data/whales/'

train_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeMultiViewImage', img_scale=img_scale, keep_ratio=True),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='ResizeMultiViewImage', img_scale=img_scale,
                 keep_ratio=True),
            dict(type='NormalizeMultiViewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=class_names,
                 with_label=False),
            dict(type='Collect3D', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_train.pkl',
        img_prefix='',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        img_prefix='',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        img_prefix='',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

# Optimizer / schedule -----------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Logging / runtime --------------------------------------------------------
checkpoint_config = dict(interval=1, max_keep_ckpts=2, save_last=True)
evaluation = dict(interval=1, save_best='pts_bbox_NuScenes/mAP',
                  rule='greater')
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
