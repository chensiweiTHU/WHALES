model = dict(
    type='ImVoxelNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_3d=dict(type='OutdoorImVoxelNeck', in_channels=64, out_channels=256),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.8, 49.6, 49.6, -1.40],
                [-49.6, -49.6, -1.7, 49.6, 49.6, -1.3],
                [-49.6, -49.6, -2.2, 49.6, 49.6, -1.4],
            ],
            sizes=[
                [4.90, 2.13, 1.51],  # vehicle
                [0.60, 0.60, 1.8],  # pedestrian
                [2.1, 0.80, 2.2],  # bicycle
            ],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    n_voxels=[216, 248, 12],
    anchor_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[-49.6, -49.6, -5, 49.6, 49.6, 3]],
        rotations=[.0]),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.2,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

dataset_type = 'WhalesDataset'
data_root = 'data/whales/'
class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=True, use_camera=True)
point_cloud_range = [-50, -50, -5, 50, 50, 3]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=[0,1,2]),
    dict(type='FilterOccludedObjects', ds=4, depth_margin=0.5),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_train.pkl',
        img_prefix="",
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        img_prefix="",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        img_prefix="",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
lr_config = dict(policy='step', step=[8, 11])
total_epochs = 24

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
