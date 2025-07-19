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
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        init_cfg=dict(type='Pretrained', checkpoint='./checkpoints/resnet50-0676ba61.pth'),
        style='pytorch'),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_3d=dict(type='OutdoorImVoxelNeck', in_channels=64, out_channels=256),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        # mean bottom height of Vehicle: -1.6576034948567357
        # mean bottom height of Pedestrian: -1.4939155503521617
        # mean size of Vehicle: [2.44746559 1.04783988 0.77427879]
        # mean size of Pedestrian: [0.19247282 0.19247282 0.85960684]
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                # [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                # [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                # [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                # [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                # [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                # [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                # [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
                [-49.6, -49.6, -1.8, 49.6, 49.6, -1.4],
                [-49.6, -49.6, -1.7, 49.6, 49.6, -1.3],
                [-49.6, -49.6, -2, 49.6, 49.6, -1.6],
            ],
            sizes=[
                [5, 2.2, 1.8],  # car
                [0.60, 0.60, 1.8],  # pedestrian
                [2.1, 0.80, 1.60],  # bicycle
                # [1.95017717, 4.60718145, 1.72270761],  # car
                # [2.44746559, 1.04783988, 0.77427879],  # car
                # [2.4560939, 6.73778078, 2.73004906],  # truck
                # [2.87427237, 12.01320693, 3.81509561],  # trailer
                # [0.60058911, 1.68452161, 1.27192197],  # bicycle
                # [0.19247282, 0.19247282, 0.85960684], # pedestrian
                # # [0.66344886, 0.7256437, 1.75748069],  # pedestrian
                # [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                # [2.49008838, 0.48578221, 0.98297065],  # barrier
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
        ranges=[[-0.16, -39.68, -3.08, 68.96, 39.68, 0.76]],
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
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

dataset_type = 'DolphinsDataset'
data_root = 'data/whales/'
class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=True, use_camera=True)
# point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='CenterCropMultiViewImage', size=(800, 448)),
    # dict(
    #     type='ResizeMultiViewImage',
    #     # img_scale=[(1173, 352), (1387, 416)],
    #     img_scale = (800, 448),
    #     # img_scale=(704, 256),
    #     keep_ratio=True,
    #     multiscale_mode='range'),
    # dict(type='ResizeMultiViewImage', img_scale=(704, 256), keep_ratio=True),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='CenterCropMultiViewImage', size=(800, 448)),
    # dict(type='ResizeMultiViewImage', img_scale=(800, 448), keep_ratio=True),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        # type='DolphinsDataset',
        # times=3,
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_train.pkl',
        # split='training',
        # pts_prefix='velodyne_reduced',
        img_prefix="",
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        # split='training',
        # pts_prefix='velodyne_reduced',
        img_prefix="",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        # split='training',
        # pts_prefix='velodyne_reduced',
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
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
