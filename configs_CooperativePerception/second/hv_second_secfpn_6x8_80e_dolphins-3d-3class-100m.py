_base_ = [
    '../_base_/models/hv_second_secfpn_dolphins.py',
    '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/cyclic_24e.py', 
    '../_base_/default_runtime.py'
]
dataset_type = 'DolphinsDataset'
data_root = 'data/dolphins-new/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
point_cloud_range = [-100, -100, -5, 100, 100, 3]
file_client_args = dict(backend='disk')
class_names = [
    'Vehicle', 'Pedestrian', 'Cyclist'
]
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
voxel_size = [0.1, 0.1, 0.2]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4, #调试时用0
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        class_range={
                "Vehicle": 100,
                "Pedestrian": 80,
                "Cyclist": 80,
                }),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
                class_range={
                "Vehicle": 100,
                "Pedestrian": 80,
                "Cyclist": 80,
                }),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
                class_range={
                "Vehicle": 100,
                "Pedestrian": 80,
                "Cyclist": 80,
                }))
# model settings
model = dict(
    voxel_layer=dict(
        max_num_points=200,
        point_cloud_range=[-100, -100, -5, 100, 100, 3],#[0, -40, -3, 70.4, 40, 1],
        voxel_size=voxel_size,
        max_voxels=(60000, 80000)),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [-99.2, -99.2, -1.8, 99.2, 99.2, -1.4],  
                [-99.2, -99.2, -1.7, 99.2, 99.2, -1.3],
                [-99.2, -99.2, -2, 99.2, 99.2, -1.6],
            ],
            sizes=[
                [5, 2.2, 1.8],  # car
                [0.60, 0.60, 1.8],  # pedestrian
                [2.1, 0.80, 1.6],  # bicycle
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2))
            )