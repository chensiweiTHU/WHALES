# _base_ = './hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_dol-3d.py'
# model settings
_base_ = [
    '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
# model settings

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
voxel_size = [0.5, 0.5, 8]
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
model = dict(
    type='MVXFasterRCNN',
     
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-100, -100, -5, 100, 100, 3],
        voxel_size=voxel_size,
        max_voxels=(60000, 80000)),
    pts_backbone=dict(
        _delete_=True,
        type='NoStemRegNet',
        arch=dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, bot_mul=1.0),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_400mf'),
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        strides=(1, 2, 2, 2),
        base_channels=64,
        stem_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(
        type='SECONDFPN',
        _delete_=True,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 160, 384],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
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
            reshape_out=True)))
