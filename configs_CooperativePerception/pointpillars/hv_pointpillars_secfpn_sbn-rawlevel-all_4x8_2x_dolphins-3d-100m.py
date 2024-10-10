_base_ = [
    '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
dataset_type = 'DolphinsDataset'
data_root = 'data/dolphins-new/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
voxel_size = [0.5, 0.5, 8]
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
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='RawlevelPointCloudFusion'),
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
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='RawlevelPointCloudFusion'),
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
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='RawlevelPointCloudFusion'),
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
    samples_per_gpu=2,
    workers_per_gpu=2, #调试时用0
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
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-100, -100, -5, 100, 100, 3],
        voxel_size=voxel_size,
        max_voxels=(60000, 80000)),
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-99.2, -99.2, -1.8, 99.2, 99.2, -1.4],
                # [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                # [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                
                [-99.2, -99.2, -1.7, 99.2, 99.2, -1.3],
                [-99.2, -99.2, -2, 99.2, 99.2, -1.6],

                # [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                # [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
            ],
            sizes=[
                [5.0, 2.2, 1.8],  # vehicle
                # [2.4560939, 6.73778078, 2.73004906],  # truck
                # [2.87427237, 12.01320693, 3.81509561],  # trailer
                
                [0.60, 0.60, 1.80],  # pedestrian
                [2.1, 0.80, 1.60],  # cyclist
                # [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                # [2.49008838, 0.48578221, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)))
optimizer = dict(type='AdamW', lr=5e-4, weight_decay=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=18,)
