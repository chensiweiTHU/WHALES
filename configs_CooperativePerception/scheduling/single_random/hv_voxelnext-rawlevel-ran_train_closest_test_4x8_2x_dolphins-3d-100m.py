point_cloud_range = [-100, -100, -5, 100, 100, 3]
voxel_size = [0.5, 0.5, 8]
_base_ = [
    '../_base_/models/voxelnext_100m.py',
    # '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
plugin = True
plugin_dir = "mmdet3d_plugin/"
class_names = [
    'Vehicle', 'Pedestrian', 'Cyclist'
]
dataset_type = 'DolphinsDataset'
data_root = 'data/whales/'
# Input modality for Dolphins2 dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='AgentScheduling',
        mode="unicast",
        submode="random", 
        basic_data_limit=6e6
        ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RawlevelPointCloudFusion'),
    # dict(type='ObjectSample', db_sampler=db_sampler),
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
    dict(type='AgentScheduling',
        mode="unicast",
        submode="closest", 
        basic_data_limit=6e6
        ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='RawlevelPointCloudFusion'),
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
    dict(type='Collect3D', keys=['points'], meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
        'depth2img', 'cam2img', 'pad_shape',
        'scale_factor', 'flip', 'pcd_horizontal_flip',
        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
        'img_norm_cfg', 'pcd_trans', 'sample_idx',
        'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
        'transformation_3d_flow',
        # new keys
        'transmitted_data_size'
        ])
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
    dict(type='AgentScheduling',
        mode="unicast",
        submode="closest", 
        basic_data_limit=6e6
        ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='LoadAnnotations3D'),
    dict(type='RawlevelPointCloudFusion'),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'], meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
        'depth2img', 'cam2img', 'pad_shape',
        'scale_factor', 'flip', 'pcd_horizontal_flip',
        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
        'img_norm_cfg', 'pcd_trans', 'sample_idx',
        'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
        'transformation_3d_flow',
        # new keys
        'transmitted_data_size'
        ])
]
# model settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4, #调试时用0
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_train.pkl',
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
        ann_file=data_root + 'whales_infos_val.pkl',
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
        ann_file=data_root + 'whales_infos_val.pkl',
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
