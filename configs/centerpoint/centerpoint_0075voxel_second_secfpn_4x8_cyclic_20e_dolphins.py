_base_ = [
        '../_base_/models/centerpoint_01voxel_second_secfpn_dolphins.py',
        '../_base_/datasets/dolphins-3d.py',
        '../_base_/schedules/schedule_2x.py',
        '../_base_/default_runtime.py',
          ]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.075, 0.075, 0.2]
model_point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# For nuScenes we usually do 10-class detection
class_names = [
'Vehicle', 'Pedestrian', 'Cyclist'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
model = dict(
    pts_voxel_layer=dict(
        voxel_size=voxel_size, point_cloud_range=model_point_cloud_range),
    pts_middle_encoder=dict(sparse_shape=[41, 1440, 1440]),
    pts_bbox_head=dict(
        bbox_coder=dict(
            voxel_size=voxel_size[:2], pc_range=model_point_cloud_range[:2])),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])))

dataset_type = 'WhalesDataset'
data_root = 'data/whales/'
file_client_args = dict(backend='disk')

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'dolphins_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#      filter_by_min_points=dict(        Vehicle=2,
#         Pedestrian=2,
#         Cyclist=2,)),
#     classes=class_names,
#     sample_groups=dict(
#         Vehicle=2,
#         Pedestrian=2,
#         Cyclist=2,
#     ),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=[0, 1, 2, 3],
#         file_client_args=file_client_args))

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
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(
    #             type='PointsRangeFilter', point_cloud_range=point_cloud_range),
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
        # ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
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
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(interval=18, pipeline=eval_pipeline)
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 1e-4),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )
# data = dict(
#     train=dict(dataset=dict(pipeline=train_pipeline)),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
