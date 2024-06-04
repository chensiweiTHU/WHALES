dataset_type = 'V2XDataset'
plugin = True
plugin_dir = "mmdet3d_plugin/"
data_root = './data/DAIR-V2X-C/cooperative-vehicle-infrastructure/'
data_info_train_path = './data_process/dairv2x/flow_data_jsons/flow_data_info_train.json'
data_info_val_path = './data_process/dairv2x/flow_data_jsons/flow_data_info_val_0.json'
# work_dir = './ffnet_work_dir/work_dir_baseline'

class_names = ['Car', 'others']
point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.04, 0.04, 0.0625]
l = int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])
h = int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])
output_shape = [h, l]
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -2.66

# voxel_size=[0.2, 0.2, 0.5]
num_point_features=4
# point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
grid_size = [2304, 2304, 64]

model = dict(
    type='VoxelNeXtCoopertive',
    pts_voxel_layer=dict(
        max_num_points=1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(240000, 320000)),
    pts_voxel_encoder=dict(
            type='MeanVFE',
            num_point_features=num_point_features
            ),
    sps_quantisizer=dict(
        type='SPSQuantisizer',
    ),
    backbone_3d=dict(
        type='VoxelResBackBone8xVoxelNeXtSPS',
        input_channels = num_point_features,
        grid_size = grid_size,
        spconv_kernel_sizes=[3, 3, 3, 3], 
        channels=[16, 32, 64, 128, 128], 
        out_channel=256,
        ),
    fusion_channels=[512,384,256],
    dense_head=dict(
        type='VoxelNeXtHead',
        model_cfg=dict(
        class_agnostic=False,
        input_features=256,
        class_names_each_head=[['Car','others']],#[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
        shared_conv_channel=256,
        kernel_size_head=3,
        use_bias_before_norm=True,
        num_hm_conv=2,
        separate_head_cfg=dict(
            head_order=['center', 'center_z', 'dim', 'rot'],#, 'vel'],
            head_dict={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                # 'vel': {'out_channels': 2, 'num_conv': 2},
            }
        ),
        target_assigner_config=dict(
            feature_map_stride=8,
            num_max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2
        ),
        loss_config=dict(
            loss_weights={'cls_weight': 1.0, 'loc_weight': 0.25, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]}#, 1.0, 1.0]}
        ),
        post_processing=dict(
            score_thresh=0.1,
            post_center_limit_range=[-15.12, -61.2, -4, 107.28, 61.2, 2],
            max_obj_per_sample=500,
            nms_config=dict(
                nms_type='nms_gpu',
                nms_thresh=0.1,
                nms_pre_maxsize=1000,
                nms_post_maxsize=150
            )
        )
    ),
    input_channels = 128,
    num_class=2,
    class_names = class_names,
    grid_size = grid_size,
    point_cloud_range = point_cloud_range,
    voxel_size = voxel_size,
    bbox_coder=dict(
        type='CenterPointBBoxCoder',
        pc_range=point_cloud_range,
        post_center_range=[-15.12, -61.2, -4, 107.28, 61.2, 2],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=9),
    ),
    post_processing=dict(
        recall_thresh_list=[0.3, 0.5, 0.7],
        eval_metric='kitti'
    ),
    proj_first=False,
    single=False
)
file_client_args = dict(backend='disk')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_info_train_path,
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    sensor_view='vehicle'),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    sensor_view='infrastructure'),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(type='ProjectCooperativePCD2ego'),
                # dict(
                #     type='ObjectSample',
                #     db_sampler=dict(
                #         data_root=data_root,
                #         info_path=data_root + 'kitti_dbinfos_train.pkl',
                #         rate=1.0,
                #         prepare=dict(
                #             filter_by_difficulty=[-1],
                #             filter_by_min_points=dict(
                #                 Car=5)),
                #         classes=class_names,
                #         sample_groups=dict(Car=15))),
                # dict(
                #     type='ObjectNoise',
                #     num_try=100,
                #     translation_std=[0.25, 0.25, 0.25],
                #     global_rot_range=[0.0, 0.0],
                #     rot_range=[-0.15707963267, 0.15707963267]),
                # dict(type='GlobalTransCP',trans_factor=[-46.08,0,0]),
                dict(
                    type='GlobalRotScaleTransCP',
                    #rot_range=[-0.78539816, 0.78539816], #go to nan when inf no point
                    # rot_range=[-0.38269908, 0.38269908],
                    # scale_ratio_range=[0.95, 1.05]),
                    rot_range=[-0, 0],
                    scale_ratio_range=[1, 1]),
                dict(type='RandomFlip3DCP', flip_ratio_bev_horizontal=0.5),
                # dict(type='GlobalTransCP',trans_factor=[+46.08,0,0]),
                dict(
                    type='PointsRangeFilterCP',
                    point_cloud_range=point_cloud_range),
                dict(type='PointShuffleCP'),
                # dict(
                #     type='PointQuantization',
                #     voxel_size = voxel_size,
                #     quantize_coords_range = point_cloud_range,
                #     ),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=point_cloud_range),
                # dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=class_names),
                dict(
                    type='Collect3D',
                    keys=['points', 'infrastructure_points', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                               'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                               'img_norm_cfg', 'pcd_trans', 'sample_idx',
                               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'inf2veh')
                )
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='vehicle'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='infrastructure'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(h, l),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    # dict(
                    #     type='GlobalRotScaleTrans',
                    #     rot_range=[0, 0],
                    #     scale_ratio_range=[1.0, 1.0],
                    #     translation_std=[0, 0, 0]),
                    # dict(type='RandomFlip3D'),
                    # dict(
                    #     type='PointsRangeFilter',
                    #     point_cloud_range=point_cloud_range),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'infrastructure_points'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                               'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                               'img_norm_cfg', 'pcd_trans', 'sample_idx',
                               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'inf2veh'))
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        split='testing',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='vehicle'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='infrastructure'),
            # dict(type='ProjectCooperativePCD2ego'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(h, l),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    # dict(
                    #     type='GlobalRotScaleTrans',
                    #     rot_range=[0, 0],
                    #     scale_ratio_range=[1.0, 1.0],
                    #     translation_std=[0, 0, 0]),
                    # dict(type='RandomFlip3D'),
                    # dict(
                    #     type='PointsRangeFilter',
                    #     point_cloud_range=point_cloud_range),
                    # dict(
                    # type='PointQuantization',
                    # voxel_size = voxel_size,
                    # quantize_coords_range = point_cloud_range,
                    # ),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'infrastructure_points'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                               'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                               'img_norm_cfg', 'pcd_trans', 'sample_idx',
                               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'inf2veh'))
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=100,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
