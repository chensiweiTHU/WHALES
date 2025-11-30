
" convert the config above into a dictionary "
plugin = True
plugin_dir = "mmdet3d_plugin/"
_base_ = [
    # '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins-3d-100m.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

dataset_type = 'WhalesDataset'
data_root = 'data/whales/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
voxel_size=[0.15, 0.15, 0.2]
num_point_features=4
point_cloud_range=[-108.0,-108.0,-5.0,108.0,108.0,3.0,]#[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
grid_size = [1440, 1440, 40]
# grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
# self.grid_size = np.round(grid_size).astype(np.int64)
model = dict(
    type='VoxelNeXt',
    pts_voxel_layer=dict(
        max_num_points=30,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000)),
    pts_voxel_encoder=dict(
            type='MeanVFE',
            num_point_features=num_point_features
            ),
    backbone_3d=dict(
        type='VoxelResBackBone8xVoxelNeXt',
        input_channels = num_point_features,
        grid_size = grid_size,
        spconv_kernel_sizes=[3, 3, 3, 3], 
        channels=[16, 32, 64, 128, 128], 
        out_channel=128,
        ),
    dense_head=dict(
        type='VoxelNeXtHead',
        model_cfg=dict(
        class_agnostic=False,
        input_features=128,
        class_names_each_head=[['Vehicle'],['Pedestrian'],['Cyclist']],#[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
        shared_conv_channel=128,
        kernel_size_head=3,
        use_bias_before_norm=True,
        num_hm_conv=2,
        separate_head_cfg=dict(
            head_order=['center', 'center_z', 'dim', 'rot', 'vel'],
            head_dict={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                'vel': {'out_channels': 2, 'num_conv': 2},
            }
        ),
        target_assigner_config=dict(
            feature_map_stride=8,
            num_max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2
        ),
        loss_config=dict(
            loss_weights={'cls_weight': 1.0, 'loc_weight': 0.25, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]}
        ),
        post_processing=dict(
            score_thresh=0.1,
            post_center_limit_range=[-122.4, -122.4, -2.0, 122.4, 122.4, 2.0],
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
    num_class=3,
    class_names = class_names,
    grid_size = grid_size,
    point_cloud_range = point_cloud_range,
    voxel_size = voxel_size,
    bbox_coder=dict(
        type='CenterPointBBoxCoder',
        pc_range=point_cloud_range,
        post_center_range=[-122.4, -122.4, -2.0, 122.4, 122.4, 2.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=9),
    loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
    loss_bbox=dict(type='L1Loss', reduction='none', loss_weight=0.25),
    train_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
    test_cfg=dict(
            post_center_limit_range=[-122.4, -122.4, -2.0, 122.4, 122.4, 2.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2),
    ),
    #     POST_PROCESSING:
#         RECALL_THRESH_LIST: [0.3, 0.5, 0.7]

#         EVAL_METRIC: kitti
    post_processing=dict(
        recall_thresh_list=[0.3, 0.5, 0.7],
        eval_metric='kitti'
    )
)
file_client_args = dict(backend='disk')
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
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(type='AgentScheduling',
    #     mode="unicast", 
    #     submode="closest", 
    #     basic_data_limit=6e6
    #     ),
    # dict(
    #     type='LoadPointsFromCooperativeAgents',
    #     coord_type='LIDAR',
    #     load_dim=4, use_dim=4,
    #     file_client_args=file_client_args
    #     ),
    dict(type='LoadAnnotations3D'),
    # dict(type='ProjectCooperativePCD2ego'),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(type='PointsRangeFilterCP', point_cloud_range=point_cloud_range),
    dict(
        type='DefaultFormatBundle3DCP',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'], 
        #  meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
        #             'depth2img', 'cam2img', 'pad_shape',
        #             'scale_factor', 'flip', 'pcd_horizontal_flip',
        #             'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
        #             'img_norm_cfg', 'pcd_trans', 'sample_idx',
        #             'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
        #             'transformation_3d_flow',
        #             # new keys
        #             'transmitted_data_size',
        #             'cooperative_agents',
        #             'ego_agent'
        #             ]
                    )
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
# runner = dict(type='EpochBasedRunner', max_epochs=36,)
# evaluation = dict(interval=6, pipeline=eval_pipeline)
# optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.001)
# lr_config = dict(policy='CosineAnnealing', warmup="linear",warmup_iters=500, min_lr=1e-7)


# optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='CosineAnnealing', warmup=500, min_lr=1e-6)
# momentum_config = None

# runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=24)